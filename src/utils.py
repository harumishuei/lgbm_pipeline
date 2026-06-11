"""
機械学習パイプライン用ユーティリティ関数

このモジュールは、LightGBMを使用した機械学習パイプラインで使用される
共通関数を提供します。

カテゴリ:
    - 共通ユーティリティ: 時間測定、タイムスタンプ、ディレクトリ管理
    - データ処理: データ読み込み、保存、分割、変換
    - モデル学習: MLflowセットアップ、ハイパーパラメータ最適化
    - モデル評価: 評価指標計算、特徴量重要度、SHAP分析
    - モデル推論: ONNX変換、モデルエクスポート
"""

import time
import logging
from functools import wraps
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import polars as pl
from tqdm.auto import tqdm

import lightgbm as lgb
import shap
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    ndcg_score
)

import mlflow
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

from onnxmltools.convert import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType


logger = logging.getLogger(__name__)


# ============================================================
# 共通ユーティリティ
# ============================================================

def timing_decorator(func):
    """関数の実行時間を計測してログ出力するデコレータ
    
    Args:
        func: 計測対象の関数
        
    Returns:
        ラップされた関数
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time: float = time.time()
        result = func(*args, **kwargs)
        elapsed_time: float = time.time() - start_time
        logger.info("%s 実行時間: %.2f秒", func.__name__, elapsed_time)
        return result
    return wrapper


def get_timestamp() -> str:
    """タイムスタンプ文字列を取得
    
    出力ディレクトリやファイル名に使用するタイムスタンプを生成します。
    フォーマット: YYMMDDHHMM
    
    Returns:
        str: タイムスタンプ文字列
    """
    return datetime.now().strftime("%y%m%d%H%M")


@timing_decorator
def create_output_dir(base_dir: str = "../output", timestamp: Optional[str] = None) -> Path:
    """出力ディレクトリを作成
    
    タイムスタンプ付きの出力ディレクトリと、その配下にmodel, params, artifactsの
    サブディレクトリを作成します。
    
    Args:
        base_dir: ベースディレクトリパス
        timestamp: タイムスタンプ（Noneの場合は自動生成）
        
    Returns:
        Path: 作成された出力ディレクトリのパス
    """
    output_timestamp: str = timestamp or get_timestamp()
    output_dir = Path(base_dir) / output_timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    subdirs = ["model", "params", "artifacts"]
    for sub in subdirs:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)
    
    logger.info("出力ディレクトリを作成しました: %s", output_dir)
    return output_dir


def get_latest_output_dir(base_dir: str = "../output") -> Path:
    """最新の出力ディレクトリを取得
    
    ベースディレクトリ内の最新のタイムスタンプディレクトリを返します。
    
    Args:
        base_dir: ベースディレクトリパス
        
    Returns:
        Path: 最新の出力ディレクトリのパス
        
    Raises:
        FileNotFoundError: 出力ディレクトリが見つからない場合
    """
    base = Path(base_dir)
    dirs = [d for d in base.iterdir() if d.is_dir()]
    
    if not dirs:
        raise FileNotFoundError("No output directories found.")
    
    latest = max(dirs, key=lambda d: d.name)
    return latest


# ============================================================
# データ処理
# ============================================================

@timing_decorator
def load_data_pd(path: str, chunk_size: int) -> pd.DataFrame:
    """pandasでCSVデータを読み込み（チャンク処理）
    
    大きなCSVファイルをチャンク単位で読み込み、進捗バーを表示します。
    
    Args:
        path: CSVファイルのパス
        chunk_size: チャンクサイズ
        
    Returns:
        pd.DataFrame: 読み込まれたデータフレーム
    """
    chunks = []
    for chunk in tqdm(pd.read_csv(path, chunksize=chunk_size), desc="Loading data"):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


@timing_decorator
def load_data_pl(path: str, chunk_size: int) -> pl.DataFrame:
    """polarsでCSVデータを読み込み（チャンク処理）
    
    大きなCSVファイルをチャンク単位で読み込み、進捗バーを表示します。
    
    Args:
        path: CSVファイルのパス
        chunk_size: チャンクサイズ
        
    Returns:
        pl.DataFrame: 読み込まれたデータフレーム
    """
    batches = pl.scan_csv(path).collect_batches(chunk_size=chunk_size)
    chunks = list(tqdm(batches, desc="Loading data"))
    return pl.concat(chunks)


def save_csvs(files: Dict[str, pd.DataFrame | pl.DataFrame]) -> None:
    """データフレームをCSVファイルに保存
    
    pandas/polars両方のデータフレームに対応した保存関数です。
    
    Args:
        files: {ファイルパス: データフレーム}の辞書
        
    Raises:
        TypeError: サポートされていないデータフレーム型の場合
    """
    for file_path, df in files.items():
        if isinstance(df, pl.DataFrame):
            df.write_csv(file_path)
        elif isinstance(df, pd.DataFrame):
            df.to_csv(file_path, index=False)
        else:
            raise TypeError(f"Unsupported type: {type(df)}")
        
        logger.info("%s 保存完了", file_path)


@timing_decorator
def split_data_kfold(
    df: pl.DataFrame,
    target: str,
    n_splits: int = 5,
    random_state: int = 42
) -> List[Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]]:
    """Stratified K-Foldでデータを分割
    
    層化K分割交差検証用にデータを分割します。
    
    Args:
        df: 入力データフレーム
        target: ターゲット列名
        n_splits: 分割数
        random_state: 乱数シード
        
    Returns:
        List[Tuple]: (X_train, X_valid, y_train, y_valid)のリスト
    """
    feature_df = df.drop(target).to_pandas()
    target_values = df.select(target).to_pandas().values.ravel()
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []
    
    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(feature_df, target_values), 1):
        x_train = pl.from_pandas(feature_df.iloc[train_idx])
        x_valid = pl.from_pandas(feature_df.iloc[valid_idx])
        y_train = pl.DataFrame({target: target_values[train_idx]})
        y_valid = pl.DataFrame({target: target_values[valid_idx]})
        
        logger.info("Fold %s: 訓練=%s, 検証=%s", fold_idx, len(x_train), len(x_valid))
        folds.append((x_train, x_valid, y_train, y_valid))
    
    return folds


@timing_decorator
def cat_num_split(x_train: pl.DataFrame) -> Tuple[List[str], List[str]]:
    """カテゴリ列と数値列を分割
    
    データフレームの列をカテゴリ列と数値列に分類します。
    
    Args:
        x_train: 入力データフレーム
        
    Returns:
        Tuple[List[str], List[str]]: (カテゴリ列リスト, 数値列リスト)
    """
    cat_cols: List[str] = []
    num_cols: List[str] = []
    schema: Dict[str, Any] = x_train.schema
    
    for col, dtype in schema.items():
        if dtype == pl.Utf8 or dtype == pl.Categorical:
            cat_cols.append(col)
        else:
            num_cols.append(col)
    
    logger.info("カテゴリ列: %s", cat_cols)
    logger.info("数値列: %s", num_cols)
    
    return cat_cols, num_cols


def to_categorical(df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    """指定列をカテゴリ型に変換
    
    Args:
        df: 入力データフレーム
        cat_cols: カテゴリ列のリスト
        
    Returns:
        pd.DataFrame: 変換後のデータフレーム
    """
    for col in cat_cols:
        df[col] = df[col].astype("category")
    return df


# ============================================================
# モデル学習
# ============================================================

@timing_decorator
def load_model(model_file: str) -> lgb.Booster:
    """LightGBMモデルを読み込み
    
    Args:
        model_file: モデルファイルのパス
        
    Returns:
        lgb.Booster: 読み込まれたモデル
    """
    model: lgb.Booster = lgb.Booster(model_file=model_file)
    logger.info("モデル読み込み完了: %s", model_file)
    return model


@timing_decorator
def setup_mlflow(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    experiment_name: str,
) -> None:
    """MLflowをAzure MLに接続してセットアップ
    
    Azure Machine Learning ワークスペースに接続し、MLflowの
    トラッキングURIと実験名を設定します。
    
    Args:
        subscription_id: AzureサブスクリプションID
        resource_group: リソースグループ名
        workspace_name: ワークスペース名
        experiment_name: 実験名
    """
    ml_client: MLClient = MLClient(
        DefaultAzureCredential(),
        subscription_id,
        resource_group,
        workspace_name,
    )

    tracking_uri: str = (
        ml_client.workspaces
        .get(workspace_name)
        .mlflow_tracking_uri
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    logger.info("MLflow tracking URI設定完了")
    logger.info("Experiment: %s", experiment_name)


@timing_decorator
def run_hpo(
    train_model,
    param_space: Dict[str, Any],
    metric: str,
    mode: str,
    num_samples: int,
):
    """ハイパーパラメータ最適化を実行（Ray Tune + Optuna）
    
    Ray TuneとOptunaを使用してハイパーパラメータ最適化を実行します。
    
    Args:
        train_model: 学習関数
        param_space: パラメータ探索空間
        metric: 最適化する評価指標
        mode: 最適化モード（"min" or "max"）
        num_samples: 試行回数
        
    Returns:
        Tuple: (results, best_result)
    """
    search_alg = OptunaSearch(metric=metric, mode=mode)
    tuner = tune.Tuner(
        train_model,
        tune_config=tune.TuneConfig(
            metric=metric,
            mode=mode,
            scheduler=ASHAScheduler(),
            search_alg=search_alg,
            num_samples=num_samples,
        ),
        param_space=param_space,
    )
    
    results = tuner.fit()
    best_result = results.get_best_result(metric=metric, mode=mode)
    
    logger.info("Best config:")
    logger.info("%s", best_result.config)
    
    return results, best_result


# ============================================================
# モデル評価
# ============================================================

@timing_decorator
def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Recall@Kを計算
    
    上位K件の予測における再現率を計算します。
    
    Args:
        y_true: 真のラベル
        y_score: 予測スコア
        k: 上位K件
        
    Returns:
        float: Recall@K
    """
    idx: np.ndarray = np.argsort(-y_score)[:k]
    return float(y_true[idx].sum() / y_true.sum())


@timing_decorator
def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Precision@Kを計算
    
    上位K件の予測における適合率を計算します。
    
    Args:
        y_true: 真のラベル
        y_score: 予測スコア
        k: 上位K件
        
    Returns:
        float: Precision@K
    """
    idx: np.ndarray = np.argsort(-y_score)[:k]
    return float(y_true[idx].sum() / k)


@timing_decorator
def mcc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Matthews Correlation Coefficientを計算
    
    二値分類の性能を評価するMCCを計算します。
    
    Args:
        y_true: 真のラベル
        y_pred: 予測ラベル
        
    Returns:
        float: MCC
    """
    return float(matthews_corrcoef(y_true, y_pred))


@timing_decorator
def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """NDCG@Kを計算
    
    上位K件の予測におけるNormalized Discounted Cumulative Gainを計算します。
    
    Args:
        y_true: 真のラベル
        y_score: 予測スコア
        k: 上位K件
        
    Returns:
        float: NDCG@K
    """
    y_true_2d: np.ndarray = y_true.reshape(1, -1)
    y_score_2d: np.ndarray = y_score.reshape(1, -1)
    return float(ndcg_score(y_true_2d, y_score_2d, k=k))


@timing_decorator
def hit_rate_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Hit Rate@Kを計算
    
    上位K件の予測に正解が含まれているかを評価します。
    
    Args:
        y_true: 真のラベル
        y_score: 予測スコア
        k: 上位K件
        
    Returns:
        float: Hit Rate@K (1.0 or 0.0)
    """
    idx: np.ndarray = np.argsort(-y_score)[:k]
    return 1.0 if y_true[idx].sum() > 0 else 0.0


@timing_decorator
def compute_metrics(
    y: np.ndarray,
    y_pred_proba: np.ndarray,
    y_pred_label: np.ndarray,
    top_k: int
) -> Dict[str, Any]:
    """各種評価指標を一括計算
    
    AUC, Recall@K, Precision@K, MCC, NDCG@K, Hit Rate@Kを計算します。
    
    Args:
        y: 真のラベル
        y_pred_proba: 予測確率
        y_pred_label: 予測ラベル
        top_k: 上位K件
        
    Returns:
        Dict[str, Any]: 評価指標の辞書
    """
    metrics: Dict[str, Any] = {
        "auc": roc_auc_score(y, y_pred_proba),
        f"recall_at_{top_k}": recall_at_k(y, y_pred_proba, top_k),
        f"precision_at_{top_k}": precision_at_k(y, y_pred_proba, top_k),
        "mcc": mcc_score(y, y_pred_label),
        f"ndcg_at_{top_k}": ndcg_at_k(y, y_pred_proba, top_k),
        f"hit_rate_at_{top_k}": hit_rate_at_k(y, y_pred_proba, top_k),
    }
    
    return metrics


@timing_decorator
def get_feature_importance(
    model: lgb.Booster,
    feature_names: List[str],
    model_name: str,
    top_n: int = 20
) -> pl.DataFrame:
    """特徴量重要度を取得して表示
    
    LightGBMモデルの特徴量重要度（gain）を取得し、
    上位N件を表示します。
    
    Args:
        model: LightGBMモデル
        feature_names: 特徴量名のリスト
        model_name: モデル名（表示用）
        top_n: 表示する上位件数
        
    Returns:
        pl.DataFrame: 特徴量重要度のデータフレーム
    """
    # 特徴量重要度取得（gain）
    importance: np.ndarray = model.feature_importance(importance_type='gain')
    
    # 正規化
    importance = importance / importance.sum()
    
    # データフレーム作成
    importance_df: pl.DataFrame = pl.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort("importance", descending=True)
    
    # 上位N件表示
    print(f"\n{'='*70}")
    print(f"{model_name}モデル - 特徴量重要度 Top {top_n}")
    print(f"{'='*70}")
    print(importance_df.head(top_n))
    
    return importance_df


def compute_shap_values(model: lgb.Booster, X: pd.DataFrame) -> Tuple[np.ndarray, float]:
    """SHAP値を計算
    
    TreeExplainerを使用してSHAP値を計算し、サマリープロットを表示します。
    
    Args:
        model: LightGBMモデル
        X: 入力データ
        
    Returns:
        Tuple[np.ndarray, float]: (SHAP値, 期待値)
    """
    explainer = shap.TreeExplainer(model)
    logger.info("Expected value: %s", explainer.expected_value)
    
    shap_values = explainer.shap_values(X)
    logger.info("X shape           : %s", X.shape)
    logger.info("shap_values shape : %s", shap_values.shape)
    
    # サマリープロット表示
    shap.summary_plot(shap_values, X, feature_names=X.columns.tolist())
    
    return shap_values, explainer.expected_value


# ============================================================
# モデル推論
# ============================================================

def export_onnx_lgbm(
    model: lgb.Booster,
    train_pd: pd.DataFrame,
    onnx_path: str,
    mlflow_artifact_path: str = "model/onnx"
) -> None:
    """LightGBMモデルをONNX形式に変換してエクスポート
    
    モデルをONNX形式に変換し、ファイルに保存してMLflowにログします。
    
    Args:
        model: LightGBMモデル
        train_pd: 学習データ（特徴量の型情報取得用）
        onnx_path: ONNX保存パス
        mlflow_artifact_path: MLflowアーティファクトパス
    """
    n_features = train_pd.shape[1]
    initial_type = [("input", FloatTensorType([None, n_features]))]
    
    # ONNX変換
    onnx_model = convert_lightgbm(model, initial_types=initial_type)
    
    # ファイル保存
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    # MLflowにログ
    mlflow.log_artifact(onnx_path, artifact_path=mlflow_artifact_path)
    
    logger.info("ONNXモデル保存完了: %s", onnx_path)

# Made with Bob

import time
import logging
from functools import wraps
import polars as pl
import pandas as pd
import numpy as np
import shap
from tqdm.auto import tqdm
import lightgbm as lgb
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime
from pathlib import Path
import mlflow
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    ndcg_score
)
from onnxmltools.convert import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType


logger = logging.getLogger(__name__)

# == 共通関数 ==
# 時間測定
def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time: float = time.time()
        result = func(*args, **kwargs)
        elapsed_time: float = time.time() - start_time
        logger.info("%s 実行時間: %.2f秒", func.__name__, elapsed_time)
        return result
    return wrapper

    
# timestampの取得
def get_timestamp() -> str:
    """出力ディレクトリやファイル名に利用するタイムスタンプ文字列を返す。"""
    return datetime.now().strftime("%y%m%d%H%M")

@timing_decorator
def create_output_dir(base_dir: str = "../output", timestamp: Optional[str] = None) -> Path:
    output_timestamp: str = timestamp or get_timestamp()
    output_dir = Path(base_dir) / output_timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    subdirs = ["model", "params", "artifacts"]
    for sub in subdirs:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)
    logger.info("出力ディレクトリを作成しました: %s", output_dir)
    return output_dir

def get_latest_output_dir(base_dir: str = "../output") -> Path:
    base = Path(base_dir)
    dirs = [d for d in base.iterdir() if d.is_dir()]
    if not dirs:
        raise FileNotFoundError("No output directories found.")
    latest = max(dirs, key=lambda d: d.name) 
    return latest
    

# データ分割（Stratified K-Fold）
@timing_decorator
def split_data_kfold(
    df: pl.DataFrame,
    target: str,
    n_splits: int = 5,
    random_state: int = 42
) -> List[Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]]:
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

# カテゴリ列と数値列の分割
@timing_decorator
def cat_num_split(x_train: pl.DataFrame) -> Tuple[List[str], List[str]]:
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


def to_categorical(df, cat_cols):
    for col in cat_cols:
        df[col] = df[col].astype("category")
    return df


# pd・pl関係なくdfをCSVファイルに保存する
def save_csvs(files: Dict[str, pd.DataFrame | pl.DataFrame]) -> None:
    for file_path, df in files.items():
        # Polars DataFrame
        if isinstance(df, pl.DataFrame):
            df.write_csv(file_path)
        # pandas DataFrame
        elif isinstance(df, pd.DataFrame):
            df.to_csv(file_path, index=False)
        else:
            raise TypeError(f"Unsupported type: {type(df)}")
        logger.info("%s 保存完了", file_path)


    
# データロード
@timing_decorator
def load_data_pl(path: str, chunk_size: int) -> pl.DataFrame:
    batches = pl.scan_csv(path).collect_batches(chunk_size=chunk_size)
    chunks = list(tqdm(batches))
    return pl.concat(chunks)

@timing_decorator
def load_data_pd(path: str, chunk_size: int) -> pd.DataFrame:
    chunks = []
    for chunk in tqdm(pd.read_csv(path, chunksize=chunk_size)):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)
    
# モデルロード
@timing_decorator
def load_model(model_file: str) -> lgb.Booster:
    model: lgb.Booster = lgb.Booster(model_file=model_file)
    logger.info("モデル読み込み完了: %s", model_file)
    return model


# MLセットアップ
@timing_decorator
def setup_mlflow(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    experiment_name: str,
) -> None:
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


# HPO
@timing_decorator
def run_hpo(
    train_model,
    param_space: Dict[str, Any],
    metric: str,
    mode: str,
    num_samples: int,
):
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

# =====================================
# モデル評価
# =====================================
# 評価指標
@timing_decorator
def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Recall@Kを計算
    Args:
        y_true: 真のラベル
        y_score: 予測スコア
        k: 上位K件
    Returns:
        Recall@K
    """
    idx: np.ndarray = np.argsort(-y_score)[:k]
    return float(y_true[idx].sum() / y_true.sum())

@timing_decorator
def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Precision@Kを計算
    Args:
        y_true: 真のラベル
        y_score: 予測スコア
        k: 上位K件
    Returns:
        Precision@K
    """
    idx: np.ndarray = np.argsort(-y_score)[:k]
    return float(y_true[idx].sum() / k)

@timing_decorator
def mcc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Matthews Correlation Coefficientを計算￥
    Args:
        y_true: 真のラベル
        y_pred: 予測ラベル
    Returns:
        MCC
    """
    return float(matthews_corrcoef(y_true, y_pred))

@timing_decorator
def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """NDCG@Kを計算
    Args:
        y_true: 真のラベル
        y_score: 予測スコア
        k: 上位K件
    Returns:
        NDCG@K
    """
    y_true_2d: np.ndarray = y_true.reshape(1, -1)
    y_score_2d: np.ndarray = y_score.reshape(1, -1)
    return float(ndcg_score(y_true_2d, y_score_2d, k=k))

@timing_decorator
def hit_rate_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Hit Rate@Kを計算
    Args:
        y_true: 真のラベル
        y_score: 予測スコア
        k: 上位K件   
    Returns:
        Hit Rate@K
    """
    idx: np.ndarray = np.argsort(-y_score)[:k]
    return 1.0 if y_true[idx].sum() > 0 else 0.0

# 評価指標計算
@timing_decorator
def compute_metrics(
    y: np.ndarray,
    y_pred_proba: np.ndarray,
    y_pred_label: np.ndarray,
    top_k: int
) -> Dict[str, Any]:

    metrics: Dict[str, Any] = {
        "auc": roc_auc_score(y, y_pred_proba),
        f"recall_at_{top_k}": recall_at_k(y, y_pred_proba, top_k),
        f"precision_at_{top_k}": precision_at_k(y, y_pred_proba, top_k),
        "mcc": mcc_score(y, y_pred_label),
        f"ndcg_at_{top_k}": ndcg_at_k(y, y_pred_proba, top_k),
        f"hit_rate_at_{top_k}": hit_rate_at_k(y, y_pred_proba, top_k),
    }

    return metrics

# 特徴量重要度
@timing_decorator
def get_feature_importance(
    model: lgb.Booster,
    feature_names: list,
    model_name: str,
    top_n: int = 20
) -> pl.DataFrame:

    # 特徴量重要度取得
    importance: np.ndarray = model.feature_importance(importance_type='gain')
    
    importance = importance / importance.sum()
    
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

# shap値
def compute_shap_values(model, X):
    explainer = shap.TreeExplainer(model)
    logger.info("Expected value: %s", explainer.expected_value)

    shap_values = explainer.shap_values(X)
    logger.info("X_valid shape     : %s", X_valid.shape)
    logger.info("shap_values shape : %s", shap_values.shape)

    shap.summary_plot(shap_values, X_valid, feature_names=X_valid.columns.tolist())

    return shap_values, explainer.expected_value



# == LGBM関連関数 ==
# モデルファイルをonnxに変換
def export_onnx_lgbm(
    model: lgb.Booster,
    train_pd: pd.DataFrame,
    onnx_path: str,
    mlflow_artifact_path: str = "model/onnx"
) -> None:
    n_features = train_pd.shape[1]
    initial_type = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_lightgbm(model, initial_types=initial_type)
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    mlflow.log_artifact(onnx_path, artifact_path=mlflow_artifact_path)


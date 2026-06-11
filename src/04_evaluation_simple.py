"""
簡易評価スクリプト
03で保存された予測結果を読み込んで評価指標を計算
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)
import glob
import logging

# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_latest_output_dir(base_dir: str = "../output") -> Path:
    """最新の出力ディレクトリを取得"""
    base = Path(base_dir)
    dirs = [d for d in base.iterdir() if d.is_dir()]
    if not dirs:
        raise FileNotFoundError("No output directories found.")
    latest = max(dirs, key=lambda d: d.name)
    return latest


def calculate_metrics(y_true, y_pred_proba, threshold=0.5):
    """評価指標を計算"""
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {
        'AUC': roc_auc_score(y_true, y_pred_proba),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred),
    }
    
    return metrics


def main():
    # 最新の出力ディレクトリを取得
    output_dir = get_latest_output_dir()
    logger.info(f"使用する出力ディレクトリ: {output_dir}")
    
    # 予測結果ファイルを探す
    prediction_files = glob.glob(f"{output_dir}/prediction_result_*.csv")
    if not prediction_files:
        raise FileNotFoundError(f"予測結果ファイルが見つかりません: {output_dir}/prediction_result_*.csv")
    
    prediction_file = prediction_files[0]
    logger.info(f"予測結果ファイル: {prediction_file}")
    
    # データ読み込み
    df = pd.read_csv(prediction_file)
    logger.info(f"データ形状: {df.shape}")
    logger.info(f"カラム: {df.columns.tolist()}")
    
    # 評価指標を計算
    results = []
    
    # baselineモデルの評価
    if 'y_pred_baseline' in df.columns:
        logger.info("\n=== Baseline Model ===")
        metrics_baseline = calculate_metrics(df['y_true'], df['y_pred_baseline'])
        for metric, value in metrics_baseline.items():
            logger.info(f"{metric}: {value:.4f}")
        metrics_baseline['model'] = 'baseline'
        results.append(metrics_baseline)
    
    # tunedモデルの評価
    if 'y_pred_tuned' in df.columns:
        logger.info("\n=== Tuned Model ===")
        metrics_tuned = calculate_metrics(df['y_true'], df['y_pred_tuned'])
        for metric, value in metrics_tuned.items():
            logger.info(f"{metric}: {value:.4f}")
        metrics_tuned['model'] = 'tuned'
        results.append(metrics_tuned)
    
    # 結果を保存
    if results:
        results_df = pd.DataFrame(results)
        artifact_dir = output_dir / "artifacts"
        artifact_dir.mkdir(exist_ok=True)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = artifact_dir / f"evaluation_metrics_{timestamp}.csv"
        results_df.to_csv(output_file, index=False)
        logger.info(f"\n評価結果を保存しました: {output_file}")
        
        # 比較表示
        if len(results) > 1:
            logger.info("\n=== Model Comparison ===")
            logger.info(f"\n{results_df.to_string(index=False)}")


if __name__ == "__main__":
    main()

# Made with Bob

# LightGBM Machine Learning Pipeline

LightGBMを使用した機械学習パイプラインプロジェクトです。モデルの学習、評価、予測を体系的に実行できます。

## 📁 プロジェクト構成

```
pipeline/
├── src/                                   # ソースコード
│   ├── 03_train_lgbm_{カテゴリ名}.ipynb      # モデル学習
│   ├── 04_evaluation_lightgbm_{カテゴリ名}.ipynb  # モデル評価
│   ├── 05_prediction_lgbm_{カテゴリ名}.ipynb # モデル推論
│   └── utils.py                           # 共通ユーティリティ関数
│
├── data/                                  # データディレクトリ
│   ├── train.csv                          # 学習データ
│   ├── test.csv                           # テストデータ
│   ├── X_train.csv                        # 学習用特徴量
│   ├── X_valid.csv                        # 検証用特徴量
│   ├── y_train.csv                        # 学習用ターゲット
│   └── y_valid.csv                        # 検証用ターゲット
│
├── output/                                # 出力ディレクトリ
│   └── YYMMDDHHMM/                        # タイムスタンプ付きディレクトリ
│       ├── model/                         # モデルファイル
│       ├── params/                        # パラメータファイル
│       └── artifacts/                     # 評価結果・特徴量重要度など
│
├── requirements.txt                       # 依存パッケージ
├── .env.example                           # 環境変数テンプレート
└── README.md                              # このファイル
```

## 🚀 セットアップ

### 1. 環境構築

```bash
# リポジトリのクローン
git clone <repository-url>
cd pipeline

# 仮想環境の作成と有効化
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 依存パッケージのインストール
pip install -r requirements.txt
```

### 2. 環境変数の設定

```bash
# .env.exampleをコピーして.envを作成
cp .env.example .env

# .envファイルを編集してAzure ML情報を設定
# SUBSCRIPTION_ID=your-subscription-id
# RESOURCE_GROUP=your-resource-group
# WORKSPACE_NAME=your-workspace-name
```

### 3. データの準備

`data/`ディレクトリに以下のファイルを配置してください：
- `train.csv`: 学習用データ
- `test.csv`: テスト用データ

## 📊 使用方法

### ステップ1: モデル学習

`src/03_train_lgbm_{カテゴリ名}.ipynb`を実行

**実施内容:**
- ベースラインモデル学習
- ハイパーパラメータチューニング（Ray Tune / AutoGluon）
- チューニング済みモデル学習
- パラメータ比較
- MLflowによる実験管理

**出力:**
- `output/YYMMDDHHMM/model/model_baseline_*.lgbm`
- `output/YYMMDDHHMM/model/model_tuned_*.lgbm`
- `output/YYMMDDHHMM/model/*.onnx`
- `output/YYMMDDHHMM/params/*.json`

### ステップ2: モデル評価

`src/04_evaluation_lightgbm_{カテゴリ名}.ipynb`を実行

**実施内容:**
- 各種メトリクス算出（AUC, Precision, Recall, F1, MCC, NDCG, Hit Rate）
- 特徴量重要度分析
- SHAP値分析
- ベースラインとチューニング済みモデルの比較

**出力:**
- `output/YYMMDDHHMM/artifacts/evaluation_*.csv`
- `output/YYMMDDHHMM/artifacts/feature_importance_*.csv`
- `output/YYMMDDHHMM/artifacts/shap_*.csv`

### ステップ3: モデル推論

`src/05_prediction_lgbm_{カテゴリ名}.ipynb`を実行

**実施内容:**
- テストデータの読み込み
- 学習済みモデルによる予測
- 予測結果のCSV出力

**出力:**
- `output/YYMMDDHHMM/prediction_result_*.csv`

## 🔧 主要機能

### ハイパーパラメータチューニング

2つの方法をサポート：
1. **Ray Tune + Optuna**: 柔軟なハイパーパラメータ探索
2. **AutoGluon**: 自動機械学習による最適化

### モデル管理

- **MLflow**: 実験管理とモデルバージョニング
- **Azure ML**: クラウドベースのモデル管理
- **ONNX**: モデルの相互運用性とデプロイメント

### モデル解釈性

- **特徴量重要度**: LightGBMネイティブの重要度
- **SHAP値**: モデル予測の説明可能性

## 📝 コメント規則

プロジェクト全体で以下のコメント規則を統一しています：

- **ロガー設定**: `# logging_config.pyの設計方針を参考にloggerを初期化`
- **共通化関数**: `# ==共通化== →utils.py`
- **モデル種別**: 
  - ベースライン: `ベースラインモデル`
  - チューニング済み: `チューニング済みモデル`

## 🛠️ 技術スタック

- **機械学習**: LightGBM, scikit-learn
- **データ処理**: pandas, polars, numpy
- **ハイパーパラメータ最適化**: Ray Tune, Optuna, AutoGluon
- **モデル管理**: MLflow, Azure ML
- **モデルエクスポート**: ONNX, ONNXRuntime
- **解釈性**: SHAP

## 📄 ライセンス

このプロジェクトのライセンスについては、プロジェクトオーナーにお問い合わせください。

## 🤝 貢献

プロジェクトへの貢献を歓迎します。Issue や Pull Request をお気軽にお送りください。

## 📧 お問い合わせ

質問や問題がある場合は、Issueを作成してください。
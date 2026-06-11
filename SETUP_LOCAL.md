# ローカル環境でのセットアップガイド

このガイドは、プロジェクトフォルダを受け取った方向けの簡易セットアップ手順です。

## 📋 前提条件

- Python 3.8以上がインストールされていること
- インターネット接続（パッケージインストール時に必要）

## 🚀 セットアップ手順

### ステップ1: プロジェクトフォルダの確認

受け取ったフォルダに以下のファイル・ディレクトリがあることを確認してください：

```
pipeline/
├── src/                          # ノートブックファイル
│   ├── 03_train_lgbm_{カテゴリ名}.ipynb
│   ├── 04_evaluation_lightgbm_{カテゴリ名}.ipynb
│   ├── 05_prediction_lgbm_{カテゴリ名}.ipynb
│   └── utils.py
├── data/                         # データフォルダ（空でも可）
├── output/                       # 出力フォルダ（空でも可）
├── requirements.txt              # 依存パッケージリスト
├── .env.example                  # 環境変数テンプレート
└── README.md                     # プロジェクト説明
```

### ステップ2: ターミナル/コマンドプロンプトを開く

**macOS/Linux:**
```bash
# ターミナルを開いて、プロジェクトフォルダに移動
cd /path/to/pipeline
```

**Windows:**
```cmd
# コマンドプロンプトまたはPowerShellを開いて、プロジェクトフォルダに移動
cd C:\path\to\pipeline
```

### ステップ3: 仮想環境の作成

```bash
# 仮想環境を作成
python -m venv .venv

# または、python3コマンドの場合
python3 -m venv .venv
```

### ステップ4: 仮想環境の有効化

**macOS/Linux:**
```bash
source .venv/bin/activate
```

**Windows (コマンドプロンプト):**
```cmd
.venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

> **注意**: PowerShellで実行ポリシーエラーが出る場合：
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### ステップ5: 依存パッケージのインストール

```bash
# requirements.txtからパッケージをインストール
pip install -r requirements.txt

# インストールに時間がかかる場合があります（5-10分程度）
```

### ステップ6: 環境変数の設定（Azure ML使用時のみ）

Azure Machine Learningを使用する場合：

```bash
# .env.exampleをコピーして.envを作成
cp .env.example .env  # Windows: copy .env.example .env

# .envファイルをテキストエディタで開いて編集
# SUBSCRIPTION_ID=あなたのサブスクリプションID
# RESOURCE_GROUP=あなたのリソースグループ名
# WORKSPACE_NAME=あなたのワークスペース名
```

### ステップ7: データの準備

`data/`フォルダに以下のファイルを配置：
- `train.csv`: 学習用データ
- `test.csv`: テスト用データ

### ステップ8: Jupyter Notebookの起動

```bash
# Jupyter Notebookを起動
jupyter notebook

# または、Jupyter Labを使用する場合
jupyter lab
```

ブラウザが自動的に開き、ノートブックが表示されます。

### ステップ9: ノートブックの実行

以下の順番でノートブックを実行してください：

1. **03_train_lgbm_{カテゴリ名}.ipynb** - モデル学習
2. **04_evaluation_lightgbm_{カテゴリ名}.ipynb** - モデル評価
3. **05_prediction_lgbm_{カテゴリ名}.ipynb** - モデル推論

## 🔧 トラブルシューティング

### パッケージインストールエラー

```bash
# pipをアップグレード
pip install --upgrade pip

# 再度インストール
pip install -r requirements.txt
```

### 仮想環境が有効化できない

```bash
# 仮想環境を削除して再作成
rm -rf .venv  # Windows: rmdir /s .venv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### Jupyter Notebookでカーネルが見つからない

```bash
# 仮想環境内でipykernelをインストール
pip install ipykernel
python -m ipykernel install --user --name=.venv
```

## 📞 サポート

問題が解決しない場合は、以下の情報を添えてお問い合わせください：
- 使用しているOS（Windows/macOS/Linux）
- Pythonのバージョン（`python --version`で確認）
- エラーメッセージの全文

## 🎉 セットアップ完了！

これで環境構築は完了です。ノートブックを順番に実行して、機械学習パイプラインをお楽しみください！
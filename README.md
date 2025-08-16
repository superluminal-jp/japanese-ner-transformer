# Japanese Named Entity Recognition with Transformers

日本語テキストに対する固有表現抽出（NER）をTransformersライブラリで実行するプロジェクトです。

## 概要

このプロジェクトは、日本語の文書から人名、組織名、場所名などの固有表現を自動的に抽出します。Hugging Face Transformersライブラリと事前学習済みの日本語NERモデルを使用しています。

### 主な機能

- **統一処理エンジン**: 短文・長文を自動判定して最適処理
- **長文対応**: チャンク分割による長文書の完全解析
- **バッチ処理**: 複数ファイル・ディレクトリの一括分析
- **詳細レポート**: CSV出力、統計分析、可視化グラフ生成
- **対応形式**: テキストファイル（.txt）、JSONファイル（.json）

## 使用モデル

- **メインモデル**: `tsmatz/xlm-roberta-ner-japanese`
  - Stockmarkの「Wikipediaを用いた日本語の固有表現抽出データセット」でファインチューニングされたモデル
  - 高い精度の固有表現抽出が可能

## 抽出可能な固有表現タイプ

| Label ID | Tag | Widget Tag | Description |
|----------|-----|------------|-------------|
| 0 | O | (None) | others or nothing |
| 1 | PER | PER | person |
| 2 | ORG | ORG | general corporation organization |
| 3 | ORG-P | P | political organization |
| 4 | ORG-O | O | other organization |
| 5 | LOC | LOC | location |
| 6 | INS | INS | institution, facility |
| 7 | PRD | PRD | product |
| 8 | EVT | EVT | event |

## セットアップ

### 必要な環境
- Python 3.7+
- CUDA対応GPU（推奨）

### 依存関係のインストール

#### 本番環境用
```bash
pip install -r requirements.txt
```

#### 開発環境用（テスト・リンタ含む）
```bash
pip install -r requirements-dev.txt
```

#### Makefileを使用した簡単セットアップ
```bash
# 本番環境用依存関係をインストール
make install

# 開発環境用依存関係をインストール
make install-dev
```

## 使用方法

### 1. デフォルト実行（デモ）

```bash
python main.py
```

`demo/sample_text.txt`のサンプルテキスト（経済産業省のGENIAC事業に関する文章）から固有表現を抽出し、完全なレポートを生成します。

#### 出力例

```
Reading documents from: demo
Found 1 documents
Analyzing: sample_text.txt
Generating statistics...
CSV saved to: output/ner_results.csv
Creating visualizations...
Generating report...
Report saved to: output/analysis_report.md

Analysis complete! Results saved to: output
- CSV: output/ner_results.csv
- Report: output/analysis_report.md
- Visualizations: output/*.png
```

### 2. バッチ処理

#### 単一ファイルの分析

```bash
python main.py document.txt
```

#### ディレクトリ内の全.txtファイルを分析

```bash
python main.py /path/to/documents/
```

#### 出力ディレクトリを指定

```bash
python main.py documents/ -o results/
```

#### 異なるモデルを使用

```bash
python main.py documents/ -m "other-model-name"
```

### 3. コマンドラインオプション

```
usage: main.py [-h] [-o OUTPUT] [-m MODEL] [input_path]

日本語固有表現抽出ツール

positional arguments:
  input_path            入力ファイルまたはディレクトリのパス（デフォルト: demo）

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        出力ディレクトリ (デフォルト: output)
  -m MODEL, --model MODEL
                        NERモデル名
```

## 出力ファイル

バッチ処理実行時、以下のファイルが出力されます：

### 1. CSV形式の詳細データ (`ner_results.csv`)

```csv
filename,word,entity_type,entity_description,score,start_pos,end_pos,analysis_time
sample_text.txt,AWS,ORG,general corporation organization,0.9979,384,387,2025-08-16T21:29:38.807819
sample_text.txt,有馬 幸介,PER,person,0.9994,1764,1769,2025-08-16T21:29:38.807819
sample_text.txt,経済産業省,P,political organization,0.9988,24,29,2025-08-16T21:29:38.807819
```

### 2. 分析レポート (`analysis_report.md`)

- 分析概要（総ドキュメント数、総固有表現数など）
- 固有表現タイプ別統計
- 最頻出固有表現ランキング
- ドキュメント別詳細

### 3. 可視化グラフ

- `entity_type_distribution.png`: 固有表現タイプ別分布
- `most_common_entities.png`: 最頻出固有表現（Top 10）
- `entities_per_document.png`: ドキュメント別固有表現数

## プロジェクト構造

```
japanese-ner-transformer/
├── main.py                     # メインエントリーポイント
├── requirements.txt            # 本番環境の依存関係
├── requirements-dev.txt        # 開発・テスト用の依存関係
├── pytest.ini                 # pytest設定ファイル
├── Makefile                   # 開発タスク自動化
├── README.md                  # このファイル
├── src/
│   └── japanese_ner/          # メインパッケージ
│       ├── __init__.py        # パッケージ初期化
│       ├── analyzer.py        # コア NER 分析機能
│       ├── batch_analyzer.py  # バッチ処理機能
│       ├── utils.py          # ファイル処理ユーティリティ
│       ├── report.py         # 統計・レポート生成
│       └── visualization.py   # グラフ・可視化機能
└── tests/                      # テストスイート
    ├── __init__.py            # テストパッケージ初期化
    ├── conftest.py            # pytest設定・共通フィクスチャ
    ├── run_tests.py           # テスト実行スクリプト
    ├── fixtures/              # テストデータ・フィクスチャ
    │   └── sample_data.py     # サンプルデータ定義
    ├── unit/                  # 単体テスト
    │   ├── test_analyzer.py   # アナライザーのテスト
    │   ├── test_report.py     # レポート機能のテスト
    │   ├── test_utils.py      # ユーティリティのテスト
    │   └── test_visualization.py # 可視化機能のテスト
    ├── integration/           # 統合テスト
    │   └── test_batch_analyzer.py # バッチ処理の統合テスト
    └── e2e/                   # エンドツーエンドテスト
        └── test_main.py       # メインスクリプトのE2Eテスト
```

### 主要クラスとモジュール

#### `NERAnalyzer` クラス (analyzer.py)
- `analyze(text)`: 単一テキストの固有表現抽出（長文自動対応）
- `analyze_documents(input_path)`: 複数ドキュメントの一括分析
- `generate_full_report(input_path, output_dir)`: 完全な分析レポート生成
- `get_entity_types()`: サポートされている固有表現タイプの取得

#### 長文処理機能
- **自動判定**: 400トークン超の長文を自動検出
- **チャンク分割**: 重複付きで最適分割
- **重複除去**: エンティティの統合処理

#### ユーティリティモジュール
- **utils.py**: ファイル読み込み、ディレクトリ処理
- **report.py**: 統計計算、CSV出力、マークダウンレポート生成
- **visualization.py**: matplotlib を使った可視化グラフ生成

## カスタマイズ

### 独自テキストでの簡単分析

`demo/sample_text.txt`ファイルを編集するか、新しいファイルを作成：

```bash
echo "あなたの解析したいテキスト" > my_text.txt
python main.py my_text.txt
```

### プログラムからの使用

```python
from src.japanese_ner import NERAnalyzer

# 単一テキスト分析（長文自動対応）
analyzer = NERAnalyzer()
entities = analyzer.analyze("分析したいテキスト")

# バッチ処理
analyzer.generate_full_report("input_directory/", "output_directory/")

# 異なるモデルの使用
custom_analyzer = NERAnalyzer("other-model-name")

# 長文テキストの処理例
long_text = """長い文書コンテンツ..."""
entities = analyzer.analyze(long_text)  # 自動的にチャンク処理
```

## テスト

このプロジェクトには包括的なテストスイートが含まれており、pytest を使用してコードの品質と機能を保証しています。

### テスト構成

テストは3つのレベルに分かれています：

#### 1. 単体テスト (Unit Tests)
個々のクラス・関数の動作をテストします。

- **test_analyzer.py**: `NERAnalyzer`クラスの各メソッドをテスト
- **test_utils.py**: ファイル読み込み・処理ユーティリティをテスト  
- **test_report.py**: 統計計算・レポート生成機能をテスト
- **test_visualization.py**: グラフ・可視化機能をテスト

#### 2. 統合テスト (Integration Tests)
複数のモジュール間の連携をテストします。

- **test_batch_analyzer.py**: `BatchNERAnalyzer`とその依存関係の統合テスト

#### 3. エンドツーエンドテスト (E2E Tests)
アプリケーション全体の動作をテストします。

- **test_main.py**: メインスクリプトのCLIインターフェース・機能フローをテスト

### テスト実行方法

#### Makefileを使用（推奨）

```bash
# 全テストを実行
make test

# 単体テストのみ実行
make test-unit

# 統合テストのみ実行
make test-integration

# E2Eテストのみ実行
make test-e2e

# カバレッジレポートを生成
make test-coverage
```

#### 専用テストランナーを使用

```bash
# 全テストを実行
python tests/run_tests.py

# 単体テストのみ実行
python tests/run_tests.py --unit

# 統合テストのみ実行
python tests/run_tests.py --integration

# E2Eテストのみ実行
python tests/run_tests.py --e2e

# カバレッジレポートを生成
python tests/run_tests.py --coverage

# 詳細出力で実行
python tests/run_tests.py --verbose

# 最初の失敗で停止
python tests/run_tests.py --fail-fast
```

#### pytest を直接使用

```bash
# 全テストを実行
pytest tests/

# 特定のテストファイルを実行
pytest tests/unit/test_analyzer.py

# 詳細出力
pytest tests/ -v

# カバレッジレポート付き
pytest tests/ --cov=src --cov-report=html
```

### テスト設定とフィクスチャ

#### pytest.ini - テスト設定
`pytest.ini`ファイルでテストの実行設定を管理しています：

- **テストマーカー**: `unit`, `integration`, `e2e`, `slow`, `mock`
- **ログ設定**: 詳細なテストログ出力
- **警告フィルタ**: 不要な警告の抑制
- **カバレッジ設定**: コメントアウトされたカバレッジオプション

#### 開発依存関係 (requirements-dev.txt)
開発・テスト環境で必要なツール：

- **pytest関連**: pytest, pytest-cov, pytest-mock, pytest-xvfb
- **コード品質**: black, isort, flake8, mypy
- **ドキュメント**: sphinx, sphinx-rtd-theme
- **開発ツール**: ipython, jupyter

#### 共通フィクスチャ (conftest.py)
- `sample_text`: 日本語のサンプルテキスト
- `ner_analyzer`: NERAnalyzerインスタンス
- `batch_analyzer`: BatchNERAnalyzerインスタンス
- `temp_dir`: 一時ディレクトリ
- `sample_documents`: テスト用サンプル文書ファイル

#### モックとテストデータ
テストではHugging Faceモデルの実際のダウンロード・実行を避けるため、適切にモック化されています。

### カバレッジレポート

カバレッジレポートを生成すると、以下が作成されます：

```bash
# HTML形式のレポート
htmlcov/index.html

# XML形式のレポート  
coverage.xml

# ターミナル出力
コンソールにカバレッジ結果が表示
```

### 継続的インテグレーション

テストスイートはCI/CDパイプラインでの使用を想定しており、以下の特徴があります：

- **並列実行対応**: 独立したテストケース
- **エラーハンドリング**: 適切な例外処理とエラーメッセージ
- **リソース管理**: 一時ファイル・ディレクトリの自動クリーンアップ
- **モック化**: 外部依存関係の分離

### 開発ワークフロー

#### コード品質管理
Makefileには開発効率を向上させるコマンドが用意されています：

```bash
# コードフォーマット（black + isort）
make format

# リンティング（flake8 + mypy）
make lint

# 一時ファイルのクリーンアップ
make clean

# デモ実行
make demo

# サンプルバッチ分析
make batch-example
```

#### 推奨開発フロー
1. 機能開発前: `make install-dev` でテスト環境セットアップ
2. コード変更後: `make format` でコード整形
3. テスト実行: `make test` で全テスト確認
4. 品質チェック: `make lint` でコード品質確認
5. 最終確認: `make test-coverage` でカバレッジ確認

### テスト開発のガイドライン

新しい機能を追加する際は、以下のテストを含めてください：

1. **単体テスト**: 新しいクラス・メソッドの基本動作
2. **エラーケース**: 異常系・境界値のテスト
3. **統合テスト**: 他のモジュールとの連携
4. **E2Eテスト**: ユーザーインターフェース経由の動作確認

#### テストマーカーの使用
pytest.iniで定義されたマーカーを適切に使用してください：

```python
@pytest.mark.unit
def test_analyzer_basic_functionality():
    """単体テストのマーカー例"""
    pass

@pytest.mark.integration  
def test_batch_processing_workflow():
    """統合テストのマーカー例"""
    pass

@pytest.mark.e2e
def test_full_application_workflow():
    """E2Eテストのマーカー例"""
    pass

@pytest.mark.slow
def test_large_dataset_processing():
    """時間のかかるテストのマーカー例"""
    pass
```

## ライセンス

このプロジェクトで使用しているモデルとデータセットのライセンスに従ってください。

## 参考資料

- [Stockmark NER Wikipedia Dataset](https://github.com/stockmarkteam/ner-wikipedia-dataset)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [XLM-RoBERTa モデル](https://huggingface.co/tsmatz/xlm-roberta-ner-japanese)
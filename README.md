# Japanese Named Entity Recognition with Transformers

日本語テキストに対する固有表現抽出（NER）をTransformersライブラリで実行するプロジェクトです。

## 概要

このプロジェクトは、日本語の文書から人名、組織名、場所名などの固有表現を自動的に抽出します。Hugging Face Transformersライブラリと事前学習済みの日本語NERモデルを使用しています。

### 主な機能

- **単一テキストの簡単分析**: サンプルテキストでのデモ実行
- **バッチ処理**: 複数ファイル・ディレクトリの一括分析
- **詳細レポート**: CSV出力、統計分析、可視化グラフ生成
- **対応形式**: テキストファイル（.txt）、JSONファイル（.json）

## 使用モデル

- **メインモデル**: `tsmatz/xlm-roberta-ner-japanese`
  - Stockmarkの「Wikipediaを用いた日本語の固有表現抽出データセット」でファインチューニングされたモデル
  - 高い精度の固有表現抽出が可能

## 抽出可能な固有表現タイプ

| タグ | 説明 |
|------|------|
| PER | 人名 |
| ORG | 一般企業・組織 |
| ORG-P | 政治組織 |
| ORG-O | その他の組織 |
| LOC | 場所・地名 |
| INS | 施設・機関 |
| PRD | 製品 |
| EVT | イベント |

## セットアップ

### 必要な環境
- Python 3.7+
- CUDA対応GPU（推奨）

### 依存関係のインストール

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 簡単なデモ実行

```bash
python main.py
# または
python main.py --demo
```

サンプルテキスト（経済産業省のGENIAC事業に関する文章）から固有表現を抽出し、結果をコンソールに表示します。

#### 出力例

```
Running simple NER demo with sample text...
============================================================
社会実装 (PRD): score=0.9845
GENIAC (ORG): score=0.9921
東京 (LOC): score=0.9876
九段会館 (INS): score=0.9654
渡辺 琢也 (PER): score=0.9832
...
============================================================
Found 45 entities in total
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
usage: main.py [-h] [-o OUTPUT] [-m MODEL] [--demo] [input_path]

日本語固有表現抽出ツール

positional arguments:
  input_path            入力ファイルまたはディレクトリのパス（省略時はデモ実行）

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        出力ディレクトリ (デフォルト: output)
  -m MODEL, --model MODEL
                        NERモデル名
  --demo                簡単なデモを実行
```

## 出力ファイル

バッチ処理実行時、以下のファイルが出力されます：

### 1. CSV形式の詳細データ (`ner_results.csv`)

```csv
filename,word,entity_type,entity_description,score,start_pos,end_pos,analysis_time
document1.txt,東京,LOC,場所・地名,0.9876,23,25,2025-08-16T10:30:45.123456
document1.txt,GENIAC,ORG,一般企業・組織,0.9921,30,36,2025-08-16T10:30:45.123456
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
├── requirements.txt            # 依存関係
├── README.md                  # このファイル
├── batch_ner_analyzer.py      # レガシーファイル（参考用）
└── src/
    └── japanese_ner/          # メインパッケージ
        ├── __init__.py        # パッケージ初期化
        ├── analyzer.py        # コア NER 分析機能
        ├── batch_analyzer.py  # バッチ処理機能
        ├── utils.py          # ファイル処理ユーティリティ
        ├── report.py         # 統計・レポート生成
        └── visualization.py   # グラフ・可視化機能
```

### 主要クラスとモジュール

#### `NERAnalyzer` クラス (analyzer.py)
- `analyze(text)`: 単一テキストの固有表現抽出
- `get_entity_types()`: サポートされている固有表現タイプの取得

#### `BatchNERAnalyzer` クラス (batch_analyzer.py)
- `analyze_documents(input_path)`: 複数ドキュメントの一括分析
- `generate_full_report(input_path, output_dir)`: 完全な分析レポート生成

#### ユーティリティモジュール
- **utils.py**: ファイル読み込み、ディレクトリ処理
- **report.py**: 統計計算、CSV出力、マークダウンレポート生成
- **visualization.py**: matplotlib を使った可視化グラフ生成

## カスタマイズ

### 独自テキストでの簡単分析

`main.py`の`simple_ner_demo()`関数内の`text`変数を変更：

```python
text = """あなたの解析したいテキストをここに入力"""
```

### プログラムからの使用

```python
from src.japanese_ner import NERAnalyzer, BatchNERAnalyzer

# 単一テキスト分析
analyzer = NERAnalyzer()
entities = analyzer.analyze("分析したいテキスト")

# バッチ処理
batch_analyzer = BatchNERAnalyzer()
batch_analyzer.generate_full_report("input_directory/", "output_directory/")

# 異なるモデルの使用
custom_analyzer = NERAnalyzer("other-model-name")
```

## ライセンス

このプロジェクトで使用しているモデルとデータセットのライセンスに従ってください。

## 参考資料

- [Stockmark NER Wikipedia Dataset](https://github.com/stockmarkteam/ner-wikipedia-dataset)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [XLM-RoBERTa モデル](https://huggingface.co/tsmatz/xlm-roberta-ner-japanese)
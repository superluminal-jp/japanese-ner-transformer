"""
Japanese Named Entity Recognition Tool

Main entry point for the Japanese NER analysis tool.
Supports both simple demo mode and batch processing mode.
"""

import argparse
import sys
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from japanese_ner import NERAnalyzer, BatchNERAnalyzer


def simple_ner_demo():
    """
    Simple NER analysis demo with sample text for evaluation.
    """
    # Comprehensive evaluation text covering all entity types
    text = """2024年11月15日、東京国際フォーラムで開催されたAI技術カンファレンスにおいて、OpenAI社のCEOサム・アルトマン氏が基調講演を行いました。

同イベントには、トヨタ自動車株式会社の豊田章男会長、ソフトバンクグループ株式会社の孫正義社長、経済産業省の田中智子局長らが参加しました。また、自由民主党のデジタル推進委員会からも代表者が出席し、政府のAI戦略について議論が交わされました。

会場では、最新のGPT-5モデルやGoogle Bardの新機能が紹介され、参加者から大きな注目を集めました。特にApple社が発表したApple Intelligenceは革新的な製品として評価されています。

講演後、六本木ヒルズ森タワーで開催されたレセションでは、マイクロソフト日本法人の樋口泰行社長と、東京大学の五神真総長が対談を行いました。

このカンファレンスは2025年春に大阪で開催される次回の関西AI展の前哨戦として位置づけられており、京都大学や奈良先端科学技術大学院大学の研究チームも参加予定です。

なお、会場ではChatGPTやClaudeなどの生成AIサービスの実演も行われ、参加者は最新技術を直接体験することができました。"""

    print("Running simple NER demo with sample text...")
    print("=" * 60)
    
    analyzer = NERAnalyzer()
    entities = analyzer.analyze(text)
    
    for entity in entities:
        print(f"{entity['word']} ({entity['entity_type']}): score={entity['score']:.4f}")
    
    print("\n" + "=" * 60)
    print(f"Found {len(entities)} entities in total")


def batch_ner_analysis(input_path: str, output_dir: str, model_name: str):
    """
    Run batch NER analysis on multiple documents.
    
    Args:
        input_path: Path to input file or directory
        output_dir: Output directory for results
        model_name: Name of the NER model to use
    """
    analyzer = BatchNERAnalyzer(model_name)
    analyzer.generate_full_report(input_path, output_dir)


def main():
    parser = argparse.ArgumentParser(description='日本語固有表現抽出ツール')
    parser.add_argument('input_path', nargs='?', help='入力ファイルまたはディレクトリのパス（省略時はデモ実行）')
    parser.add_argument('-o', '--output', default='output', help='出力ディレクトリ (デフォルト: output)')
    parser.add_argument('-m', '--model', default='tsmatz/xlm-roberta-ner-japanese', help='NERモデル名')
    parser.add_argument('--demo', action='store_true', help='簡単なデモを実行')
    
    args = parser.parse_args()
    
    if args.demo or args.input_path is None:
        # デモモード：元のmain.pyの機能
        simple_ner_demo()
    else:
        # バッチ処理モード：batch_ner_analyzer.pyの機能
        batch_ner_analysis(args.input_path, args.output, args.model)


if __name__ == "__main__":
    main()

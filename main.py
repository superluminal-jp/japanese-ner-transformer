"""
Japanese Named Entity Recognition Tool

Main entry point for the Japanese NER analysis tool.
Supports both simple demo mode and batch processing mode.
"""

import argparse
import sys
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from japanese_ner import NERAnalyzer




def batch_ner_analysis(input_path: str, output_dir: str, model_name: str):
    """
    Run batch NER analysis on multiple documents.

    Args:
        input_path: Path to input file or directory
        output_dir: Output directory for results
        model_name: Name of the NER model to use
    """
    analyzer = NERAnalyzer(model_name)
    analyzer.generate_full_report(input_path, output_dir)


def main():
    parser = argparse.ArgumentParser(description="日本語固有表現抽出ツール")
    parser.add_argument(
        "input_path",
        nargs="?",
        default="demo",
        help="入力ファイルまたはディレクトリのパス（デフォルト: demo）",
    )
    parser.add_argument(
        "-o", "--output", default="output", help="出力ディレクトリ (デフォルト: output)"
    )
    parser.add_argument(
        "-m", "--model", default="tsmatz/xlm-roberta-ner-japanese", help="NERモデル名"
    )

    args = parser.parse_args()

    # 統一されたバッチ処理
    batch_ner_analysis(args.input_path, args.output, args.model)


if __name__ == "__main__":
    main()

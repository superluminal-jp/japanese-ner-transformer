import os
import csv
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

class BatchNERAnalyzer:
    def __init__(self, model_name: str = "tsmatz/xlm-roberta-ner-japanese"):
        """
        複数ドキュメントのNER分析を行うクラス
        
        Args:
            model_name: 使用するNERモデル名
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.ner = pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
        )
        
        # 固有表現タイプの説明
        self.entity_descriptions = {
            'PER': '人名',
            'ORG': '一般企業・組織',
            'ORG-P': '政治組織',
            'ORG-O': 'その他の組織',
            'LOC': '場所・地名',
            'INS': '施設・機関',
            'PRD': '製品',
            'EVT': 'イベント'
        }

    def read_documents(self, input_path: str) -> List[Dict[str, str]]:
        """
        入力パスからドキュメントを読み込む
        
        Args:
            input_path: 入力ファイルまたはディレクトリのパス
            
        Returns:
            ドキュメントのリスト（各要素は{'filename': str, 'content': str}）
        """
        documents = []
        path = Path(input_path)
        
        if path.is_file():
            # 単一ファイルの場合
            if path.suffix == '.txt':
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                documents.append({
                    'filename': path.name,
                    'content': content
                })
            elif path.suffix == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for i, item in enumerate(data):
                            documents.append({
                                'filename': f"{path.stem}_{i+1}",
                                'content': str(item)
                            })
                    else:
                        documents.append({
                            'filename': path.name,
                            'content': str(data)
                        })
        elif path.is_dir():
            # ディレクトリの場合
            for file_path in path.glob('*.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                documents.append({
                    'filename': file_path.name,
                    'content': content
                })
        else:
            raise ValueError(f"Invalid input path: {input_path}")
            
        return documents

    def analyze_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        複数ドキュメントのNER分析を実行
        
        Args:
            documents: ドキュメントのリスト
            
        Returns:
            分析結果のリスト
        """
        results = []
        
        for doc in documents:
            print(f"Analyzing: {doc['filename']}")
            
            # NER実行
            ner_results = self.ner(doc['content'])
            
            # 結果を整理
            entities = []
            for entity in ner_results:
                entities.append({
                    'word': entity['word'],
                    'entity_type': entity['entity_group'],
                    'score': entity['score'],
                    'start': entity.get('start', 0),
                    'end': entity.get('end', 0)
                })
            
            results.append({
                'filename': doc['filename'],
                'content': doc['content'],
                'entities': entities,
                'entity_count': len(entities),
                'analysis_time': datetime.now().isoformat()
            })
            
        return results

    def save_to_csv(self, results: List[Dict[str, Any]], output_path: str):
        """
        分析結果をCSVファイルに保存
        
        Args:
            results: 分析結果
            output_path: 出力ファイルパス
        """
        csv_data = []
        
        for result in results:
            for entity in result['entities']:
                csv_data.append({
                    'filename': result['filename'],
                    'word': entity['word'],
                    'entity_type': entity['entity_type'],
                    'entity_description': self.entity_descriptions.get(entity['entity_type'], '不明'),
                    'score': entity['score'],
                    'start_pos': entity['start'],
                    'end_pos': entity['end'],
                    'analysis_time': result['analysis_time']
                })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"CSV saved to: {output_path}")

    def generate_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        統計分析を実行
        
        Args:
            results: 分析結果
            
        Returns:
            統計情報
        """
        stats = {
            'total_documents': len(results),
            'total_entities': sum(result['entity_count'] for result in results),
            'entity_type_counts': Counter(),
            'entity_word_counts': Counter(),
            'documents_stats': [],
            'avg_entities_per_doc': 0,
            'most_common_entities': [],
            'entity_type_distribution': {}
        }
        
        all_entities = []
        for result in results:
            doc_entities = [entity['entity_type'] for entity in result['entities']]
            all_entities.extend(doc_entities)
            
            stats['entity_type_counts'].update(doc_entities)
            stats['entity_word_counts'].update([entity['word'] for entity in result['entities']])
            
            stats['documents_stats'].append({
                'filename': result['filename'],
                'entity_count': result['entity_count'],
                'unique_entity_types': len(set(doc_entities)),
                'text_length': len(result['content'])
            })
        
        if stats['total_documents'] > 0:
            stats['avg_entities_per_doc'] = stats['total_entities'] / stats['total_documents']
        
        stats['most_common_entities'] = stats['entity_word_counts'].most_common(10)
        
        # エンティティタイプの分布
        total_entities = sum(stats['entity_type_counts'].values())
        if total_entities > 0:
            stats['entity_type_distribution'] = {
                entity_type: count / total_entities * 100
                for entity_type, count in stats['entity_type_counts'].items()
            }
        
        return stats

    def create_visualizations(self, stats: Dict[str, Any], output_dir: str):
        """
        統計データの可視化
        
        Args:
            stats: 統計情報
            output_dir: 出力ディレクトリ
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Set Japanese font for matplotlib
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Hiragino Sans']
        
        # 1. エンティティタイプ分布
        if stats['entity_type_counts']:
            plt.figure(figsize=(10, 6))
            entity_types = list(stats['entity_type_counts'].keys())
            counts = list(stats['entity_type_counts'].values())
            
            bars = plt.bar(entity_types, counts)
            plt.title('固有表現タイプ別出現回数')
            plt.xlabel('固有表現タイプ')
            plt.ylabel('出現回数')
            plt.xticks(rotation=45)
            
            # バーの上に値を表示
            for bar, count in zip(bars, counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(output_path / 'entity_type_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. 最頻出固有表現
        if stats['most_common_entities']:
            plt.figure(figsize=(12, 8))
            words, counts = zip(*stats['most_common_entities'])
            
            bars = plt.barh(range(len(words)), counts)
            plt.yticks(range(len(words)), words)
            plt.title('最頻出固有表現 (Top 10)')
            plt.xlabel('出現回数')
            plt.gca().invert_yaxis()
            
            # バーの右側に値を表示
            for i, (bar, count) in enumerate(zip(bars, counts)):
                plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                        str(count), ha='left', va='center')
            
            plt.tight_layout()
            plt.savefig(output_path / 'most_common_entities.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. ドキュメント別エンティティ数
        if stats['documents_stats']:
            plt.figure(figsize=(12, 6))
            filenames = [doc['filename'] for doc in stats['documents_stats']]
            entity_counts = [doc['entity_count'] for doc in stats['documents_stats']]
            
            bars = plt.bar(range(len(filenames)), entity_counts)
            plt.title('ドキュメント別固有表現数')
            plt.xlabel('ドキュメント')
            plt.ylabel('固有表現数')
            plt.xticks(range(len(filenames)), filenames, rotation=45, ha='right')
            
            for bar, count in zip(bars, entity_counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(output_path / 'entities_per_document.png', dpi=300, bbox_inches='tight')
            plt.close()

    def generate_report(self, stats: Dict[str, Any], output_path: str):
        """
        分析レポートを生成
        
        Args:
            stats: 統計情報
            output_path: レポート出力パス
        """
        report = f"""# 固有表現抽出 分析レポート

## 分析概要
- **分析日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **使用モデル**: {self.model_name}
- **総ドキュメント数**: {stats['total_documents']}
- **総固有表現数**: {stats['total_entities']}
- **ドキュメント平均固有表現数**: {stats['avg_entities_per_doc']:.2f}

## 固有表現タイプ別統計

| タイプ | 説明 | 出現回数 | 割合 |
|--------|------|----------|------|
"""
        
        for entity_type, count in stats['entity_type_counts'].most_common():
            description = self.entity_descriptions.get(entity_type, '不明')
            percentage = stats['entity_type_distribution'].get(entity_type, 0)
            report += f"| {entity_type} | {description} | {count} | {percentage:.1f}% |\n"
        
        report += f"""
## 最頻出固有表現 (Top 10)

| 順位 | 固有表現 | 出現回数 |
|------|----------|----------|
"""
        
        for i, (word, count) in enumerate(stats['most_common_entities'], 1):
            report += f"| {i} | {word} | {count} |\n"
        
        report += f"""
## ドキュメント別詳細

| ファイル名 | 固有表現数 | ユニークタイプ数 | 文字数 |
|------------|------------|------------------|--------|
"""
        
        for doc_stat in stats['documents_stats']:
            report += f"| {doc_stat['filename']} | {doc_stat['entity_count']} | {doc_stat['unique_entity_types']} | {doc_stat['text_length']} |\n"
        
        report += f"""
## 分析結果ファイル

1. **CSV形式**: 全固有表現の詳細データ
2. **可視化グラフ**: 
   - `entity_type_distribution.png`: 固有表現タイプ別分布
   - `most_common_entities.png`: 最頻出固有表現
   - `entities_per_document.png`: ドキュメント別固有表現数

## 使用した固有表現タイプ

"""
        
        for entity_type, description in self.entity_descriptions.items():
            report += f"- **{entity_type}**: {description}\n"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='複数ドキュメントのNER分析ツール')
    parser.add_argument('input_path', help='入力ファイルまたはディレクトリのパス')
    parser.add_argument('-o', '--output', default='output', help='出力ディレクトリ (デフォルト: output)')
    parser.add_argument('-m', '--model', default='tsmatz/xlm-roberta-ner-japanese', help='NERモデル名')
    
    args = parser.parse_args()
    
    # 出力ディレクトリ作成
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # 分析器初期化
    analyzer = BatchNERAnalyzer(args.model)
    
    # ドキュメント読み込み
    print(f"Reading documents from: {args.input_path}")
    documents = analyzer.read_documents(args.input_path)
    print(f"Found {len(documents)} documents")
    
    # NER分析実行
    print("Starting NER analysis...")
    results = analyzer.analyze_documents(documents)
    
    # CSV出力
    csv_path = output_dir / 'ner_results.csv'
    analyzer.save_to_csv(results, str(csv_path))
    
    # 統計分析
    print("Generating statistics...")
    stats = analyzer.generate_statistics(results)
    
    # 可視化
    print("Creating visualizations...")
    analyzer.create_visualizations(stats, str(output_dir))
    
    # レポート生成
    print("Generating report...")
    report_path = output_dir / 'analysis_report.md'
    analyzer.generate_report(stats, str(report_path))
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print(f"- CSV: {csv_path}")
    print(f"- Report: {report_path}")
    print(f"- Visualizations: {output_dir}/*.png")


if __name__ == "__main__":
    main()
"""
Report generation and statistics calculation for NER analysis.
"""

import pandas as pd
from collections import Counter
from datetime import datetime
from typing import List, Dict, Any


def calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics from NER analysis results.
    
    Args:
        results: List of analysis results from documents
        
    Returns:
        Dictionary containing various statistics
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
    
    # Calculate entity type distribution percentages
    total_entities = sum(stats['entity_type_counts'].values())
    if total_entities > 0:
        stats['entity_type_distribution'] = {
            entity_type: count / total_entities * 100
            for entity_type, count in stats['entity_type_counts'].items()
        }
    
    return stats


def save_csv_report(results: List[Dict[str, Any]], output_path: str, entity_descriptions: Dict[str, str]):
    """
    Save detailed analysis results to CSV file.
    
    Args:
        results: Analysis results
        output_path: Path to save CSV file
        entity_descriptions: Mapping of entity types to descriptions
    """
    csv_data = []
    
    for result in results:
        for entity in result['entities']:
            csv_data.append({
                'filename': result['filename'],
                'word': entity['word'],
                'entity_type': entity['entity_type'],
                'entity_description': entity_descriptions.get(entity['entity_type'], '不明'),
                'score': entity['score'],
                'start_pos': entity['start'],
                'end_pos': entity['end'],
                'analysis_time': result['analysis_time']
            })
    
    df = pd.DataFrame(csv_data)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"CSV saved to: {output_path}")


def generate_markdown_report(stats: Dict[str, Any], model_name: str, entity_descriptions: Dict[str, str]) -> str:
    """
    Generate a comprehensive markdown report.
    
    Args:
        stats: Statistics dictionary
        model_name: Name of the model used
        entity_descriptions: Entity type descriptions
        
    Returns:
        Markdown formatted report string
    """
    report = f"""# 固有表現抽出 分析レポート

## 分析概要
- **分析日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **使用モデル**: {model_name}
- **総ドキュメント数**: {stats['total_documents']}
- **総固有表現数**: {stats['total_entities']}
- **ドキュメント平均固有表現数**: {stats['avg_entities_per_doc']:.2f}

## 固有表現タイプ別統計

| タイプ | 説明 | 出現回数 | 割合 |
|--------|------|----------|------|
"""
    
    for entity_type, count in stats['entity_type_counts'].most_common():
        description = entity_descriptions.get(entity_type, '不明')
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
    
    for entity_type, description in entity_descriptions.items():
        report += f"- **{entity_type}**: {description}\n"
    
    return report


def save_markdown_report(stats: Dict[str, Any], output_path: str, model_name: str, entity_descriptions: Dict[str, str]):
    """
    Save markdown report to file.
    
    Args:
        stats: Statistics dictionary
        output_path: Path to save the report
        model_name: Name of the model used
        entity_descriptions: Entity type descriptions
    """
    report = generate_markdown_report(stats, model_name, entity_descriptions)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Report saved to: {output_path}")
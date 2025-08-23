"""
Report generation and statistics calculation for NER analysis.
"""

import pandas as pd
import math
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from .logger import get_logger


def calculate_tf_idf_metrics(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate TF, IDF, and DF metrics for entities across documents.
    
    Args:
        results: List of analysis results from documents
        
    Returns:
        Dictionary containing tf, idf, df metrics for each entity
    """
    # Collect all entities and their document frequencies
    entity_document_count = defaultdict(set)  # entity -> set of documents containing it
    entity_term_counts = defaultdict(lambda: defaultdict(int))  # document -> entity -> count
    document_entity_counts = defaultdict(int)  # document -> total entity count
    
    total_documents = len(results)
    
    # First pass: collect entity frequencies
    for result in results:
        filename = result['filename']
        entities = result['entities']
        
        # Count entities in this document
        entity_counts = Counter(entity['word'] for entity in entities)
        document_entity_counts[filename] = sum(entity_counts.values())
        
        for entity_word, count in entity_counts.items():
            entity_document_count[entity_word].add(filename)
            entity_term_counts[filename][entity_word] = count
    
    # Calculate metrics for each entity
    metrics = {}
    
    for entity_word in entity_document_count:
        # Document Frequency (DF): number of documents containing the entity
        df = len(entity_document_count[entity_word])
        
        # Inverse Document Frequency (IDF): log(N/DF)
        idf = math.log(total_documents / df) if df > 0 else 0
        
        # Calculate TF for each document containing this entity
        tf_scores = {}
        for filename in entity_document_count[entity_word]:
            # Term Frequency (TF): (entity_count / total_entities_in_doc)
            entity_count = entity_term_counts[filename][entity_word]
            total_entities_in_doc = document_entity_counts[filename]
            tf = entity_count / total_entities_in_doc if total_entities_in_doc > 0 else 0
            tf_scores[filename] = tf
        
        # Store metrics
        metrics[entity_word] = {
            'df': df,
            'idf': idf,
            'tf_scores': tf_scores,
            'tf_idf_scores': {filename: tf_scores[filename] * idf for filename in tf_scores}
        }
    
    return metrics


def calculate_quality_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate data quality and model performance metrics.
    
    Args:
        results: List of analysis results from documents
        
    Returns:
        Dictionary containing quality metrics
    """
    if not results:
        return {}
    
    all_scores = []
    entity_lengths = []
    entity_positions = []
    score_by_type = defaultdict(list)
    
    for result in results:
        doc_length = len(result['content'])
        for entity in result['entities']:
            score = entity['score']
            all_scores.append(score)
            
            # Entity characteristics
            entity_text = entity['word']
            entity_lengths.append(len(entity_text))
            
            # Position analysis (relative position in document)
            start_pos = entity.get('start', 0)
            relative_pos = start_pos / doc_length if doc_length > 0 else 0
            entity_positions.append(relative_pos)
            
            # Score by entity type
            entity_type = entity['entity_type']
            score_by_type[entity_type].append(score)
    
    # Calculate statistics
    quality_metrics = {
        'score_statistics': {
            'mean': np.mean(all_scores) if all_scores else 0,
            'median': np.median(all_scores) if all_scores else 0,
            'std_dev': np.std(all_scores) if all_scores else 0,
            'min': np.min(all_scores) if all_scores else 0,
            'max': np.max(all_scores) if all_scores else 0,
            'q25': np.percentile(all_scores, 25) if all_scores else 0,
            'q75': np.percentile(all_scores, 75) if all_scores else 0
        },
        'entity_length_stats': {
            'mean': np.mean(entity_lengths) if entity_lengths else 0,
            'median': np.median(entity_lengths) if entity_lengths else 0,
            'std_dev': np.std(entity_lengths) if entity_lengths else 0
        },
        'position_distribution': {
            'early_doc': sum(1 for p in entity_positions if p < 0.33) / len(entity_positions) if entity_positions else 0,
            'mid_doc': sum(1 for p in entity_positions if 0.33 <= p < 0.67) / len(entity_positions) if entity_positions else 0,
            'late_doc': sum(1 for p in entity_positions if p >= 0.67) / len(entity_positions) if entity_positions else 0
        },
        'confidence_by_type': {
            entity_type: {
                'mean_confidence': np.mean(scores),
                'min_confidence': np.min(scores),
                'max_confidence': np.max(scores),
                'count': len(scores)
            }
            for entity_type, scores in score_by_type.items()
        },
        'high_confidence_entities': len([s for s in all_scores if s > 0.9]),
        'low_confidence_entities': len([s for s in all_scores if s < 0.7]),
        'total_entities': len(all_scores)
    }
    
    return quality_metrics


def calculate_entity_relationships(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze relationships and co-occurrence patterns between entities.
    
    Args:
        results: List of analysis results from documents
        
    Returns:
        Dictionary containing relationship analysis
    """
    if not results:
        return {}
    
    # Co-occurrence analysis (entities appearing in same document)
    doc_entities = []
    entity_pairs = Counter()
    entity_contexts = defaultdict(list)
    
    for result in results:
        filename = result['filename']
        entities = result['entities']
        
        # Extract entity words for this document
        doc_entity_words = [entity['word'] for entity in entities]
        doc_entities.append(set(doc_entity_words))
        
        # Count co-occurrences (pairs of entities in same document)
        for i, entity1 in enumerate(doc_entity_words):
            for j, entity2 in enumerate(doc_entity_words):
                if i < j:  # Avoid duplicates and self-pairs
                    pair = tuple(sorted([entity1, entity2]))
                    entity_pairs[pair] += 1
        
        # Collect context information
        content = result['content']
        for entity in entities:
            start = entity.get('start', 0)
            end = entity.get('end', start + len(entity['word']))
            
            # Extract surrounding context (50 characters before and after)
            context_start = max(0, start - 50)
            context_end = min(len(content), end + 50)
            context = content[context_start:context_end]
            
            entity_contexts[entity['word']].append({
                'document': filename,
                'context': context,
                'position': start
            })
    
    # Calculate entity co-occurrence strength
    total_docs = len(results)
    entity_cooccurrence = {}
    
    for (entity1, entity2), count in entity_pairs.most_common(10):
        # Calculate how often these entities appear together vs separately
        docs_with_both = count
        docs_with_entity1 = sum(1 for doc_ents in doc_entities if entity1 in doc_ents)
        docs_with_entity2 = sum(1 for doc_ents in doc_entities if entity2 in doc_ents)
        
        # Jaccard similarity
        union_size = docs_with_entity1 + docs_with_entity2 - docs_with_both
        jaccard = docs_with_both / union_size if union_size > 0 else 0
        
        entity_cooccurrence[f"{entity1} + {entity2}"] = {
            'cooccurrence_count': docs_with_both,
            'entity1_docs': docs_with_entity1,
            'entity2_docs': docs_with_entity2,
            'jaccard_similarity': jaccard,
            'cooccurrence_rate': docs_with_both / total_docs
        }
    
    return {
        'top_cooccurrences': entity_cooccurrence,
        'entity_contexts': dict(entity_contexts),
        'total_unique_pairs': len(entity_pairs)
    }


def generate_insights_and_recommendations(stats: Dict[str, Any]) -> List[str]:
    """
    Generate actionable insights and recommendations based on analysis.
    
    Args:
        stats: Complete statistics dictionary
        
    Returns:
        List of insight strings
    """
    insights = []
    
    # Document coverage insights
    total_docs = stats.get('total_documents', 0)
    total_entities = stats.get('total_entities', 0)
    
    if total_docs == 1:
        insights.append("⚠️ 単一文書分析: TF-IDFスコアは複数文書がある場合により意味を持ちます")
    
    # Entity distribution insights
    entity_type_counts = stats.get('entity_type_counts', Counter())
    if entity_type_counts:
        most_common_type = entity_type_counts.most_common(1)[0]
        percentage = (most_common_type[1] / total_entities) * 100
        if percentage > 60:
            insights.append(f"{most_common_type[0]}タイプが全体の{percentage:.1f}%を占めており、特定分野に偏っています")
    
    # Quality insights
    quality_metrics = stats.get('quality_metrics', {})
    if quality_metrics:
        score_stats = quality_metrics.get('score_statistics', {})
        mean_score = score_stats.get('mean', 0)
        low_confidence = quality_metrics.get('low_confidence_entities', 0)
        high_confidence = quality_metrics.get('high_confidence_entities', 0)
        
        if mean_score > 0.9:
            insights.append(f"平均信頼度が{mean_score:.3f}と高く、抽出品質は良好です")
        elif mean_score < 0.7:
            insights.append(f"⚠️ 平均信頼度が{mean_score:.3f}と低く、抽出精度の改善が必要です")
        
        if low_confidence > total_entities * 0.2:
            insights.append(f"{low_confidence}個の低信頼度エンティティがあり、手動確認を推奨します")
    
    # Entity relationship insights
    relationships = stats.get('entity_relationships', {})
    if relationships:
        top_cooccurrences = relationships.get('top_cooccurrences', {})
        if top_cooccurrences:
            strongest_pair = list(top_cooccurrences.keys())[0]
            insights.append(f"最も強い関連性: {strongest_pair}")
    
    # Recommendations
    insights.append("\n## 推奨アクション")
    
    if total_docs < 5:
        insights.append("- より多くの文書を分析してTF-IDF分析の精度を向上させる")
    
    if quality_metrics.get('low_confidence_entities', 0) > 0:
        insights.append("- 信頼度の低いエンティティを手動で確認し、誤検出を修正する")
    
    insights.append("- 抽出されたエンティティを業務固有の辞書で補完する")
    insights.append("- 定期的な分析により新しいエンティティパターンを発見する")
    
    return insights


def calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics from NER analysis results.
    
    Args:
        results: List of analysis results from documents
        
    Returns:
        Dictionary containing various statistics
    """
    # Calculate TF-IDF metrics
    tf_idf_metrics = calculate_tf_idf_metrics(results)
    
    # Calculate quality metrics
    quality_metrics = calculate_quality_metrics(results)
    
    # Calculate entity relationships
    entity_relationships = calculate_entity_relationships(results)
    
    stats = {
        'total_documents': len(results),
        'total_entities': sum(result['entity_count'] for result in results),
        'entity_type_counts': Counter(),
        'entity_word_counts': Counter(),
        'documents_stats': [],
        'avg_entities_per_doc': 0,
        'most_common_entities': [],
        'entity_type_distribution': {},
        'tf_idf_metrics': tf_idf_metrics,
        'quality_metrics': quality_metrics,
        'entity_relationships': entity_relationships
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
    
    # Generate insights and recommendations
    insights = generate_insights_and_recommendations(stats)
    stats['insights_and_recommendations'] = insights
    
    return stats


def save_csv_report(results: List[Dict[str, Any]], output_path: str, entity_descriptions: Dict[str, str]):
    """
    Save detailed analysis results to CSV file with frequency and TF-IDF rankings.
    
    Args:
        results: Analysis results
        output_path: Path to save CSV file
        entity_descriptions: Mapping of entity types to descriptions
    """
    # Calculate TF-IDF metrics for the CSV
    tf_idf_metrics = calculate_tf_idf_metrics(results)
    
    # Calculate entity frequency across all documents
    entity_frequency = Counter()
    for result in results:
        for entity in result['entities']:
            entity_frequency[entity['word']] += 1
    
    # Create frequency rankings (1 = most frequent)
    frequency_rankings = {}
    for rank, (entity_word, count) in enumerate(entity_frequency.most_common(), 1):
        frequency_rankings[entity_word] = rank
    
    # Collect all TF-IDF scores for ranking
    all_tf_idf_scores = []
    for result in results:
        filename = result['filename']
        for entity in result['entities']:
            entity_word = entity['word']
            metrics = tf_idf_metrics.get(entity_word, {})
            tf_idf_score = metrics.get('tf_idf_scores', {}).get(filename, 0.0)
            
            all_tf_idf_scores.append({
                'entity_word': entity_word,
                'filename': filename,
                'tf_idf_score': tf_idf_score
            })
    
    # Sort by TF-IDF score descending and assign rankings
    sorted_tf_idf = sorted(all_tf_idf_scores, key=lambda x: x['tf_idf_score'], reverse=True)
    tf_idf_rankings = {}
    for rank, item in enumerate(sorted_tf_idf, 1):
        key = f"{item['entity_word']}_{item['filename']}"
        tf_idf_rankings[key] = rank
    
    csv_data = []
    
    for result in results:
        filename = result['filename']
        for entity in result['entities']:
            entity_word = entity['word']
            
            # Get TF-IDF metrics for this entity
            metrics = tf_idf_metrics.get(entity_word, {})
            tf_score = metrics.get('tf_scores', {}).get(filename, 0.0)
            idf_score = metrics.get('idf', 0.0)
            df_value = metrics.get('df', 0)
            tf_idf_score = metrics.get('tf_idf_scores', {}).get(filename, 0.0)
            
            # Get rankings
            freq_rank = frequency_rankings.get(entity_word, 0)
            tf_idf_rank = tf_idf_rankings.get(f"{entity_word}_{filename}", 0)
            
            csv_data.append({
                'filename': result['filename'],
                'word': entity['word'],
                'entity_type': entity['entity_type'],
                'entity_description': entity_descriptions.get(entity['entity_type'], '不明'),
                'score': entity['score'],
                'start_pos': entity['start'],
                'end_pos': entity['end'],
                'analysis_time': result['analysis_time'],
                'frequency_rank': freq_rank,
                'tf_idf_rank': tf_idf_rank,
                'tf': round(tf_score, 6),
                'idf': round(idf_score, 6),
                'df': df_value,
                'tf_idf': round(tf_idf_score, 6)
            })
    
    df = pd.DataFrame(csv_data)
    df.to_csv(output_path, index=False, encoding='utf-8')
    logger = get_logger("report")
    logger.info(f"CSV saved to: {output_path}")


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

| 順位 | 固有表現 | 出現回数 | DF | 平均IDF | 最大TF-IDF |
|------|----------|----------|----|---------|-----------| 
"""
    
    for i, (word, count) in enumerate(stats['most_common_entities'], 1):
        # Get TF-IDF metrics for this entity (if available)
        tf_idf_metrics = stats.get('tf_idf_metrics', {})
        metrics = tf_idf_metrics.get(word, {})
        df_value = metrics.get('df', 0)
        idf_value = metrics.get('idf', 0.0)
        tf_idf_scores = metrics.get('tf_idf_scores', {})
        max_tf_idf = max(tf_idf_scores.values()) if tf_idf_scores else 0.0
        
        report += f"| {i} | {word} | {count} | {df_value} | {idf_value:.4f} | {max_tf_idf:.4f} |\n"
    
    report += f"""
## ドキュメント別詳細

| ファイル名 | 固有表現数 | ユニークタイプ数 | 文字数 |
|------------|------------|------------------|--------|
"""
    
    for doc_stat in stats['documents_stats']:
        report += f"| {doc_stat['filename']} | {doc_stat['entity_count']} | {doc_stat['unique_entity_types']} | {doc_stat['text_length']} |\n"
    
    # Add Quality Analysis section
    quality_metrics = stats.get('quality_metrics', {})
    if quality_metrics:
        report += f"""
## 品質分析

### 信頼度統計
"""
        score_stats = quality_metrics.get('score_statistics', {})
        report += f"""
- **平均信頼度**: {score_stats.get('mean', 0):.3f}
- **中央値**: {score_stats.get('median', 0):.3f}
- **標準偏差**: {score_stats.get('std_dev', 0):.3f}
- **最小値**: {score_stats.get('min', 0):.3f}
- **最大値**: {score_stats.get('max', 0):.3f}
- **第1四分位**: {score_stats.get('q25', 0):.3f}
- **第3四分位**: {score_stats.get('q75', 0):.3f}

### 品質指標
- **高信頼度エンティティ (>0.9)**: {quality_metrics.get('high_confidence_entities', 0)}個
- **低信頼度エンティティ (<0.7)**: {quality_metrics.get('low_confidence_entities', 0)}個
- **品質率**: {(quality_metrics.get('high_confidence_entities', 0) / max(quality_metrics.get('total_entities', 1), 1) * 100):.1f}%

### エンティティタイプ別信頼度

| エンティティタイプ | 平均信頼度 | 最小信頼度 | 最大信頼度 | 数量 |
|-------------------|------------|------------|------------|------|
"""
        
        confidence_by_type = quality_metrics.get('confidence_by_type', {})
        for entity_type, conf_stats in confidence_by_type.items():
            report += f"| {entity_type} | {conf_stats['mean_confidence']:.3f} | {conf_stats['min_confidence']:.3f} | {conf_stats['max_confidence']:.3f} | {conf_stats['count']} |\n"
        
    
    
    # Add TF-IDF analysis section (enhanced)
    tf_idf_metrics = stats.get('tf_idf_metrics', {})
    if tf_idf_metrics:
        report += f"""
## TF-IDF分析

**TF (Term Frequency / 単語頻度)**: 特定の文書内でその固有表現が出現する頻度を示します。TF値が高いほど、その文書においてその固有表現が重要であることを意味します。

**IDF (Inverse Document Frequency / 逆文書頻度)**: すべての文書において、その固有表現がどれほど珍しいかを示します。IDF値が高いほど、その固有表現は特定の文書に特有であることを意味します。

**TF-IDF**: TFとIDFを掛け合わせた指標で、特定の文書における固有表現の重要度を総合的に評価します。TF-IDF値が高いほど、その固有表現はその文書の特徴を表す重要な語句であることを示します。

### 高TF-IDFスコア固有表現 (Top 10)

| 順位 | 固有表現 | ドキュメント | TF | IDF | TF-IDF |
|------|----------|--------------|----|----|--------|
"""
        
        # Collect all TF-IDF scores and sort them
        all_tf_idf_scores = []
        for entity_word, metrics in tf_idf_metrics.items():
            tf_idf_scores = metrics.get('tf_idf_scores', {})
            for filename, score in tf_idf_scores.items():
                all_tf_idf_scores.append({
                    'entity': entity_word,
                    'document': filename,
                    'tf': metrics.get('tf_scores', {}).get(filename, 0.0),
                    'idf': metrics.get('idf', 0.0),
                    'tf_idf': score
                })
        
        # Sort by TF-IDF score descending and take top 10
        top_tf_idf = sorted(all_tf_idf_scores, key=lambda x: x['tf_idf'], reverse=True)[:10]
        
        for i, item in enumerate(top_tf_idf, 1):
            report += f"| {i} | {item['entity']} | {item['document']} | {item['tf']:.4f} | {item['idf']:.4f} | {item['tf_idf']:.4f} |\n"
    
    # Add insights and recommendations
    insights = stats.get('insights_and_recommendations', [])
    if insights:
        report += f"""
## 分析結果と推奨事項

"""
        for insight in insights:
            report += f"{insight}\n"
    
    report += f"""
## 分析結果ファイル

1. **CSV形式**: 全固有表現の詳細データ (頻度ランキング, TF-IDFランキング, TF, IDF, DF, TF-IDF値を含む)

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
    
    logger = get_logger("report")
    logger.info(f"Report saved to: {output_path}")
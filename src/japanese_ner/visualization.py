"""
Visualization functions for NER analysis results.
"""

from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt


def setup_japanese_fonts():
    """Setup matplotlib for Japanese text rendering."""
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Hiragino Sans']


def create_entity_type_chart(stats: Dict[str, Any], output_dir: Path):
    """
    Create bar chart for entity type distribution.
    
    Args:
        stats: Statistics dictionary containing entity type counts
        output_dir: Directory to save the chart
    """
    if not stats['entity_type_counts']:
        return
        
    setup_japanese_fonts()
    
    plt.figure(figsize=(10, 6))
    entity_types = list(stats['entity_type_counts'].keys())
    counts = list(stats['entity_type_counts'].values())
    
    bars = plt.bar(entity_types, counts)
    plt.title('固有表現タイプ別出現回数')
    plt.xlabel('固有表現タイプ')
    plt.ylabel('出現回数')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'entity_type_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_common_entities_chart(stats: Dict[str, Any], output_dir: Path):
    """
    Create horizontal bar chart for most common entities.
    
    Args:
        stats: Statistics dictionary containing most common entities
        output_dir: Directory to save the chart
    """
    if not stats['most_common_entities']:
        return
        
    setup_japanese_fonts()
    
    plt.figure(figsize=(12, 8))
    words, counts = zip(*stats['most_common_entities'])
    
    bars = plt.barh(range(len(words)), counts)
    plt.yticks(range(len(words)), words)
    plt.title('最頻出固有表現 (Top 10)')
    plt.xlabel('出現回数')
    plt.gca().invert_yaxis()
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                str(count), ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'most_common_entities.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_document_entities_chart(stats: Dict[str, Any], output_dir: Path):
    """
    Create bar chart for entities per document.
    
    Args:
        stats: Statistics dictionary containing document stats
        output_dir: Directory to save the chart
    """
    if not stats['documents_stats']:
        return
        
    setup_japanese_fonts()
    
    plt.figure(figsize=(12, 6))
    filenames = [doc['filename'] for doc in stats['documents_stats']]
    entity_counts = [doc['entity_count'] for doc in stats['documents_stats']]
    
    bars = plt.bar(range(len(filenames)), entity_counts)
    plt.title('ドキュメント別固有表現数')
    plt.xlabel('ドキュメント')
    plt.ylabel('固有表現数')
    plt.xticks(range(len(filenames)), filenames, rotation=45, ha='right')
    
    # Add value labels
    for bar, count in zip(bars, entity_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'entities_per_document.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_all_visualizations(stats: Dict[str, Any], output_dir: str):
    """
    Create all visualization charts.
    
    Args:
        stats: Statistics dictionary
        output_dir: Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    create_entity_type_chart(stats, output_path)
    create_common_entities_chart(stats, output_path)
    create_document_entities_chart(stats, output_path)
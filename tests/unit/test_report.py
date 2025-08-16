"""
Unit tests for report.py module.
"""

import pytest
import pandas as pd
from collections import Counter
from datetime import datetime
from pathlib import Path
from japanese_ner.report import (
    calculate_statistics, 
    save_csv_report, 
    generate_markdown_report, 
    save_markdown_report
)


class TestCalculateStatistics:
    """Test cases for calculate_statistics function."""
    
    @pytest.fixture
    def sample_results(self):
        """Sample analysis results for testing."""
        return [
            {
                'filename': 'doc1.txt',
                'content': 'これは田中太郎のテストです。',
                'entities': [
                    {'word': '田中太郎', 'entity_type': 'PER', 'score': 0.99},
                    {'word': 'テスト', 'entity_type': 'PRD', 'score': 0.95}
                ],
                'entity_count': 2,
                'analysis_time': '2024-01-01T10:00:00'
            },
            {
                'filename': 'doc2.txt',
                'content': 'トヨタ自動車は東京にあります。',
                'entities': [
                    {'word': 'トヨタ自動車', 'entity_type': 'ORG', 'score': 0.98},
                    {'word': '東京', 'entity_type': 'LOC', 'score': 0.97}
                ],
                'entity_count': 2,
                'analysis_time': '2024-01-01T10:01:00'
            }
        ]
    
    def test_basic_statistics(self, sample_results):
        """Test basic statistics calculation."""
        stats = calculate_statistics(sample_results)
        
        assert stats['total_documents'] == 2
        assert stats['total_entities'] == 4
        assert stats['avg_entities_per_doc'] == 2.0
    
    def test_entity_type_counts(self, sample_results):
        """Test entity type counting."""
        stats = calculate_statistics(sample_results)
        
        assert isinstance(stats['entity_type_counts'], Counter)
        assert stats['entity_type_counts']['PER'] == 1
        assert stats['entity_type_counts']['ORG'] == 1
        assert stats['entity_type_counts']['LOC'] == 1
        assert stats['entity_type_counts']['PRD'] == 1
    
    def test_entity_word_counts(self, sample_results):
        """Test entity word counting."""
        stats = calculate_statistics(sample_results)
        
        assert isinstance(stats['entity_word_counts'], Counter)
        assert stats['entity_word_counts']['田中太郎'] == 1
        assert stats['entity_word_counts']['トヨタ自動車'] == 1
        assert stats['entity_word_counts']['東京'] == 1
        assert stats['entity_word_counts']['テスト'] == 1
    
    def test_most_common_entities(self, sample_results):
        """Test most common entities extraction."""
        # Add duplicate entity to test counting
        sample_results[1]['entities'].append(
            {'word': '東京', 'entity_type': 'LOC', 'score': 0.96}
        )
        sample_results[1]['entity_count'] = 3
        
        stats = calculate_statistics(sample_results)
        
        assert len(stats['most_common_entities']) <= 10
        # 東京 should be most common with count 2
        most_common = stats['most_common_entities'][0]
        assert most_common[0] == '東京'
        assert most_common[1] == 2
    
    def test_documents_stats(self, sample_results):
        """Test per-document statistics."""
        stats = calculate_statistics(sample_results)
        
        assert len(stats['documents_stats']) == 2
        
        doc1_stats = stats['documents_stats'][0]
        assert doc1_stats['filename'] == 'doc1.txt'
        assert doc1_stats['entity_count'] == 2
        assert doc1_stats['unique_entity_types'] == 2
        assert doc1_stats['text_length'] == len('これは田中太郎のテストです。')
    
    def test_entity_type_distribution(self, sample_results):
        """Test entity type percentage distribution."""
        stats = calculate_statistics(sample_results)
        
        # Each entity type appears once out of 4 total = 25%
        assert stats['entity_type_distribution']['PER'] == 25.0
        assert stats['entity_type_distribution']['ORG'] == 25.0
        assert stats['entity_type_distribution']['LOC'] == 25.0
        assert stats['entity_type_distribution']['PRD'] == 25.0
    
    def test_empty_results(self):
        """Test statistics calculation with empty results."""
        stats = calculate_statistics([])
        
        assert stats['total_documents'] == 0
        assert stats['total_entities'] == 0
        assert stats['avg_entities_per_doc'] == 0
        assert len(stats['entity_type_counts']) == 0
        assert len(stats['most_common_entities']) == 0
        assert len(stats['documents_stats']) == 0


class TestSaveCsvReport:
    """Test cases for save_csv_report function."""
    
    @pytest.fixture
    def sample_results(self):
        """Sample results for CSV testing."""
        return [
            {
                'filename': 'test.txt',
                'entities': [
                    {
                        'word': '田中太郎',
                        'entity_type': 'PER',
                        'score': 0.99,
                        'start': 0,
                        'end': 3
                    }
                ],
                'analysis_time': '2024-01-01T10:00:00'
            }
        ]
    
    def test_csv_creation(self, sample_results, temp_dir):
        """Test CSV file creation."""
        csv_path = temp_dir / "test.csv"
        entity_descriptions = {'PER': '人名'}
        
        save_csv_report(sample_results, str(csv_path), entity_descriptions)
        
        assert csv_path.exists()
        
        # Read and verify CSV content
        df = pd.read_csv(csv_path)
        assert len(df) == 1
        assert df.iloc[0]['word'] == '田中太郎'
        assert df.iloc[0]['entity_type'] == 'PER'
        assert df.iloc[0]['entity_description'] == '人名'
        assert df.iloc[0]['score'] == 0.99
    
    def test_csv_multiple_entities(self, temp_dir):
        """Test CSV with multiple entities."""
        results = [
            {
                'filename': 'test.txt',
                'entities': [
                    {'word': '田中', 'entity_type': 'PER', 'score': 0.99, 'start': 0, 'end': 2},
                    {'word': '東京', 'entity_type': 'LOC', 'score': 0.98, 'start': 3, 'end': 5}
                ],
                'analysis_time': '2024-01-01T10:00:00'
            }
        ]
        
        csv_path = temp_dir / "multi.csv"
        entity_descriptions = {'PER': '人名', 'LOC': '場所'}
        
        save_csv_report(results, str(csv_path), entity_descriptions)
        
        df = pd.read_csv(csv_path)
        assert len(df) == 2
        assert '田中' in df['word'].values
        assert '東京' in df['word'].values


class TestGenerateMarkdownReport:
    """Test cases for generate_markdown_report function."""
    
    @pytest.fixture
    def sample_stats(self):
        """Sample statistics for report testing."""
        return {
            'total_documents': 2,
            'total_entities': 4,
            'avg_entities_per_doc': 2.0,
            'entity_type_counts': Counter({'PER': 2, 'LOC': 1, 'ORG': 1}),
            'entity_type_distribution': {'PER': 50.0, 'LOC': 25.0, 'ORG': 25.0},
            'most_common_entities': [('田中太郎', 2), ('東京', 1)],
            'documents_stats': [
                {
                    'filename': 'doc1.txt',
                    'entity_count': 3,
                    'unique_entity_types': 2,
                    'text_length': 100
                }
            ]
        }
    
    def test_report_structure(self, sample_stats):
        """Test basic report structure."""
        model_name = "test-model"
        entity_descriptions = {'PER': '人名', 'LOC': '場所', 'ORG': '組織'}
        
        report = generate_markdown_report(sample_stats, model_name, entity_descriptions)
        
        assert "# 固有表現抽出 分析レポート" in report
        assert "## 分析概要" in report
        assert "## 固有表現タイプ別統計" in report
        assert "## 最頻出固有表現" in report
        assert "## ドキュメント別詳細" in report
        assert model_name in report
    
    def test_report_statistics_content(self, sample_stats):
        """Test report content accuracy."""
        model_name = "test-model"
        entity_descriptions = {'PER': '人名', 'LOC': '場所', 'ORG': '組織'}
        
        report = generate_markdown_report(sample_stats, model_name, entity_descriptions)
        
        assert "総ドキュメント数**: 2" in report
        assert "総固有表現数**: 4" in report
        assert "平均固有表現数**: 2.00" in report
        assert "田中太郎" in report
        assert "doc1.txt" in report
    
    def test_report_entity_table(self, sample_stats):
        """Test entity type table in report."""
        model_name = "test-model"
        entity_descriptions = {'PER': '人名', 'LOC': '場所', 'ORG': '組織'}
        
        report = generate_markdown_report(sample_stats, model_name, entity_descriptions)
        
        # Should contain table with entity types and percentages
        assert "| PER | 人名 | 2 | 50.0% |" in report
        assert "| LOC | 場所 | 1 | 25.0% |" in report
        assert "| ORG | 組織 | 1 | 25.0% |" in report


class TestSaveMarkdownReport:
    """Test cases for save_markdown_report function."""
    
    def test_report_file_creation(self, temp_dir):
        """Test markdown report file creation."""
        stats = {
            'total_documents': 1,
            'total_entities': 1,
            'avg_entities_per_doc': 1.0,
            'entity_type_counts': Counter({'PER': 1}),
            'entity_type_distribution': {'PER': 100.0},
            'most_common_entities': [('田中', 1)],
            'documents_stats': []
        }
        
        report_path = temp_dir / "report.md"
        model_name = "test-model"
        entity_descriptions = {'PER': '人名'}
        
        save_markdown_report(stats, str(report_path), model_name, entity_descriptions)
        
        assert report_path.exists()
        
        # Verify file content
        content = report_path.read_text(encoding='utf-8')
        assert "# 固有表現抽出 分析レポート" in content
        assert "test-model" in content
        assert "田中" in content
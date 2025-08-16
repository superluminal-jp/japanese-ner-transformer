"""
Unit tests for analyzer.py module.
"""

import pytest
from unittest.mock import Mock, patch
from japanese_ner.analyzer import NERAnalyzer


class TestNERAnalyzer:
    """Test cases for NERAnalyzer class."""
    
    def test_init_default_model(self):
        """Test analyzer initialization with default model."""
        analyzer = NERAnalyzer()
        assert analyzer.model_name == "tsmatz/xlm-roberta-ner-japanese"
        assert 'PER' in analyzer.entity_descriptions
        assert len(analyzer.entity_descriptions) == 8
    
    def test_init_custom_model(self):
        """Test analyzer initialization with custom model."""
        custom_model = "custom-model-name"
        with patch('japanese_ner.analyzer.AutoTokenizer'), \
             patch('japanese_ner.analyzer.AutoModelForTokenClassification'), \
             patch('japanese_ner.analyzer.pipeline'):
            analyzer = NERAnalyzer(custom_model)
            assert analyzer.model_name == custom_model
    
    def test_entity_descriptions_complete(self):
        """Test that all expected entity types are in descriptions."""
        analyzer = NERAnalyzer()
        expected_types = ['PER', 'ORG', 'ORG-P', 'ORG-O', 'LOC', 'INS', 'PRD', 'EVT']
        
        for entity_type in expected_types:
            assert entity_type in analyzer.entity_descriptions
            assert isinstance(analyzer.entity_descriptions[entity_type], str)
            assert len(analyzer.entity_descriptions[entity_type]) > 0
    
    @patch('japanese_ner.analyzer.pipeline')
    @patch('japanese_ner.analyzer.AutoModelForTokenClassification')
    @patch('japanese_ner.analyzer.AutoTokenizer')
    def test_analyze_text_basic(self, mock_tokenizer, mock_model, mock_pipeline):
        """Test basic text analysis functionality."""
        # Mock the pipeline
        mock_ner = Mock()
        mock_ner.return_value = [
            {
                'word': '田中太郎',
                'entity_group': 'PER',
                'score': 0.9999,
                'start': 0,
                'end': 3
            }
        ]
        mock_pipeline.return_value = mock_ner
        
        analyzer = NERAnalyzer()
        result = analyzer.analyze("田中太郎は東京にいます。")
        
        assert len(result) == 1
        assert result[0]['word'] == '田中太郎'
        assert result[0]['entity_type'] == 'PER'
        assert result[0]['score'] == 0.9999
        assert result[0]['description'] == '人名'
    
    @patch('japanese_ner.analyzer.pipeline')
    @patch('japanese_ner.analyzer.AutoModelForTokenClassification')
    @patch('japanese_ner.analyzer.AutoTokenizer')
    def test_analyze_empty_text(self, mock_tokenizer, mock_model, mock_pipeline):
        """Test analysis of empty text."""
        mock_ner = Mock()
        mock_ner.return_value = []
        mock_pipeline.return_value = mock_ner
        
        analyzer = NERAnalyzer()
        result = analyzer.analyze("")
        
        assert result == []
    
    @patch('japanese_ner.analyzer.pipeline')
    @patch('japanese_ner.analyzer.AutoModelForTokenClassification')
    @patch('japanese_ner.analyzer.AutoTokenizer')
    def test_analyze_multiple_entities(self, mock_tokenizer, mock_model, mock_pipeline):
        """Test analysis with multiple entities."""
        mock_ner = Mock()
        mock_ner.return_value = [
            {
                'word': '田中太郎',
                'entity_group': 'PER',
                'score': 0.9999,
                'start': 0,
                'end': 3
            },
            {
                'word': '東京',
                'entity_group': 'LOC',
                'score': 0.9998,
                'start': 4,
                'end': 6
            },
            {
                'word': 'トヨタ',
                'entity_group': 'ORG',
                'score': 0.9997,
                'start': 7,
                'end': 10
            }
        ]
        mock_pipeline.return_value = mock_ner
        
        analyzer = NERAnalyzer()
        result = analyzer.analyze("田中太郎は東京のトヨタで働いています。")
        
        assert len(result) == 3
        
        # Check each entity
        assert result[0]['entity_type'] == 'PER'
        assert result[0]['description'] == '人名'
        
        assert result[1]['entity_type'] == 'LOC'
        assert result[1]['description'] == '場所・地名'
        
        assert result[2]['entity_type'] == 'ORG'
        assert result[2]['description'] == '一般企業・組織'
    
    @patch('japanese_ner.analyzer.pipeline')
    @patch('japanese_ner.analyzer.AutoModelForTokenClassification')
    @patch('japanese_ner.analyzer.AutoTokenizer')
    def test_analyze_unknown_entity_type(self, mock_tokenizer, mock_model, mock_pipeline):
        """Test handling of unknown entity types."""
        mock_ner = Mock()
        mock_ner.return_value = [
            {
                'word': 'テスト',
                'entity_group': 'UNKNOWN',
                'score': 0.5000,
                'start': 0,
                'end': 3
            }
        ]
        mock_pipeline.return_value = mock_ner
        
        analyzer = NERAnalyzer()
        result = analyzer.analyze("テスト")
        
        assert len(result) == 1
        assert result[0]['entity_type'] == 'UNKNOWN'
        assert result[0]['description'] == '不明'
    
    def test_get_entity_types(self):
        """Test get_entity_types method."""
        analyzer = NERAnalyzer()
        entity_types = analyzer.get_entity_types()
        
        # Should return a copy, not the original
        assert entity_types == analyzer.entity_descriptions
        assert entity_types is not analyzer.entity_descriptions
        
        # Modifying returned dict shouldn't affect original
        entity_types['TEST'] = 'テスト'
        assert 'TEST' not in analyzer.entity_descriptions
    
    @patch('japanese_ner.analyzer.pipeline')
    @patch('japanese_ner.analyzer.AutoModelForTokenClassification')
    @patch('japanese_ner.analyzer.AutoTokenizer')
    def test_analyze_missing_start_end(self, mock_tokenizer, mock_model, mock_pipeline):
        """Test handling of entities without start/end positions."""
        mock_ner = Mock()
        mock_ner.return_value = [
            {
                'word': '田中太郎',
                'entity_group': 'PER',
                'score': 0.9999
                # Missing 'start' and 'end' keys
            }
        ]
        mock_pipeline.return_value = mock_ner
        
        analyzer = NERAnalyzer()
        result = analyzer.analyze("田中太郎")
        
        assert len(result) == 1
        assert result[0]['start'] == 0  # Default value
        assert result[0]['end'] == 0    # Default value
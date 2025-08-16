"""
Integration tests for batch_analyzer.py module.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
from japanese_ner.batch_analyzer import BatchNERAnalyzer


class TestBatchNERAnalyzer:
    """Integration test cases for BatchNERAnalyzer class."""
    
    def test_init_with_default_model(self):
        """Test initialization with default model."""
        with patch('japanese_ner.batch_analyzer.NERAnalyzer') as mock_analyzer:
            batch_analyzer = BatchNERAnalyzer()
            
            mock_analyzer.assert_called_once_with("tsmatz/xlm-roberta-ner-japanese")
            assert batch_analyzer.model_name == "tsmatz/xlm-roberta-ner-japanese"
    
    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        custom_model = "custom-model-name"
        
        with patch('japanese_ner.batch_analyzer.NERAnalyzer') as mock_analyzer:
            batch_analyzer = BatchNERAnalyzer(custom_model)
            
            mock_analyzer.assert_called_once_with(custom_model)
            assert batch_analyzer.model_name == custom_model


class TestAnalyzeDocuments:
    """Test cases for analyze_documents method."""
    
    @pytest.fixture
    def mock_analyzer(self):
        """Mock NER analyzer."""
        analyzer = Mock()
        analyzer.analyze.return_value = [
            {'word': 'テスト', 'entity_type': 'PRD', 'score': 0.95}
        ]
        return analyzer
    
    @pytest.fixture
    def batch_analyzer_with_mock(self, mock_analyzer):
        """Batch analyzer with mocked components."""
        with patch('japanese_ner.batch_analyzer.NERAnalyzer') as mock_ner_class:
            mock_ner_class.return_value = mock_analyzer
            batch_analyzer = BatchNERAnalyzer()
            batch_analyzer.analyzer = mock_analyzer
            return batch_analyzer
    
    @patch('japanese_ner.batch_analyzer.read_documents')
    def test_analyze_documents_basic(self, mock_read_docs, batch_analyzer_with_mock, mock_analyzer):
        """Test basic document analysis."""
        # Mock document reading
        mock_read_docs.return_value = [
            {'filename': 'test.txt', 'content': 'これはテストです。'}
        ]
        
        results = batch_analyzer_with_mock.analyze_documents('/fake/path')
        
        assert len(results) == 1
        assert results[0]['filename'] == 'test.txt'
        assert results[0]['content'] == 'これはテストです。'
        assert results[0]['entity_count'] == 1
        assert 'analysis_time' in results[0]
        assert len(results[0]['entities']) == 1
    
    @patch('japanese_ner.batch_analyzer.read_documents')
    def test_analyze_multiple_documents(self, mock_read_docs, batch_analyzer_with_mock, mock_analyzer):
        """Test analysis of multiple documents."""
        # Mock multiple documents
        mock_read_docs.return_value = [
            {'filename': 'doc1.txt', 'content': '文書1'},
            {'filename': 'doc2.txt', 'content': '文書2'},
            {'filename': 'doc3.txt', 'content': '文書3'}
        ]
        
        results = batch_analyzer_with_mock.analyze_documents('/fake/path')
        
        assert len(results) == 3
        assert results[0]['filename'] == 'doc1.txt'
        assert results[1]['filename'] == 'doc2.txt'
        assert results[2]['filename'] == 'doc3.txt'
        
        # Each document should be analyzed
        assert mock_analyzer.analyze.call_count == 3
    
    @patch('japanese_ner.batch_analyzer.read_documents')
    def test_analyze_empty_documents(self, mock_read_docs, batch_analyzer_with_mock):
        """Test analysis with no documents."""
        mock_read_docs.return_value = []
        
        results = batch_analyzer_with_mock.analyze_documents('/fake/path')
        
        assert results == []
    
    @patch('japanese_ner.batch_analyzer.read_documents')
    def test_analyze_documents_with_no_entities(self, mock_read_docs, batch_analyzer_with_mock, mock_analyzer):
        """Test analysis where no entities are found."""
        mock_read_docs.return_value = [
            {'filename': 'empty.txt', 'content': 'これは普通の文章です。'}
        ]
        mock_analyzer.analyze.return_value = []  # No entities found
        
        results = batch_analyzer_with_mock.analyze_documents('/fake/path')
        
        assert len(results) == 1
        assert results[0]['entity_count'] == 0
        assert results[0]['entities'] == []


class TestGenerateFullReport:
    """Test cases for generate_full_report method."""
    
    @pytest.fixture
    def mock_batch_analyzer(self):
        """Mock batch analyzer with all dependencies."""
        with patch('japanese_ner.batch_analyzer.NERAnalyzer'), \
             patch('japanese_ner.batch_analyzer.ensure_output_directory') as mock_ensure_dir, \
             patch('japanese_ner.batch_analyzer.calculate_statistics') as mock_calc_stats, \
             patch('japanese_ner.batch_analyzer.save_csv_report') as mock_save_csv, \
             patch('japanese_ner.batch_analyzer.create_all_visualizations') as mock_viz, \
             patch('japanese_ner.batch_analyzer.save_markdown_report') as mock_save_md:
            
            batch_analyzer = BatchNERAnalyzer()
            
            # Mock the analyze_documents method
            batch_analyzer.analyze_documents = Mock(return_value=[
                {
                    'filename': 'test.txt',
                    'content': 'テスト内容',
                    'entities': [{'word': 'テスト', 'entity_type': 'PRD', 'score': 0.95}],
                    'entity_count': 1,
                    'analysis_time': '2024-01-01T10:00:00'
                }
            ])
            
            # Mock return values
            mock_ensure_dir.return_value = Path('/fake/output')
            mock_calc_stats.return_value = {'total_documents': 1, 'total_entities': 1}
            
            # Return all mocks for verification
            return (batch_analyzer, mock_ensure_dir, mock_calc_stats, 
                   mock_save_csv, mock_viz, mock_save_md)
    
    def test_full_report_workflow(self, mock_batch_analyzer):
        """Test complete report generation workflow."""
        (batch_analyzer, mock_ensure_dir, mock_calc_stats, 
         mock_save_csv, mock_viz, mock_save_md) = mock_batch_analyzer
        
        input_path = '/fake/input'
        output_dir = '/fake/output'
        
        batch_analyzer.generate_full_report(input_path, output_dir)
        
        # Verify all steps are called in order
        mock_ensure_dir.assert_called_once_with(output_dir)
        batch_analyzer.analyze_documents.assert_called_once_with(input_path)
        mock_calc_stats.assert_called_once()
        mock_save_csv.assert_called_once()
        mock_viz.assert_called_once()
        mock_save_md.assert_called_once()
    
    def test_output_directory_creation(self, mock_batch_analyzer):
        """Test output directory is created."""
        (batch_analyzer, mock_ensure_dir, *_) = mock_batch_analyzer
        
        output_dir = '/test/output'
        batch_analyzer.generate_full_report('/fake/input', output_dir)
        
        mock_ensure_dir.assert_called_once_with(output_dir)
    
    def test_csv_report_parameters(self, mock_batch_analyzer):
        """Test CSV report is called with correct parameters."""
        (batch_analyzer, _, _, mock_save_csv, _, _) = mock_batch_analyzer
        
        batch_analyzer.generate_full_report('/fake/input', '/fake/output')
        
        # Should be called with results, path, and entity descriptions
        call_args = mock_save_csv.call_args
        assert len(call_args[0]) == 3  # results, path, descriptions
        assert 'ner_results.csv' in str(call_args[0][1])
    
    def test_visualization_parameters(self, mock_batch_analyzer):
        """Test visualization is called with correct parameters."""
        (batch_analyzer, _, _, _, mock_viz, _) = mock_batch_analyzer
        
        batch_analyzer.generate_full_report('/fake/input', '/fake/output')
        
        # Should be called with stats and output directory
        call_args = mock_viz.call_args
        assert len(call_args[0]) == 2  # stats, output_dir
    
    def test_markdown_report_parameters(self, mock_batch_analyzer):
        """Test markdown report is called with correct parameters."""
        (batch_analyzer, _, _, _, _, mock_save_md) = mock_batch_analyzer
        
        batch_analyzer.generate_full_report('/fake/input', '/fake/output')
        
        # Should be called with stats, path, model_name, descriptions
        call_args = mock_save_md.call_args
        assert len(call_args[0]) == 4  # stats, path, model_name, descriptions
        assert 'analysis_report.md' in str(call_args[0][1])


class TestBatchAnalyzerIntegration:
    """Integration tests with real file operations."""
    
    def test_real_file_integration(self, sample_documents, temp_dir):
        """Test with real files (using mocked NER analysis)."""
        output_dir = temp_dir / "output"
        
        with patch('japanese_ner.batch_analyzer.NERAnalyzer') as mock_ner_class:
            # Mock the analyzer to return predictable results
            mock_analyzer = Mock()
            mock_analyzer.analyze.return_value = [
                {'word': 'テスト', 'entity_type': 'PRD', 'score': 0.95, 'start': 0, 'end': 3}
            ]
            mock_analyzer.entity_descriptions = {'PRD': '製品'}
            mock_ner_class.return_value = mock_analyzer
            
            # Create batch analyzer
            batch_analyzer = BatchNERAnalyzer()
            
            # Analyze the sample documents
            results = batch_analyzer.analyze_documents(str(sample_documents))
            
            # Should find at least the txt files
            assert len(results) >= 2
            
            # Check result structure
            for result in results:
                assert 'filename' in result
                assert 'content' in result
                assert 'entities' in result
                assert 'entity_count' in result
                assert 'analysis_time' in result
    
    def test_error_handling_invalid_path(self):
        """Test error handling for invalid input path."""
        with patch('japanese_ner.batch_analyzer.NERAnalyzer'):
            batch_analyzer = BatchNERAnalyzer()
            
            with pytest.raises(ValueError):
                batch_analyzer.analyze_documents('/nonexistent/path')
    
    @patch('japanese_ner.batch_analyzer.create_all_visualizations')
    @patch('japanese_ner.batch_analyzer.save_markdown_report')
    @patch('japanese_ner.batch_analyzer.save_csv_report')
    @patch('japanese_ner.batch_analyzer.calculate_statistics')
    def test_exception_handling_in_report_generation(self, mock_calc_stats, mock_save_csv, 
                                                   mock_save_md, mock_viz, sample_documents):
        """Test exception handling during report generation."""
        # Make one of the report functions raise an exception
        mock_viz.side_effect = Exception("Visualization error")
        
        with patch('japanese_ner.batch_analyzer.NERAnalyzer') as mock_ner_class:
            mock_analyzer = Mock()
            mock_analyzer.analyze.return_value = []
            mock_analyzer.entity_descriptions = {}
            mock_ner_class.return_value = mock_analyzer
            
            batch_analyzer = BatchNERAnalyzer()
            
            # Should raise the exception
            with pytest.raises(Exception, match="Visualization error"):
                batch_analyzer.generate_full_report(str(sample_documents), '/fake/output')
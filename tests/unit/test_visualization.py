"""
Unit tests for visualization.py module.
"""

import pytest
from unittest.mock import patch, Mock
from pathlib import Path
from collections import Counter
from japanese_ner.visualization import (
    setup_japanese_fonts,
    create_entity_type_chart,
    create_common_entities_chart,
    create_document_entities_chart,
    create_all_visualizations
)


class TestSetupJapaneseFonts:
    """Test cases for setup_japanese_fonts function."""
    
    @patch('japanese_ner.visualization.plt')
    def test_font_setup(self, mock_plt):
        """Test Japanese font configuration."""
        setup_japanese_fonts()
        
        expected_fonts = ['DejaVu Sans', 'Arial Unicode MS', 'Hiragino Sans']
        mock_plt.rcParams.__setitem__.assert_called_with('font.family', expected_fonts)


class TestCreateEntityTypeChart:
    """Test cases for create_entity_type_chart function."""
    
    @pytest.fixture
    def sample_stats(self):
        """Sample statistics for testing."""
        return {
            'entity_type_counts': Counter({'PER': 10, 'ORG': 8, 'LOC': 5})
        }
    
    @pytest.fixture
    def empty_stats(self):
        """Empty statistics for testing."""
        return {
            'entity_type_counts': Counter()
        }
    
    @patch('japanese_ner.visualization.plt')
    def test_chart_creation(self, mock_plt, sample_stats, temp_dir):
        """Test entity type chart creation."""
        create_entity_type_chart(sample_stats, temp_dir)
        
        # Verify matplotlib calls
        mock_plt.figure.assert_called_once_with(figsize=(10, 6))
        mock_plt.bar.assert_called_once()
        mock_plt.title.assert_called_with('固有表現タイプ別出現回数')
        mock_plt.xlabel.assert_called_with('固有表現タイプ')
        mock_plt.ylabel.assert_called_with('出現回数')
        mock_plt.xticks.assert_called_with(rotation=45)
        mock_plt.tight_layout.assert_called_once()
        mock_plt.close.assert_called_once()
    
    @patch('japanese_ner.visualization.plt')
    def test_chart_save_path(self, mock_plt, sample_stats, temp_dir):
        """Test chart save path."""
        create_entity_type_chart(sample_stats, temp_dir)
        
        expected_path = temp_dir / 'entity_type_distribution.png'
        mock_plt.savefig.assert_called_with(expected_path, dpi=300, bbox_inches='tight')
    
    @patch('japanese_ner.visualization.plt')
    def test_empty_stats_no_chart(self, mock_plt, empty_stats, temp_dir):
        """Test no chart creation with empty statistics."""
        create_entity_type_chart(empty_stats, temp_dir)
        
        # Should not create chart if no data
        mock_plt.figure.assert_not_called()
        mock_plt.bar.assert_not_called()
    
    @patch('japanese_ner.visualization.plt')
    def test_value_labels(self, mock_plt, sample_stats, temp_dir):
        """Test value labels on bars."""
        # Mock bar objects
        mock_bars = [Mock(), Mock(), Mock()]
        for i, bar in enumerate(mock_bars):
            bar.get_x.return_value = i
            bar.get_width.return_value = 1
            bar.get_height.return_value = [10, 8, 5][i]
        
        mock_plt.bar.return_value = mock_bars
        
        create_entity_type_chart(sample_stats, temp_dir)
        
        # Should call text for each bar
        assert mock_plt.text.call_count == 3


class TestCreateCommonEntitiesChart:
    """Test cases for create_common_entities_chart function."""
    
    @pytest.fixture
    def sample_stats(self):
        """Sample statistics with common entities."""
        return {
            'most_common_entities': [('田中太郎', 5), ('東京', 3), ('トヨタ', 2)]
        }
    
    @pytest.fixture
    def empty_stats(self):
        """Empty statistics."""
        return {
            'most_common_entities': []
        }
    
    @patch('japanese_ner.visualization.plt')
    def test_horizontal_chart_creation(self, mock_plt, sample_stats, temp_dir):
        """Test horizontal bar chart creation."""
        create_common_entities_chart(sample_stats, temp_dir)
        
        mock_plt.figure.assert_called_once_with(figsize=(12, 8))
        mock_plt.barh.assert_called_once()
        mock_plt.title.assert_called_with('最頻出固有表現 (Top 10)')
        mock_plt.xlabel.assert_called_with('出現回数')
        mock_plt.gca().invert_yaxis.assert_called_once()
    
    @patch('japanese_ner.visualization.plt')
    def test_chart_save_path(self, mock_plt, sample_stats, temp_dir):
        """Test chart save path."""
        create_common_entities_chart(sample_stats, temp_dir)
        
        expected_path = temp_dir / 'most_common_entities.png'
        mock_plt.savefig.assert_called_with(expected_path, dpi=300, bbox_inches='tight')
    
    @patch('japanese_ner.visualization.plt')
    def test_empty_stats_no_chart(self, mock_plt, empty_stats, temp_dir):
        """Test no chart creation with empty statistics."""
        create_common_entities_chart(empty_stats, temp_dir)
        
        mock_plt.figure.assert_not_called()


class TestCreateDocumentEntitiesChart:
    """Test cases for create_document_entities_chart function."""
    
    @pytest.fixture
    def sample_stats(self):
        """Sample document statistics."""
        return {
            'documents_stats': [
                {'filename': 'doc1.txt', 'entity_count': 10},
                {'filename': 'doc2.txt', 'entity_count': 15},
                {'filename': 'doc3.txt', 'entity_count': 7}
            ]
        }
    
    @pytest.fixture
    def empty_stats(self):
        """Empty document statistics."""
        return {
            'documents_stats': []
        }
    
    @patch('japanese_ner.visualization.plt')
    def test_document_chart_creation(self, mock_plt, sample_stats, temp_dir):
        """Test document entities chart creation."""
        create_document_entities_chart(sample_stats, temp_dir)
        
        mock_plt.figure.assert_called_once_with(figsize=(12, 6))
        mock_plt.bar.assert_called_once()
        mock_plt.title.assert_called_with('ドキュメント別固有表現数')
        mock_plt.xlabel.assert_called_with('ドキュメント')
        mock_plt.ylabel.assert_called_with('固有表現数')
        mock_plt.xticks.assert_called_once()
    
    @patch('japanese_ner.visualization.plt')
    def test_rotated_labels(self, mock_plt, sample_stats, temp_dir):
        """Test rotated x-axis labels."""
        create_document_entities_chart(sample_stats, temp_dir)
        
        # Check that xticks is called with rotation
        args, kwargs = mock_plt.xticks.call_args
        assert kwargs.get('rotation') == 45
        assert kwargs.get('ha') == 'right'
    
    @patch('japanese_ner.visualization.plt')
    def test_chart_save_path(self, mock_plt, sample_stats, temp_dir):
        """Test chart save path."""
        create_document_entities_chart(sample_stats, temp_dir)
        
        expected_path = temp_dir / 'entities_per_document.png'
        mock_plt.savefig.assert_called_with(expected_path, dpi=300, bbox_inches='tight')
    
    @patch('japanese_ner.visualization.plt')
    def test_empty_stats_no_chart(self, mock_plt, empty_stats, temp_dir):
        """Test no chart creation with empty statistics."""
        create_document_entities_chart(empty_stats, temp_dir)
        
        mock_plt.figure.assert_not_called()


class TestCreateAllVisualizations:
    """Test cases for create_all_visualizations function."""
    
    @pytest.fixture
    def complete_stats(self):
        """Complete statistics for testing."""
        return {
            'entity_type_counts': Counter({'PER': 5, 'ORG': 3}),
            'most_common_entities': [('田中', 3), ('東京', 2)],
            'documents_stats': [
                {'filename': 'doc1.txt', 'entity_count': 5},
                {'filename': 'doc2.txt', 'entity_count': 3}
            ]
        }
    
    @patch('japanese_ner.visualization.create_document_entities_chart')
    @patch('japanese_ner.visualization.create_common_entities_chart')
    @patch('japanese_ner.visualization.create_entity_type_chart')
    def test_all_charts_created(self, mock_entity_type, mock_common, mock_document, 
                               complete_stats, temp_dir):
        """Test that all chart functions are called."""
        create_all_visualizations(complete_stats, str(temp_dir))
        
        # All chart creation functions should be called
        mock_entity_type.assert_called_once()
        mock_common.assert_called_once()
        mock_document.assert_called_once()
    
    @patch('japanese_ner.visualization.create_document_entities_chart')
    @patch('japanese_ner.visualization.create_common_entities_chart')
    @patch('japanese_ner.visualization.create_entity_type_chart')
    def test_output_directory_creation(self, mock_entity_type, mock_common, mock_document,
                                     complete_stats, temp_dir):
        """Test output directory creation."""
        new_output_dir = temp_dir / "visualizations"
        
        create_all_visualizations(complete_stats, str(new_output_dir))
        
        # Directory should be created
        assert new_output_dir.exists()
        assert new_output_dir.is_dir()
    
    def test_integration_with_real_stats(self, temp_dir):
        """Integration test with real statistics."""
        stats = {
            'entity_type_counts': Counter({'PER': 2, 'LOC': 1}),
            'most_common_entities': [('田中', 1), ('東京', 1)],
            'documents_stats': [
                {'filename': 'test.txt', 'entity_count': 3}
            ]
        }
        
        # This should not raise any exceptions
        with patch('japanese_ner.visualization.plt'):
            create_all_visualizations(stats, str(temp_dir))


class TestVisualizationIntegration:
    """Integration tests for visualization module."""
    
    @patch('japanese_ner.visualization.plt')
    def test_font_setup_called_in_charts(self, mock_plt, temp_dir):
        """Test that font setup is called in chart creation."""
        stats = {
            'entity_type_counts': Counter({'PER': 1}),
            'most_common_entities': [('田中', 1)],
            'documents_stats': [{'filename': 'test.txt', 'entity_count': 1}]
        }
        
        create_all_visualizations(stats, str(temp_dir))
        
        # Font setup should be called in each chart function
        expected_fonts = ['DejaVu Sans', 'Arial Unicode MS', 'Hiragino Sans']
        font_calls = [call for call in mock_plt.rcParams.__setitem__.call_args_list 
                     if call[0][0] == 'font.family']
        
        # Should be called at least once (possibly multiple times)
        assert len(font_calls) >= 1
        assert font_calls[0][0][1] == expected_fonts
"""
End-to-end tests for main.py module.
"""

import pytest
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, Mock


class TestMainScript:
    """End-to-end tests for the main script."""
    
    @pytest.fixture
    def main_script_path(self):
        """Path to the main.py script."""
        return Path(__file__).parent.parent.parent / "main.py"
    
    def test_main_script_exists(self, main_script_path):
        """Test that main.py exists and is executable."""
        assert main_script_path.exists()
        assert main_script_path.is_file()


class TestSimpleNERDemo:
    """Test cases for simple_ner_demo function."""
    
    @patch('main.NERAnalyzer')
    def test_demo_function_execution(self, mock_ner_class):
        """Test demo function runs without errors."""
        # Mock the analyzer
        mock_analyzer = Mock()
        mock_analyzer.analyze.return_value = [
            {
                'word': 'AI技術カンファレンス',
                'entity_type': 'EVT',
                'score': 0.9999,
                'start': 0,
                'end': 9
            },
            {
                'word': '東京国際フォーラム',
                'entity_type': 'INS',
                'score': 0.9998,
                'start': 10,
                'end': 18
            }
        ]
        mock_ner_class.return_value = mock_analyzer
        
        # Import and run the demo function
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from main import simple_ner_demo
        
        # Should run without exceptions
        simple_ner_demo()
        
        # Verify analyzer was called
        mock_analyzer.analyze.assert_called_once()
    
    @patch('main.NERAnalyzer')
    def test_demo_with_empty_results(self, mock_ner_class):
        """Test demo function with no entities found."""
        mock_analyzer = Mock()
        mock_analyzer.analyze.return_value = []
        mock_ner_class.return_value = mock_analyzer
        
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from main import simple_ner_demo
        
        # Should handle empty results gracefully
        simple_ner_demo()
        
        mock_analyzer.analyze.assert_called_once()


class TestBatchNERAnalysis:
    """Test cases for batch_ner_analysis function."""
    
    @patch('main.BatchNERAnalyzer')
    def test_batch_analysis_function(self, mock_batch_class):
        """Test batch analysis function execution."""
        mock_analyzer = Mock()
        mock_batch_class.return_value = mock_analyzer
        
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from main import batch_ner_analysis
        
        input_path = "/fake/input"
        output_dir = "/fake/output"
        model_name = "test-model"
        
        batch_ner_analysis(input_path, output_dir, model_name)
        
        # Verify correct initialization and method call
        mock_batch_class.assert_called_once_with(model_name)
        mock_analyzer.generate_full_report.assert_called_once_with(input_path, output_dir)


class TestMainFunction:
    """Test cases for main function and argument parsing."""
    
    @patch('main.simple_ner_demo')
    @patch('main.batch_ner_analysis')
    def test_main_demo_mode_no_args(self, mock_batch, mock_demo):
        """Test main function in demo mode with no arguments."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from main import main
        
        # Mock sys.argv to simulate no arguments
        with patch('sys.argv', ['main.py']):
            main()
        
        # Should call demo function
        mock_demo.assert_called_once()
        mock_batch.assert_not_called()
    
    @patch('main.simple_ner_demo')
    @patch('main.batch_ner_analysis')
    def test_main_explicit_demo_flag(self, mock_batch, mock_demo):
        """Test main function with explicit --demo flag."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from main import main
        
        with patch('sys.argv', ['main.py', '--demo']):
            main()
        
        mock_demo.assert_called_once()
        mock_batch.assert_not_called()
    
    @patch('main.simple_ner_demo')
    @patch('main.batch_ner_analysis')
    def test_main_batch_mode(self, mock_batch, mock_demo):
        """Test main function in batch mode."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from main import main
        
        with patch('sys.argv', ['main.py', '/fake/input']):
            main()
        
        mock_batch.assert_called_once_with('/fake/input', 'output', 'tsmatz/xlm-roberta-ner-japanese')
        mock_demo.assert_not_called()
    
    @patch('main.simple_ner_demo')
    @patch('main.batch_ner_analysis')
    def test_main_batch_mode_with_options(self, mock_batch, mock_demo):
        """Test main function in batch mode with custom options."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from main import main
        
        with patch('sys.argv', ['main.py', '/fake/input', '-o', 'custom_output', '-m', 'custom-model']):
            main()
        
        mock_batch.assert_called_once_with('/fake/input', 'custom_output', 'custom-model')
        mock_demo.assert_not_called()


class TestCommandLineInterface:
    """Test CLI interface through subprocess calls."""
    
    @pytest.fixture
    def main_script(self):
        """Path to main script."""
        return str(Path(__file__).parent.parent.parent / "main.py")
    
    def test_help_option(self, main_script):
        """Test --help option works."""
        result = subprocess.run(
            [sys.executable, main_script, '--help'],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert '日本語固有表現抽出ツール' in result.stdout
        assert '--demo' in result.stdout
        assert '--output' in result.stdout
        assert '--model' in result.stdout
    
    @patch('main.simple_ner_demo')
    def test_demo_mode_cli(self, mock_demo, main_script):
        """Test demo mode through CLI."""
        # This test would require mocking at the subprocess level
        # which is complex. Instead, we test the import and function call.
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        
        # Test that the script can be imported without errors
        import main
        assert hasattr(main, 'simple_ner_demo')
        assert hasattr(main, 'batch_ner_analysis')
        assert hasattr(main, 'main')


class TestScriptIntegration:
    """Integration tests for the complete script."""
    
    def test_import_and_basic_structure(self):
        """Test that script imports and has required functions."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        
        import main
        
        # Check required functions exist
        assert callable(getattr(main, 'simple_ner_demo', None))
        assert callable(getattr(main, 'batch_ner_analysis', None))
        assert callable(getattr(main, 'main', None))
    
    @patch('main.NERAnalyzer')
    @patch('main.BatchNERAnalyzer')
    def test_module_dependencies(self, mock_batch_class, mock_ner_class):
        """Test that all required modules can be imported."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        
        # This should not raise ImportError
        import main
        
        # The modules should be importable
        assert main.NERAnalyzer == mock_ner_class
        assert main.BatchNERAnalyzer == mock_batch_class
    
    def test_sample_text_content(self):
        """Test that sample text contains expected entities."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        
        # Read the source to check sample text
        main_file = Path(__file__).parent.parent.parent / "main.py"
        content = main_file.read_text(encoding='utf-8')
        
        # Sample text should contain various entity types for good evaluation
        assert '2024年11月15日' in content  # Date/time
        assert '東京国際フォーラム' in content  # Institution
        assert 'OpenAI社' in content  # Organization
        assert 'サム・アルトマン氏' in content  # Person
        assert 'GPT-5モデル' in content  # Product
        assert 'AI技術カンファレンス' in content  # Event
    
    @patch('builtins.print')
    @patch('main.NERAnalyzer')
    def test_demo_output_format(self, mock_ner_class, mock_print):
        """Test that demo produces expected output format."""
        mock_analyzer = Mock()
        mock_analyzer.analyze.return_value = [
            {'word': 'テスト', 'entity_type': 'PRD', 'score': 0.9999}
        ]
        mock_ner_class.return_value = mock_analyzer
        
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from main import simple_ner_demo
        
        simple_ner_demo()
        
        # Check that print was called with expected format
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        
        # Should have header, separator, entity output, separator, summary
        assert any("Running simple NER demo" in call for call in print_calls)
        assert any("=" * 60 in call for call in print_calls)
        assert any("Found" in call and "entities in total" in call for call in print_calls)


class TestErrorHandling:
    """Test error handling in main script."""
    
    @patch('main.batch_ner_analysis')
    def test_batch_analysis_error_propagation(self, mock_batch):
        """Test that errors in batch analysis are properly handled."""
        mock_batch.side_effect = Exception("Test error")
        
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from main import main
        
        with patch('sys.argv', ['main.py', '/fake/input']):
            with pytest.raises(Exception, match="Test error"):
                main()
    
    @patch('main.simple_ner_demo')
    def test_demo_error_propagation(self, mock_demo):
        """Test that errors in demo are properly handled."""
        mock_demo.side_effect = Exception("Demo error")
        
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from main import main
        
        with patch('sys.argv', ['main.py']):
            with pytest.raises(Exception, match="Demo error"):
                main()
"""
Unit tests for utils.py module.
"""

import pytest
import json
import tempfile
from pathlib import Path
from japanese_ner.utils import read_documents, ensure_output_directory, _read_single_file, _read_directory


class TestReadDocuments:
    """Test cases for read_documents function."""
    
    def test_read_single_txt_file(self, temp_dir):
        """Test reading a single text file."""
        # Create test file
        test_file = temp_dir / "test.txt"
        test_content = "これはテストファイルです。"
        test_file.write_text(test_content, encoding='utf-8')
        
        # Test reading
        documents = read_documents(str(test_file))
        
        assert len(documents) == 1
        assert documents[0]['filename'] == "test.txt"
        assert documents[0]['content'] == test_content
    
    def test_read_single_json_file_string(self, temp_dir):
        """Test reading a JSON file with string content."""
        # Create test JSON file
        test_file = temp_dir / "test.json"
        test_data = "これはJSONテストです。"
        test_file.write_text(json.dumps(test_data, ensure_ascii=False), encoding='utf-8')
        
        # Test reading
        documents = read_documents(str(test_file))
        
        assert len(documents) == 1
        assert documents[0]['filename'] == "test.json"
        assert documents[0]['content'] == test_data
    
    def test_read_single_json_file_list(self, temp_dir):
        """Test reading a JSON file with list content."""
        # Create test JSON file
        test_file = temp_dir / "test.json"
        test_data = ["テスト1", "テスト2", "テスト3"]
        test_file.write_text(json.dumps(test_data, ensure_ascii=False), encoding='utf-8')
        
        # Test reading
        documents = read_documents(str(test_file))
        
        assert len(documents) == 3
        assert documents[0]['filename'] == "test_1"
        assert documents[0]['content'] == "テスト1"
        assert documents[1]['filename'] == "test_2"
        assert documents[1]['content'] == "テスト2"
        assert documents[2]['filename'] == "test_3"
        assert documents[2]['content'] == "テスト3"
    
    def test_read_directory(self, temp_dir):
        """Test reading all text files from a directory."""
        # Create test files
        file1 = temp_dir / "doc1.txt"
        file1.write_text("ドキュメント1の内容", encoding='utf-8')
        
        file2 = temp_dir / "doc2.txt"
        file2.write_text("ドキュメント2の内容", encoding='utf-8')
        
        # Create non-txt file (should be ignored)
        other_file = temp_dir / "readme.md"
        other_file.write_text("This should be ignored", encoding='utf-8')
        
        # Test reading
        documents = read_documents(str(temp_dir))
        
        assert len(documents) == 2
        filenames = [doc['filename'] for doc in documents]
        assert "doc1.txt" in filenames
        assert "doc2.txt" in filenames
        assert "readme.md" not in filenames
    
    def test_read_nonexistent_path(self):
        """Test reading from non-existent path."""
        with pytest.raises(ValueError, match="Invalid input path"):
            read_documents("/nonexistent/path")
    
    def test_read_empty_directory(self, temp_dir):
        """Test reading from empty directory."""
        documents = read_documents(str(temp_dir))
        assert documents == []


class TestReadSingleFile:
    """Test cases for _read_single_file function."""
    
    def test_read_txt_file(self, temp_dir):
        """Test reading a text file."""
        test_file = temp_dir / "test.txt"
        test_content = "テスト内容"
        test_file.write_text(test_content, encoding='utf-8')
        
        documents = _read_single_file(test_file)
        
        assert len(documents) == 1
        assert documents[0]['filename'] == "test.txt"
        assert documents[0]['content'] == test_content
    
    def test_read_json_file_dict(self, temp_dir):
        """Test reading JSON file with dictionary."""
        test_file = temp_dir / "test.json"
        test_data = {"key": "value", "text": "テストデータ"}
        test_file.write_text(json.dumps(test_data, ensure_ascii=False), encoding='utf-8')
        
        documents = _read_single_file(test_file)
        
        assert len(documents) == 1
        assert documents[0]['filename'] == "test.json"
        assert "テストデータ" in documents[0]['content']
    
    def test_read_unsupported_file(self, temp_dir):
        """Test reading unsupported file type."""
        test_file = temp_dir / "test.pdf"
        test_file.write_text("Some content", encoding='utf-8')
        
        documents = _read_single_file(test_file)
        
        assert documents == []


class TestReadDirectory:
    """Test cases for _read_directory function."""
    
    def test_read_multiple_files(self, temp_dir):
        """Test reading multiple text files."""
        # Create multiple files
        for i in range(3):
            file_path = temp_dir / f"doc{i}.txt"
            file_path.write_text(f"内容{i}", encoding='utf-8')
        
        documents = _read_directory(temp_dir)
        
        assert len(documents) == 3
        contents = [doc['content'] for doc in documents]
        assert "内容0" in contents
        assert "内容1" in contents
        assert "内容2" in contents
    
    def test_read_mixed_files(self, temp_dir):
        """Test reading directory with mixed file types."""
        # Create txt files
        txt_file = temp_dir / "doc.txt"
        txt_file.write_text("テキストファイル", encoding='utf-8')
        
        # Create other files (should be ignored)
        json_file = temp_dir / "data.json"
        json_file.write_text('{"key": "value"}', encoding='utf-8')
        
        documents = _read_directory(temp_dir)
        
        assert len(documents) == 1
        assert documents[0]['filename'] == "doc.txt"
        assert documents[0]['content'] == "テキストファイル"


class TestEnsureOutputDirectory:
    """Test cases for ensure_output_directory function."""
    
    def test_create_new_directory(self, temp_dir):
        """Test creating a new directory."""
        new_dir = temp_dir / "new_output"
        assert not new_dir.exists()
        
        result = ensure_output_directory(str(new_dir))
        
        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result == new_dir
    
    def test_existing_directory(self, temp_dir):
        """Test with existing directory."""
        existing_dir = temp_dir / "existing"
        existing_dir.mkdir()
        
        result = ensure_output_directory(str(existing_dir))
        
        assert existing_dir.exists()
        assert result == existing_dir
    
    def test_nested_directory_creation(self, temp_dir):
        """Test creating nested directories."""
        nested_dir = temp_dir / "level1" / "level2" / "output"
        
        result = ensure_output_directory(str(nested_dir))
        
        assert nested_dir.exists()
        assert nested_dir.is_dir()
        assert result == nested_dir


class TestUtilsIntegration:
    """Integration tests for utils functions."""
    
    def test_full_workflow(self, sample_documents):
        """Test complete workflow with real files."""
        # Read documents
        documents = read_documents(str(sample_documents))
        
        # Should read 2 txt files + 2 items from JSON
        assert len(documents) >= 2
        
        # Check content types
        filenames = [doc['filename'] for doc in documents]
        txt_files = [f for f in filenames if f.endswith('.txt')]
        assert len(txt_files) >= 2
        
        # Verify content
        for doc in documents:
            assert 'filename' in doc
            assert 'content' in doc
            assert len(doc['content']) > 0
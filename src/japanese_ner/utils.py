"""
Utility functions for file handling and data processing.
"""

import json
from pathlib import Path
from typing import List, Dict, Any


def read_documents(input_path: str) -> List[Dict[str, str]]:
    """
    Read documents from file or directory.
    
    Args:
        input_path: Path to input file or directory
        
    Returns:
        List of documents with filename and content
        
    Raises:
        ValueError: If input path is invalid
    """
    documents = []
    path = Path(input_path)
    
    if path.is_file():
        documents.extend(_read_single_file(path))
    elif path.is_dir():
        documents.extend(_read_directory(path))
    else:
        raise ValueError(f"Invalid input path: {input_path}")
        
    return documents


def _read_single_file(file_path: Path) -> List[Dict[str, str]]:
    """
    Read a single file and return document(s).
    
    Args:
        file_path: Path to the file
        
    Returns:
        List of documents
    """
    documents = []
    
    if file_path.suffix == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        documents.append({
            'filename': file_path.name,
            'content': content
        })
    elif file_path.suffix == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                for i, item in enumerate(data):
                    documents.append({
                        'filename': f"{file_path.stem}_{i+1}",
                        'content': str(item)
                    })
            else:
                documents.append({
                    'filename': file_path.name,
                    'content': str(data)
                })
    
    return documents


def _read_directory(dir_path: Path) -> List[Dict[str, str]]:
    """
    Read all .txt files from a directory.
    
    Args:
        dir_path: Path to the directory
        
    Returns:
        List of documents
    """
    documents = []
    
    for file_path in dir_path.glob('*.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        documents.append({
            'filename': file_path.name,
            'content': content
        })
    
    return documents


def ensure_output_directory(output_path: str) -> Path:
    """
    Ensure output directory exists.
    
    Args:
        output_path: Path to output directory
        
    Returns:
        Path object for the directory
    """
    path = Path(output_path)
    path.mkdir(exist_ok=True)
    return path
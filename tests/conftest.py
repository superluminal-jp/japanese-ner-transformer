"""
Pytest configuration and shared fixtures.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import sys

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from japanese_ner import NERAnalyzer, BatchNERAnalyzer


@pytest.fixture
def sample_text():
    """Sample Japanese text for testing."""
    return """2024年11月15日、東京国際フォーラムで開催されたAI技術カンファレンスにおいて、OpenAI社のCEOサム・アルトマン氏が基調講演を行いました。
同イベントには、トヨタ自動車株式会社の豊田章男会長が参加しました。会場では、最新のGPT-5モデルが紹介されました。"""


@pytest.fixture
def simple_text():
    """Simple text for basic testing."""
    return "田中太郎さんは東京でトヨタの車を購入しました。"


@pytest.fixture
def ner_analyzer():
    """Create NERAnalyzer instance for testing."""
    return NERAnalyzer()


@pytest.fixture
def batch_analyzer():
    """Create BatchNERAnalyzer instance for testing."""
    return BatchNERAnalyzer()


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_documents(temp_dir):
    """Create sample documents in temporary directory."""
    # Create sample text files
    doc1 = temp_dir / "doc1.txt"
    doc1.write_text("田中太郎は東京大学の教授です。", encoding='utf-8')
    
    doc2 = temp_dir / "doc2.txt"
    doc2.write_text("佐藤花子はソニー株式会社で働いています。", encoding='utf-8')
    
    # Create JSON file
    json_doc = temp_dir / "data.json"
    json_doc.write_text('["山田次郎は大阪に住んでいます。", "鈴木一郎はAppleの製品を使っています。"]', encoding='utf-8')
    
    return temp_dir


@pytest.fixture
def expected_entities():
    """Expected entities for validation."""
    return [
        {'word': '田中太郎', 'entity_type': 'PER'},
        {'word': '東京', 'entity_type': 'LOC'},
        {'word': 'トヨタ', 'entity_type': 'ORG'}
    ]


@pytest.fixture
def mock_ner_result():
    """Mock NER pipeline result."""
    return [
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
        }
    ]
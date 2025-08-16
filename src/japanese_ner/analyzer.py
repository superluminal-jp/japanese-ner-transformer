"""
Core NER analysis functionality.
"""

from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


class NERAnalyzer:
    """
    Core Named Entity Recognition analyzer for Japanese text.
    """
    
    def __init__(self, model_name: str = "tsmatz/xlm-roberta-ner-japanese"):
        """
        Initialize the NER analyzer.
        
        Args:
            model_name: Name of the pre-trained NER model to use
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.ner = pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
        )
        
        # Entity type descriptions in Japanese
        self.entity_descriptions = {
            'PER': '人名',
            'ORG': '一般企業・組織',
            'ORG-P': '政治組織',
            'ORG-O': 'その他の組織',
            'LOC': '場所・地名',
            'INS': '施設・機関',
            'PRD': '製品',
            'EVT': 'イベント'
        }

    def analyze(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of extracted entities with metadata
        """
        ner_results = self.ner(text)
        entities = []
        
        for entity in ner_results:
            entities.append({
                'word': entity['word'],
                'entity_type': entity['entity_group'],
                'score': entity['score'],
                'start': entity.get('start', 0),
                'end': entity.get('end', 0),
                'description': self.entity_descriptions.get(entity['entity_group'], '不明')
            })
            
        return entities

    def get_entity_types(self) -> Dict[str, str]:
        """
        Get supported entity types and their descriptions.
        
        Returns:
            Dictionary mapping entity tags to descriptions
        """
        return self.entity_descriptions.copy()
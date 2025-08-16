"""
Core NER analysis functionality.
"""

from typing import List, Dict, Any
from datetime import datetime
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
        
        # Entity type descriptions based on model specification
        self.entity_descriptions = {
            'O': 'others or nothing',
            'PER': 'person',
            'ORG': 'general corporation organization', 
            'ORG-P': 'political organization',
            'P': 'political organization',  # Widget tag mapping for ORG-P
            'ORG-O': 'other organization',
            'LOC': 'location',
            'INS': 'institution, facility',
            'PRD': 'product',
            'EVT': 'event'
        }

    def analyze(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text. Automatically handles long text using chunking strategy.
        
        Args:
            text: Input text to analyze (can be short or very long)
            
        Returns:
            List of extracted entities with metadata
        """
        # Check if text is too long for single processing
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= 400:
            # Short text: use direct processing
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
        else:
            # Long text: use chunking strategy
            chunks = self._split_text_into_chunks(text)
            all_entities = []
            
            for chunk in chunks:
                chunk_entities = self.ner(chunk["text"])
                
                # Adjust entity positions to global coordinates
                for entity in chunk_entities:
                    adjusted_entity = {
                        'word': entity['word'],
                        'entity_type': entity['entity_group'],
                        'score': entity['score'],
                        'start': entity.get('start', 0) + chunk["start_offset"],
                        'end': entity.get('end', 0) + chunk["start_offset"],
                        'description': self.entity_descriptions.get(entity['entity_group'], '不明')
                    }
                    all_entities.append(adjusted_entity)
            
            # Merge overlapping entities
            return self._merge_overlapping_entities(all_entities)

    def get_entity_types(self) -> Dict[str, str]:
        """
        Get supported entity types and their descriptions.
        
        Returns:
            Dictionary mapping entity tags to descriptions
        """
        return self.entity_descriptions.copy()

    def _split_text_into_chunks(self, text: str, max_tokens: int = 400, overlap: int = 50) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks for processing.
        
        Args:
            text: Input text to split
            max_tokens: Maximum tokens per chunk
            overlap: Overlap tokens between chunks
            
        Returns:
            List of chunk dictionaries with text and offset information
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        
        if len(tokens) <= max_tokens:
            return [{"text": text, "start_offset": 0, "end_offset": len(text)}]
        
        start = 0
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            # Find actual character positions
            start_char = len(self.tokenizer.decode(tokens[:start], skip_special_tokens=True))
            end_char = len(self.tokenizer.decode(tokens[:end], skip_special_tokens=True))
            
            chunks.append({
                "text": chunk_text,
                "start_offset": start_char,
                "end_offset": end_char
            })
            
            start = end - overlap
            if start >= len(tokens) - overlap:
                break
                
        return chunks

    def _merge_overlapping_entities(self, all_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge overlapping entities from different chunks.
        
        Args:
            all_entities: List of all entities from all chunks
            
        Returns:
            Deduplicated and merged entity list
        """
        if not all_entities:
            return []
        
        # Sort by start position
        sorted_entities = sorted(all_entities, key=lambda x: x['start'])
        merged = [sorted_entities[0]]
        
        for current in sorted_entities[1:]:
            last = merged[-1]
            
            # Check for overlap or very close entities of same type
            overlap_threshold = min(len(last['word']), len(current['word'])) * 0.5
            
            if (current['start'] <= last['end'] + 5 and 
                current['entity_type'] == last['entity_type'] and
                abs(current['start'] - last['start']) <= overlap_threshold):
                
                # Merge: keep the one with higher score
                if current['score'] > last['score']:
                    merged[-1] = current
            else:
                merged.append(current)
                
        return merged

    def analyze_documents(self, input_path: str) -> List[Dict[str, Any]]:
        """
        Analyze multiple documents from file or directory.
        
        Args:
            input_path: Path to input file or directory
            
        Returns:
            List of analysis results
        """
        from .utils import read_documents
        
        documents = read_documents(input_path)
        results = []
        
        print(f"Found {len(documents)} documents")
        
        for doc in documents:
            print(f"Analyzing: {doc['filename']}")
            
            entities = self.analyze(doc['content'])
            
            results.append({
                'filename': doc['filename'],
                'content': doc['content'],
                'entities': entities,
                'entity_count': len(entities),
                'analysis_time': datetime.now().isoformat()
            })
            
        return results

    def generate_full_report(self, input_path: str, output_dir: str):
        """
        Perform complete batch analysis with all outputs.
        
        Args:
            input_path: Path to input documents
            output_dir: Directory for output files
        """
        from .utils import ensure_output_directory
        from .report import calculate_statistics, save_csv_report, save_markdown_report
        from .visualization import create_all_visualizations
        
        output_path = ensure_output_directory(output_dir)
        
        # Analyze documents
        print(f"Reading documents from: {input_path}")
        results = self.analyze_documents(input_path)
        
        # Generate statistics
        print("Generating statistics...")
        stats = calculate_statistics(results)
        
        # Save CSV report
        csv_path = output_path / 'ner_results.csv'
        save_csv_report(results, str(csv_path), self.entity_descriptions)
        
        # Create visualizations
        print("Creating visualizations...")
        create_all_visualizations(stats, str(output_path))
        
        # Generate markdown report
        print("Generating report...")
        report_path = output_path / 'analysis_report.md'
        save_markdown_report(stats, str(report_path), self.model_name, self.entity_descriptions)
        
        print(f"\nAnalysis complete! Results saved to: {output_path}")
        print(f"- CSV: {csv_path}")
        print(f"- Report: {report_path}")
        print(f"- Visualizations: {output_path}/*.png")


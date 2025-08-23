"""
Batch NER analysis for multiple documents.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from .analyzer import NERAnalyzer
from .utils import read_documents, ensure_output_directory
from .report import calculate_statistics, save_csv_report, save_markdown_report
from .logger import setup_logger, get_logger


class BatchNERAnalyzer:
    """
    Batch processor for analyzing multiple documents with NER.
    """
    
    def setup_logging(self):
        """
        Configure logging for batch analysis tracking.
        """
        self.logger = setup_logger("batch_analyzer")
        self.logger.info(f"Initialized BatchNERAnalyzer with model: {getattr(self, 'model_name', 'unknown')}")
    
    def __init__(self, model_name: str = "tsmatz/xlm-roberta-ner-japanese"):
        """
        Initialize batch analyzer.
        
        Args:
            model_name: Name of the pre-trained NER model to use
        """
        self.model_name = model_name
        self.setup_logging()
        self.analyzer = NERAnalyzer(model_name)

    def analyze_documents(self, input_path: str) -> List[Dict[str, Any]]:
        """
        Analyze multiple documents from file or directory.
        
        Args:
            input_path: Path to input file or directory
            
        Returns:
            List of analysis results
        """
        documents = read_documents(input_path)
        results = []
        
        self.logger.info(f"Found {len(documents)} documents")
        self.logger.info(f"Starting batch analysis of {len(documents)} documents")
        
        for i, doc in enumerate(documents, 1):
            self.logger.info(f"Analyzing: {doc['filename']} ({i}/{len(documents)})")
            self.logger.info(f"Processing document {i}/{len(documents)}: {doc['filename']}")
            
            entities = self.analyzer.analyze(doc['content'])
            self.logger.info(f"Extracted {len(entities)} entities from {doc['filename']}")
            
            results.append({
                'filename': doc['filename'],
                'content': doc['content'],
                'entities': entities,
                'entity_count': len(entities),
                'analysis_time': datetime.now().isoformat()
            })
        
        self.logger.info(f"Batch analysis complete. Processed {len(results)} documents with {sum(r['entity_count'] for r in results)} total entities")    
        return results

    def generate_full_report(self, input_path: str, output_dir: str):
        """
        Perform complete batch analysis with all outputs.
        
        Args:
            input_path: Path to input documents
            output_dir: Directory for output files
        """
        output_path = ensure_output_directory(output_dir)
        
        # Analyze documents
        self.logger.info(f"Reading documents from: {input_path}")
        results = self.analyze_documents(input_path)
        
        # Generate statistics
        self.logger.info("Generating statistics...")
        self.logger.info("Calculating TF-IDF metrics and generating statistics")
        stats = calculate_statistics(results)
        
        # Save CSV report
        csv_path = output_path / 'ner_results.csv'
        self.logger.info(f"Saving CSV report to {csv_path}")
        save_csv_report(results, str(csv_path), self.analyzer.entity_descriptions)
        
        # Generate markdown report
        self.logger.info("Generating markdown report...")
        report_path = output_path / 'analysis_report.md'
        self.logger.info(f"Saving markdown report to {report_path}")
        save_markdown_report(stats, str(report_path), self.model_name, self.analyzer.entity_descriptions)
        
        self.logger.info(f"Analysis complete! Results saved to: {output_path}")
        self.logger.info(f"CSV report: {csv_path}")
        self.logger.info(f"Markdown report: {report_path}")
        self.logger.info(f"Batch analysis completed successfully. Output saved to {output_path}")
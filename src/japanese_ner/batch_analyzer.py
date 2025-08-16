"""
Batch NER analysis for multiple documents.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from .analyzer import NERAnalyzer
from .utils import read_documents, ensure_output_directory
from .report import calculate_statistics, save_csv_report, save_markdown_report
from .visualization import create_all_visualizations


class BatchNERAnalyzer:
    """
    Batch processor for analyzing multiple documents with NER.
    """
    
    def __init__(self, model_name: str = "tsmatz/xlm-roberta-ner-japanese"):
        """
        Initialize batch analyzer.
        
        Args:
            model_name: Name of the pre-trained NER model to use
        """
        self.analyzer = NERAnalyzer(model_name)
        self.model_name = model_name

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
        
        print(f"Found {len(documents)} documents")
        
        for doc in documents:
            print(f"Analyzing: {doc['filename']}")
            
            entities = self.analyzer.analyze(doc['content'])
            
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
        output_path = ensure_output_directory(output_dir)
        
        # Analyze documents
        print(f"Reading documents from: {input_path}")
        results = self.analyze_documents(input_path)
        
        # Generate statistics
        print("Generating statistics...")
        stats = calculate_statistics(results)
        
        # Save CSV report
        csv_path = output_path / 'ner_results.csv'
        save_csv_report(results, str(csv_path), self.analyzer.entity_descriptions)
        
        # Create visualizations
        print("Creating visualizations...")
        create_all_visualizations(stats, str(output_path))
        
        # Generate markdown report
        print("Generating report...")
        report_path = output_path / 'analysis_report.md'
        save_markdown_report(stats, str(report_path), self.model_name, self.analyzer.entity_descriptions)
        
        print(f"\nAnalysis complete! Results saved to: {output_path}")
        print(f"- CSV: {csv_path}")
        print(f"- Report: {report_path}")
        print(f"- Visualizations: {output_path}/*.png")
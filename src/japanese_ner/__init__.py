"""
Japanese Named Entity Recognition package using Transformers.

This package provides tools for extracting named entities from Japanese text
using pre-trained transformer models.
"""

from .analyzer import NERAnalyzer
from .batch_analyzer import BatchNERAnalyzer

__version__ = "1.0.0"
__all__ = ["NERAnalyzer", "BatchNERAnalyzer"]
"""
DataHawk Evaluation Module

Comprehensive evaluation framework for the research system.
"""

from .metrics import (
    MetricsCalculator,
    TopicMetrics,
    TrendMetrics,
    ClassificationMetrics
)
from .ablation import AblationStudy, AblationResult

__all__ = [
    'MetricsCalculator',
    'TopicMetrics',
    'TrendMetrics',
    'ClassificationMetrics',
    'AblationStudy',
    'AblationResult',
]

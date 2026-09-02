"""
DataHawk Social Media Intelligence Platform
Trend Detection Module

Implements composite trend scoring, emerging trend classification,
temporal analysis, and explainable predictions.
"""

from .trend_score import TrendScorer, compute_trend_score
from .emerging import EmergingTrendClassifier, classify_trend
from .temporal import TemporalAnalyzer, aggregate_by_time
from .explanation import TrendExplainer, explain_trend

__all__ = [
    'TrendScorer',
    'compute_trend_score',
    'EmergingTrendClassifier',
    'classify_trend',
    'TemporalAnalyzer',
    'aggregate_by_time',
    'TrendExplainer',
    'explain_trend',
]

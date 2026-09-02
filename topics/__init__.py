"""
DataHawk Topics Module

Topic discovery using semantic clustering, LLM labeling, and quality evaluation.
"""

from .clustering import TopicClusterer, cluster_topics, ClusterResult
from .labeling import TopicLabeler, label_topics, TopicLabel
from .coherence import CoherenceEvaluator, compute_cluster_coherence
from .baselines import TfidfKMeansBaseline, LDABaseline, run_baseline_comparison

__all__ = [
    'TopicClusterer',
    'cluster_topics',
    'ClusterResult',
    'TopicLabeler',
    'label_topics',
    'TopicLabel',
    'CoherenceEvaluator',
    'compute_cluster_coherence',
    'TfidfKMeansBaseline',
    'LDABaseline',
    'run_baseline_comparison',
]

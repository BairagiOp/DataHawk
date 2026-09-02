"""
DataHawk Social Media Intelligence Platform
NLP Analysis Module

This module provides sentiment analysis, emotion classification,
named entity recognition, semantic embeddings, and keyword extraction.
"""

from .sentiment import SentimentAnalyzer, analyze_sentiment
from .emotion import EmotionClassifier, classify_emotion
from .ner import NamedEntityRecognizer, extract_entities
from .embeddings import EmbeddingModel, generate_embeddings
from .keywords import KeywordExtractor, extract_keywords

__all__ = [
    'SentimentAnalyzer',
    'analyze_sentiment',
    'EmotionClassifier',
    'classify_emotion',
    'NamedEntityRecognizer',
    'extract_entities',
    'EmbeddingModel',
    'generate_embeddings',
    'KeywordExtractor',
    'extract_keywords',
]

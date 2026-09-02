"""
Unit Tests for NLP Module

Tests for sentiment analysis, emotion classification, NER, and embeddings.
"""

import unittest
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.sentiment import SentimentAnalyzer
from nlp.emotion import EmotionClassifier
from nlp.keywords import KeywordExtractor
from nlp.embeddings import EmbeddingModel


class TestSentimentAnalyzer(unittest.TestCase):
    """Test sentiment analysis"""

    def setUp(self):
        self.analyzer = SentimentAnalyzer(method='vader')

    def test_positive_sentiment(self):
        """Test positive sentiment detection"""
        text = "This is absolutely wonderful! I love it!"
        result = self.analyzer.analyze(text)

        self.assertEqual(result['sentiment'], 'positive')
        self.assertGreater(result['details']['positive'], 0.4)

    def test_negative_sentiment(self):
        """Test negative sentiment detection"""
        text = "This is terrible. Very disappointed and frustrated."
        result = self.analyzer.analyze(text)

        self.assertEqual(result['sentiment'], 'negative')
        self.assertGreater(result['details']['negative'], 0.4)

    def test_neutral_sentiment(self):
        """Test neutral sentiment detection"""
        text = "The system processes data."
        result = self.analyzer.analyze(text)

        self.assertEqual(result['sentiment'], 'neutral')

    def test_batch_analysis(self):
        """Test batch sentiment analysis"""
        texts = [
            "Great work!",
            "This is bad.",
            "Normal text here."
        ]

        results = self.analyzer.analyze_batch(texts)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]['sentiment'], 'positive')
        self.assertEqual(results[1]['sentiment'], 'negative')


class MockLLMClient:
    def __init__(self, response):
        self.response = response

    def generate_json(self, prompt):
        return self.response


class TestEmotionClassifier(unittest.TestCase):
    """Test emotion classification"""

    def test_neutral_fallback(self):
        """Test fallback to neutral when no LLM is present"""
        classifier = EmotionClassifier(llm_client=None)
        result = classifier.classify("I am happy")
        self.assertEqual(result['emotion'], 'neutral')
        self.assertEqual(result['confidence'], 0.0)

    def test_joy_emotion(self):
        """Test joy emotion detection"""
        mock_response = {
            "emotion": "joy",
            "confidence": 0.9,
            "intensities": {
                "joy": 0.9, "anger": 0.0, "sadness": 0.0,
                "fear": 0.0, "surprise": 0.1, "disgust": 0.0, "neutral": 0.0
            }
        }
        classifier = EmotionClassifier(llm_client=MockLLMClient(mock_response))
        result = classifier.classify("I'm so happy and excited about this!")

        self.assertEqual(result['emotion'], 'joy')
        self.assertGreater(result['intensities']['joy'], 0.5)

    def test_anger_emotion(self):
        """Test anger emotion detection"""
        mock_response = {
            "emotion": "anger",
            "confidence": 0.85,
            "intensities": {
                "joy": 0.0, "anger": 0.85, "sadness": 0.0,
                "fear": 0.0, "surprise": 0.0, "disgust": 0.1, "neutral": 0.0
            }
        }
        classifier = EmotionClassifier(llm_client=MockLLMClient(mock_response))
        result = classifier.classify("I'm furious! This is unacceptable!")

        self.assertEqual(result['emotion'], 'anger')
        self.assertGreater(result['intensities']['anger'], 0.5)


class TestKeywordExtractor(unittest.TestCase):
    """Test keyword extraction"""

    def setUp(self):
        self.extractor = KeywordExtractor()

    def test_keyword_extraction(self):
        """Test basic keyword extraction"""
        text = "Machine learning and artificial intelligence are transforming healthcare"
        keywords = self.extractor.extract(text, top_k=5)

        self.assertTrue(len(keywords) > 0)
        self.assertTrue(any('machine' in kw[0].lower() or 'learning' in kw[0].lower()
                          for kw in keywords))

    def test_batch_extraction(self):
        """Test batch keyword extraction"""
        texts = [
            "AI and machine learning",
            "Deep learning neural networks",
            "Natural language processing"
        ]

        from nlp.keywords import extract_keywords
        all_keywords = []
        for text in texts:
            all_keywords.extend(extract_keywords(text, method='tfidf', top_k=2))

        self.assertTrue(len(all_keywords) > 0)


class TestEmbeddingModel(unittest.TestCase):
    """Test text embeddings"""

    def setUp(self):
        self.model = EmbeddingModel()

    def test_single_embedding(self):
        """Test single text embedding"""
        text = "Machine learning is fascinating"
        embedding = self.model.encode_single(text)

        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(len(embedding), 384)  # MiniLM dimension

    def test_batch_embedding(self):
        """Test batch embedding"""
        texts = [
            "First text",
            "Second text",
            "Third text"
        ]

        embeddings = self.model.encode(texts)

        self.assertEqual(embeddings.shape[0], 3)
        self.assertEqual(embeddings.shape[1], 384)

    def test_similarity(self):
        """Test semantic similarity"""
        text1 = "Machine learning is great"
        text2 = "AI and ML are wonderful"
        text3 = "The weather is sunny"

        emb1 = self.model.encode_single(text1)
        emb2 = self.model.encode_single(text2)
        emb3 = self.model.encode_single(text3)

        sim_related = self.model.compute_similarity(emb1, emb2)
        sim_unrelated = self.model.compute_similarity(emb1, emb3)

        # Related texts should have higher similarity
        self.assertGreater(sim_related, sim_unrelated)


if __name__ == '__main__':
    unittest.main()

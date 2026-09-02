"""
Sentiment Analysis for Social Media Posts

Implements:
1. Baseline: VADER (rule-based, social media optimized)
2. Proposed: LLM-based sentiment classification

Comparison for research evaluation.
"""

from typing import Dict, Optional, List
from enum import Enum


class SentimentLabel(str, Enum):
    """Sentiment labels"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class SentimentAnalyzer:
    """
    Sentiment analysis with multiple methods.

    Methods:
    - VADER (baseline): Rule-based, social media optimized
    - LLM (proposed): Gemini-based classification
    """

    def __init__(self, method: str = "vader", llm_client=None):
        """
        Args:
            method: "vader" or "llm"
            llm_client: LLM client for LLM-based analysis
        """
        self.method = method
        self.llm_client = llm_client

        # Initialize VADER if needed
        if method == "vader":
            self._init_vader()

    def _init_vader(self):
        """Initialize VADER sentiment analyzer"""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.vader = SentimentIntensityAnalyzer()
        except ImportError:
            print("Warning: vaderSentiment not installed.")
            print("Install with: pip install vaderSentiment")
            self.vader = None

    def analyze(self, text: str) -> Dict[str, any]:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze

        Returns:
            {
                "sentiment": str,  # positive/negative/neutral/mixed
                "score": float,    # [-1, 1] negative to positive
                "confidence": float,  # [0, 1]
                "method": str,
                "details": dict  # Method-specific details
            }
        """
        if not text or not isinstance(text, str):
            return {
                "sentiment": SentimentLabel.NEUTRAL,
                "score": 0.0,
                "confidence": 0.0,
                "method": self.method,
                "details": {}
            }

        if self.method == "vader":
            return self._analyze_vader(text)
        elif self.method == "llm":
            return self._analyze_llm(text)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _analyze_vader(self, text: str) -> Dict[str, any]:
        """Analyze sentiment using VADER"""
        if not self.vader:
            raise RuntimeError("VADER not available")

        scores = self.vader.polarity_scores(text)

        # Extract scores
        compound = scores['compound']
        pos = scores['pos']
        neg = scores['neg']
        neu = scores['neu']

        # Classify sentiment
        # VADER compound score: -1 (most negative) to +1 (most positive)
        if compound >= 0.05:
            sentiment = SentimentLabel.POSITIVE
        elif compound <= -0.05:
            sentiment = SentimentLabel.NEGATIVE
        else:
            sentiment = SentimentLabel.NEUTRAL

        # Check for mixed sentiment
        # If both pos and neg are significant, classify as mixed
        if pos > 0.3 and neg > 0.3:
            sentiment = SentimentLabel.MIXED

        # Confidence is based on how far from neutral
        confidence = abs(compound)

        return {
            "sentiment": sentiment.value,
            "score": compound,
            "confidence": confidence,
            "method": "vader",
            "details": {
                "positive": pos,
                "negative": neg,
                "neutral": neu,
                "compound": compound
            }
        }

    def _analyze_llm(self, text: str) -> Dict[str, any]:
        """Analyze sentiment using LLM"""
        if not self.llm_client:
            raise RuntimeError("LLM client not provided")

        prompt = f"""Classify the sentiment of this text.

Text: "{text}"

Return JSON with:
- sentiment: one of "positive", "negative", "neutral", or "mixed"
- score: float from -1.0 (most negative) to 1.0 (most positive)
- confidence: float from 0.0 to 1.0

Rules:
- Positive: expresses satisfaction, approval, optimism, enthusiasm
- Negative: expresses dissatisfaction, criticism, pessimism, concern
- Neutral: factual, informational, no clear sentiment
- Mixed: contains both positive and negative sentiments

Return only valid JSON.
"""

        try:
            response = self.llm_client.generate_json(prompt)

            sentiment = response.get("sentiment", "neutral")
            score = float(response.get("score", 0.0))
            confidence = float(response.get("confidence", 0.5))

            # Validate sentiment
            if sentiment not in [e.value for e in SentimentLabel]:
                sentiment = SentimentLabel.NEUTRAL.value

            # Clamp score and confidence
            score = max(-1.0, min(1.0, score))
            confidence = max(0.0, min(1.0, confidence))

            return {
                "sentiment": sentiment,
                "score": score,
                "confidence": confidence,
                "method": "llm",
                "details": response
            }

        except Exception as e:
            # Fallback to neutral on error
            return {
                "sentiment": SentimentLabel.NEUTRAL.value,
                "score": 0.0,
                "confidence": 0.0,
                "method": "llm",
                "details": {"error": str(e)}
            }

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze sentiment for batch of texts"""
        return [self.analyze(text) for text in texts]


# Convenience functions
def analyze_sentiment(text: str, method: str = "vader", llm_client=None) -> str:
    """
    Quick sentiment analysis returning just the label.

    Args:
        text: Text to analyze
        method: "vader" or "llm"
        llm_client: LLM client for LLM method

    Returns:
        Sentiment label: "positive", "negative", "neutral", or "mixed"
    """
    analyzer = SentimentAnalyzer(method=method, llm_client=llm_client)
    result = analyzer.analyze(text)
    return result["sentiment"]


def get_sentiment_score(text: str, method: str = "vader", llm_client=None) -> float:
    """
    Get sentiment score [-1, 1].

    Returns:
        Score from -1 (most negative) to 1 (most positive)
    """
    analyzer = SentimentAnalyzer(method=method, llm_client=llm_client)
    result = analyzer.analyze(text)
    return result["score"]


class SentimentComparison:
    """Compare sentiment analysis methods"""

    def __init__(self, llm_client=None):
        self.vader_analyzer = SentimentAnalyzer(method="vader")
        self.llm_analyzer = SentimentAnalyzer(method="llm", llm_client=llm_client) if llm_client else None

    def compare(self, text: str) -> Dict:
        """Compare VADER and LLM sentiment analysis"""
        vader_result = self.vader_analyzer.analyze(text)

        comparison = {
            "text": text[:100],
            "vader": vader_result,
        }

        if self.llm_analyzer:
            llm_result = self.llm_analyzer.analyze(text)
            comparison["llm"] = llm_result

            # Check agreement
            comparison["agreement"] = (
                vader_result["sentiment"] == llm_result["sentiment"]
            )

        return comparison


# Testing
if __name__ == "__main__":
    # Test VADER
    analyzer = SentimentAnalyzer(method="vader")

    test_texts = [
        "This is absolutely amazing! I love it! 🔥",
        "This is terrible. Worst experience ever.",
        "The product arrived on time.",
        "I like some features but hate others. Mixed feelings.",
        "AI agents are transforming software development.",
    ]

    print("Sentiment Analysis Tests (VADER):\n")
    for text in test_texts:
        result = analyzer.analyze(text)
        print(f"Text: {text}")
        print(f"  Sentiment: {result['sentiment']}")
        print(f"  Score: {result['score']:.3f}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Details: {result['details']}")
        print()

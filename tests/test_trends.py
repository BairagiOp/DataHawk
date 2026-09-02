"""
Unit Tests for Trend Detection Module

Tests for trend scoring, classification, and temporal analysis.
"""

import unittest
import sys
import os
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trends.trend_score import TrendScorer, TrendFeatures, TrendScore
from trends.emerging import EmergingTrendClassifier, TrendCategory, TrendClassification
from trends.temporal import TemporalAnalyzer
import pandas as pd


def _make_features(volume=50.0, volume_normalized=0.5, baseline_volume=30.0,
                   growth_rate=50.0, growth_absolute=20.0, velocity=10.0,
                   acceleration=2.0, engagement=100.0, engagement_normalized=0.5,
                   engagement_growth=10.0, days_since_first=5, novelty_score=0.5,
                   unique_authors=10, cross_source_presence=2):
    """Helper to build TrendFeatures with sensible defaults."""
    return TrendFeatures(
        volume=volume, volume_normalized=volume_normalized,
        baseline_volume=baseline_volume, growth_rate=growth_rate,
        growth_absolute=growth_absolute, velocity=velocity,
        acceleration=acceleration, engagement=engagement,
        engagement_normalized=engagement_normalized,
        engagement_growth=engagement_growth,
        days_since_first=days_since_first, novelty_score=novelty_score,
        unique_authors=unique_authors, cross_source_presence=cross_source_presence,
        timestamp=datetime(2026, 8, 1)
    )


class TestTrendScorer(unittest.TestCase):
    """Test trend scoring functionality"""

    def setUp(self):
        self.scorer = TrendScorer(alpha=0.2, beta=0.4, gamma=0.3, delta=0.1)

    def test_high_growth_trend(self):
        """Test scoring for high growth trend"""
        features = _make_features(
            volume=80.0, volume_normalized=0.8, baseline_volume=20.0,
            growth_rate=300.0, growth_absolute=60.0, velocity=100.0,
            acceleration=20.0, engagement=160.0, engagement_normalized=0.8,
            engagement_growth=300.0, days_since_first=1, novelty_score=0.8
        )
        result = self.scorer.compute_score(features)

        # High growth should produce high score
        self.assertGreater(result.score, 0.5)

    def test_stable_trend(self):
        """Test scoring for stable trend"""
        features = _make_features(
            volume=50.0, volume_normalized=0.5, baseline_volume=48.0,
            growth_rate=4.0, growth_absolute=2.0, velocity=2.0,
            acceleration=0.5, engagement=100.0, engagement_normalized=0.5,
            engagement_growth=2.0, days_since_first=30, novelty_score=0.2
        )
        result = self.scorer.compute_score(features)

        # Stable trend should have moderate score
        self.assertLess(result.score, 0.7)

    def test_weight_configuration(self):
        """Test different weight configurations"""
        features = _make_features(
            volume=30.0, volume_normalized=0.3, baseline_volume=10.0,
            growth_rate=100.0, growth_absolute=20.0, velocity=50.0,
            acceleration=10.0, engagement=60.0, engagement_normalized=0.3,
            engagement_growth=50.0, days_since_first=3, novelty_score=0.5
        )

        # Heavy volume weight
        scorer1 = TrendScorer(alpha=0.7, beta=0.1, gamma=0.1, delta=0.1)
        score1 = scorer1.compute_score(features).score

        # Heavy growth weight
        scorer2 = TrendScorer(alpha=0.1, beta=0.7, gamma=0.1, delta=0.1)
        score2 = scorer2.compute_score(features).score

        # Scores should differ based on weights
        self.assertNotEqual(score1, score2)


class TestEmergingTrendClassifier(unittest.TestCase):
    """Test emerging trend classification"""

    def setUp(self):
        self.classifier = EmergingTrendClassifier()

    def test_viral_classification(self):
        """Test viral trend detection"""
        result = self.classifier.classify(
            volume=200.0, baseline_volume=10.0, growth_rate=2.5
        )

        self.assertEqual(result.category, TrendCategory.VIRAL)

    def test_stable_classification(self):
        """Test stable trend detection"""
        result = self.classifier.classify(
            volume=100.0, baseline_volume=100.0, growth_rate=0.02
        )

        self.assertEqual(result.category, TrendCategory.STABLE)

    def test_declining_classification(self):
        """Test declining trend detection"""
        result = self.classifier.classify(
            volume=20.0, baseline_volume=100.0, growth_rate=-0.5
        )

        self.assertEqual(result.category, TrendCategory.DECLINING)


class TestTemporalAnalyzer(unittest.TestCase):
    """Test temporal analysis"""

    def _make_posts(self, topic_counts, days=3):
        """Build sample posts, topic_assignments, and topic_labels.

        Args:
            topic_counts: dict  topic_id -> per-day volume, e.g. {0: [10, 20, 30]}
            days: number of days (overridden by list length)

        Returns:
            posts, topic_assignments, topic_labels
        """
        posts = []
        assignments = {}
        labels = {}
        base = datetime(2026, 8, 1)
        idx = 0
        for tid, volumes in topic_counts.items():
            labels[tid] = f"topic_{tid}"
            for day, vol in enumerate(volumes):
                for _ in range(vol):
                    pid = f"p{idx}"
                    posts.append({
                        'post_id': pid,
                        'timestamp': base + timedelta(days=day),
                        'author_id_hash': f'a{idx % 5}',
                        'text': f't{tid} day{day}',
                        'likes': 1, 'comments': 0, 'shares': 0, 'views': 10,
                    })
                    assignments[pid] = tid
                    idx += 1
        return posts, assignments, labels

    def test_daily_aggregation(self):
        """Test aggregation by day returns expected columns and rows"""
        posts, assigns, labels = self._make_posts({0: [10, 20, 30]})
        analyzer = TemporalAnalyzer(time_window='1D')
        result = analyzer.aggregate_by_time(posts, assigns, labels)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('volume', result.columns)
        self.assertIn('topic_label', result.columns)
        # 3 days, 1 topic -> 3 rows
        self.assertEqual(len(result), 3)
        # Volumes should match input
        vols = sorted(result['volume'].tolist())
        self.assertEqual(vols, [10, 20, 30])

    def test_multiple_topics(self):
        """Test aggregation with two topics"""
        posts, assigns, labels = self._make_posts({
            0: [5, 10],
            1: [15, 20],
        })
        analyzer = TemporalAnalyzer(time_window='1D')
        result = analyzer.aggregate_by_time(posts, assigns, labels)

        # 2 days x 2 topics = 4 rows
        self.assertEqual(len(result), 4)
        topic_ids = set(result['topic_id'].tolist())
        self.assertEqual(topic_ids, {0, 1})

    def test_growth_rate_calculation(self):
        """Test growth rate computation via compute_growth_rates"""
        posts, assigns, labels = self._make_posts({0: [100, 200, 300]})
        analyzer = TemporalAnalyzer(time_window='1D')
        agg = analyzer.aggregate_by_time(posts, assigns, labels)
        result = analyzer.compute_growth_rates(agg)

        self.assertIn('volume_growth', result.columns)
        # Second and third rows should show positive growth
        growths = result.sort_values('timestamp')['volume_growth'].tolist()
        self.assertGreater(growths[1], 0)
        self.assertGreater(growths[2], 0)


if __name__ == '__main__':
    unittest.main()

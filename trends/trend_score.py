"""
Composite Trend Scoring Algorithm

Implements the research-oriented trend scoring formula:

TrendScore(topic, t) = α·V(t) + β·G(t) + γ·E(t) + δ·N(t)

Where:
- V(t) = Normalized volume (post count)
- G(t) = Growth rate (% change from previous period)
- E(t) = Normalized engagement (likes, shares, comments)
- N(t) = Novelty score (topic emergence from low baseline)
- α, β, γ, δ = Configurable weights (sum to 1.0)

This is a KEY RESEARCH COMPONENT.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class TrendFeatures:
    """
    Features used for trend scoring.

    All features should be extracted from actual data.
    """
    # Volume features
    volume: float  # Current volume (post count)
    volume_normalized: float  # Normalized to [0, 1]
    baseline_volume: float  # Average volume in baseline period

    # Growth features
    growth_rate: float  # Percentage growth from previous period
    growth_absolute: float  # Absolute change in volume
    velocity: float  # Average growth over last N periods
    acceleration: float  # Change in growth rate (second derivative)

    # Engagement features
    engagement: float  # Total engagement (likes + comments + shares)
    engagement_normalized: float  # Normalized to [0, 1]
    engagement_growth: float  # Percentage growth in engagement

    # Novelty features
    days_since_first: int  # Days since topic first appeared
    novelty_score: float  # Novelty decay: exp(-days / decay_constant)

    # Additional context
    unique_authors: int  # Number of unique authors
    cross_source_presence: int  # Number of different sources
    timestamp: datetime  # When features were computed


@dataclass
class TrendScore:
    """
    Trend score result with breakdown.
    """
    topic_id: int
    topic_label: str
    score: float  # Final composite score [0, 1]
    timestamp: datetime

    # Component scores
    volume_component: float
    growth_component: float
    engagement_component: float
    novelty_component: float

    # Features used
    features: TrendFeatures

    # Rank
    rank: Optional[int] = None


class TrendScorer:
    """
    Compute composite trend scores for topics.

    This is the PROPOSED METHOD in the research paper.
    Compare against baseline methods (frequency, TF-IDF, volume+growth).
    """

    def __init__(self,
                 alpha: float = 0.2,  # Volume weight
                 beta: float = 0.4,   # Growth weight
                 gamma: float = 0.3,  # Engagement weight
                 delta: float = 0.1,  # Novelty weight
                 novelty_decay: float = 7.0):  # Days for novelty decay
        """
        Args:
            alpha: Volume weight (recommend 0.2)
            beta: Growth weight (recommend 0.4) - most important
            gamma: Engagement weight (recommend 0.3)
            delta: Novelty weight (recommend 0.1)
            novelty_decay: Decay constant for novelty score (days)
        """
        # Validate weights
        total_weight = alpha + beta + gamma + delta
        if not (0.99 <= total_weight <= 1.01):  # Allow small float error
            raise ValueError(f"Weights must sum to 1.0 (got {total_weight})")

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.novelty_decay = novelty_decay

        # Track min/max for normalization (updated dynamically)
        self.volume_min = 0
        self.volume_max = 1
        self.engagement_min = 0
        self.engagement_max = 1

    def compute_score(self, features: TrendFeatures) -> TrendScore:
        """
        Compute composite trend score from features.

        Args:
            features: TrendFeatures instance

        Returns:
            TrendScore with breakdown
        """
        # Normalize volume (already normalized in features)
        V = features.volume_normalized

        # Normalize growth rate
        # Growth rate is already a percentage, map to [0, 1]
        # Assume growth rates typically range from -100% to +500%
        G = self._normalize_growth(features.growth_rate)

        # Normalize engagement (already normalized in features)
        E = features.engagement_normalized

        # Novelty score (already computed in features)
        N = features.novelty_score

        # Compute composite score
        score = (
            self.alpha * V +
            self.beta * G +
            self.gamma * E +
            self.delta * N
        )

        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))

        return TrendScore(
            topic_id=0,  # Set by caller
            topic_label="",  # Set by caller
            score=score,
            timestamp=features.timestamp,
            volume_component=self.alpha * V,
            growth_component=self.beta * G,
            engagement_component=self.gamma * E,
            novelty_component=self.delta * N,
            features=features
        )

    def _normalize_growth(self, growth_rate: float) -> float:
        """
        Normalize growth rate to [0, 1].

        Maps growth rates:
        - Negative growth → 0-0.4
        - Zero growth → 0.5
        - Positive growth → 0.5-1.0
        """
        if growth_rate <= -100:
            return 0.0
        elif growth_rate < 0:
            # Map [-100, 0] to [0, 0.5]
            return 0.5 + (growth_rate / 200)
        elif growth_rate == 0:
            return 0.5
        else:
            # Map [0, 500] to [0.5, 1.0]
            # Cap at 500% growth
            return 0.5 + min(growth_rate / 1000, 0.5)

    def compute_novelty_score(self, days_since_first: int) -> float:
        """
        Compute novelty score using exponential decay.

        Formula: N(t) = exp(-days_since_first / decay_constant)

        Args:
            days_since_first: Days since topic first appeared

        Returns:
            Novelty score [0, 1]
        """
        return np.exp(-days_since_first / self.novelty_decay)

    def rank_topics(self, scores: List[TrendScore]) -> List[TrendScore]:
        """
        Rank topics by score (descending).

        Args:
            scores: List of TrendScore objects

        Returns:
            Sorted list with rank field set
        """
        # Sort by score (descending)
        sorted_scores = sorted(scores, key=lambda x: x.score, reverse=True)

        # Assign ranks
        for rank, score in enumerate(sorted_scores, start=1):
            score.rank = rank

        return sorted_scores


class BaselineScorer:
    """
    Baseline trend scoring methods for comparison.

    Baselines:
    1. Frequency only (volume)
    2. TF-IDF weighted volume
    3. Volume + Growth only
    """

    @staticmethod
    def frequency_score(features: TrendFeatures) -> float:
        """
        Baseline 1: Pure frequency (volume only).

        Returns normalized volume.
        """
        return features.volume_normalized

    @staticmethod
    def volume_growth_score(features: TrendFeatures) -> float:
        """
        Baseline 3: Volume + Growth only.

        Equal weights to volume and growth.
        """
        scorer = TrendScorer(alpha=0.5, beta=0.5, gamma=0.0, delta=0.0)
        result = scorer.compute_score(features)
        return result.score


def compute_trend_score(features: TrendFeatures,
                       alpha: float = 0.2,
                       beta: float = 0.4,
                       gamma: float = 0.3,
                       delta: float = 0.1) -> float:
    """
    Convenience function for computing trend score.

    Args:
        features: TrendFeatures instance
        alpha, beta, gamma, delta: Weights

    Returns:
        Trend score [0, 1]
    """
    scorer = TrendScorer(alpha=alpha, beta=beta, gamma=gamma, delta=delta)
    result = scorer.compute_score(features)
    return result.score


def normalize_features(volumes: List[float],
                      engagements: List[float]) -> Tuple[List[float], List[float]]:
    """
    Normalize volume and engagement features using min-max normalization.

    Args:
        volumes: List of volume values
        engagements: List of engagement values

    Returns:
        (normalized_volumes, normalized_engagements)
    """
    def min_max_normalize(values: List[float]) -> List[float]:
        if not values or max(values) == min(values):
            return [0.5] * len(values)

        min_val = min(values)
        max_val = max(values)

        return [(v - min_val) / (max_val - min_val) for v in values]

    return min_max_normalize(volumes), min_max_normalize(engagements)


# Testing
if __name__ == "__main__":
    print("Trend Scoring Tests:\n")

    # Create sample features for different scenarios

    # Scenario 1: Emerging trend (low baseline, high growth)
    emerging_features = TrendFeatures(
        volume=25,
        volume_normalized=0.3,
        baseline_volume=8,
        growth_rate=212.0,  # 212% growth
        growth_absolute=17,
        velocity=50.0,
        acceleration=15.0,
        engagement=450,
        engagement_normalized=0.4,
        engagement_growth=180.0,
        days_since_first=2,
        novelty_score=0.85,
        unique_authors=18,
        cross_source_presence=4,
        timestamp=datetime.now()
    )

    # Scenario 2: Stable popular trend (high baseline, low growth)
    stable_features = TrendFeatures(
        volume=500,
        volume_normalized=0.9,
        baseline_volume=480,
        growth_rate=4.2,  # 4.2% growth
        growth_absolute=20,
        velocity=3.0,
        acceleration=0.5,
        engagement=8500,
        engagement_normalized=0.85,
        engagement_growth=6.0,
        days_since_first=45,
        novelty_score=0.02,
        unique_authors=320,
        cross_source_presence=12,
        timestamp=datetime.now()
    )

    # Scenario 3: Declining trend
    declining_features = TrendFeatures(
        volume=80,
        volume_normalized=0.4,
        baseline_volume=150,
        growth_rate=-46.7,  # -46.7% decline
        growth_absolute=-70,
        velocity=-30.0,
        acceleration=-8.0,
        engagement=1200,
        engagement_normalized=0.3,
        engagement_growth=-40.0,
        days_since_first=30,
        novelty_score=0.08,
        unique_authors=55,
        cross_source_presence=3,
        timestamp=datetime.now()
    )

    # Compute scores
    scorer = TrendScorer()

    scenarios = [
        ("Emerging Trend", emerging_features),
        ("Stable Popular Trend", stable_features),
        ("Declining Trend", declining_features)
    ]

    for name, features in scenarios:
        result = scorer.compute_score(features)

        print(f"{name}:")
        print(f"  Total Score: {result.score:.3f}")
        print(f"  Breakdown:")
        print(f"    Volume:     {result.volume_component:.3f} (α={scorer.alpha})")
        print(f"    Growth:     {result.growth_component:.3f} (β={scorer.beta})")
        print(f"    Engagement: {result.engagement_component:.3f} (γ={scorer.gamma})")
        print(f"    Novelty:    {result.novelty_component:.3f} (δ={scorer.delta})")
        print(f"  Features:")
        print(f"    Volume: {features.volume} (normalized: {features.volume_normalized:.2f})")
        print(f"    Growth Rate: {features.growth_rate:.1f}%")
        print(f"    Engagement: {features.engagement} (normalized: {features.engagement_normalized:.2f})")
        print(f"    Days Since First: {features.days_since_first}")
        print(f"    Novelty: {features.novelty_score:.3f}")
        print()

    # Compare baselines
    print("\nBaseline Comparisons:")
    print("Method                  | Emerging | Stable | Declining")
    print("-" * 60)

    for name, features in scenarios:
        freq_score = BaselineScorer.frequency_score(features)
        vg_score = BaselineScorer.volume_growth_score(features)
        full_score = scorer.compute_score(features).score

        print(f"Frequency (volume only) | {freq_score:.3f}    | {BaselineScorer.frequency_score(stable_features):.3f}   | {BaselineScorer.frequency_score(declining_features):.3f}")
        break

    print(f"Volume + Growth         | {BaselineScorer.volume_growth_score(emerging_features):.3f}    | {BaselineScorer.volume_growth_score(stable_features):.3f}   | {BaselineScorer.volume_growth_score(declining_features):.3f}")
    print(f"Proposed (Full)         | {scorer.compute_score(emerging_features).score:.3f}    | {scorer.compute_score(stable_features).score:.3f}   | {scorer.compute_score(declining_features).score:.3f}")

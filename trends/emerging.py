"""
Emerging Trend Classification

Classifies topics into trend categories based on temporal patterns:
- EMERGING: Low baseline → rapid growth
- VIRAL: Explosive growth (>200%)
- RISING: Consistent positive growth
- STABLE: Low variance in volume
- DECLINING: Negative growth trend

This is a KEY RESEARCH COMPONENT for RQ4 (early detection).
"""

from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


class TrendCategory(str, Enum):
    """Trend category labels"""
    EMERGING = "EMERGING"
    VIRAL = "VIRAL"
    RISING = "RISING"
    STABLE = "STABLE"
    DECLINING = "DECLINING"
    UNKNOWN = "UNKNOWN"


@dataclass
class TrendClassification:
    """
    Result of trend classification.
    """
    topic_id: int
    topic_label: str
    category: TrendCategory
    confidence: float  # [0, 1]
    timestamp: datetime

    # Criteria met
    criteria_met: List[str]

    # Feature values that led to classification
    volume: float
    baseline_volume: float
    growth_rate: float
    consecutive_growth_periods: int

    # Early warning score (higher = more likely to become major trend)
    early_warning_score: Optional[float] = None


class EmergingTrendClassifier:
    """
    Classify trends into categories based on temporal dynamics.

    Configurable thresholds allow for ablation studies.
    """

    def __init__(self,
                 emerging_threshold_low: int = 10,
                 emerging_threshold_growth: float = 0.5,
                 viral_threshold: float = 2.0,
                 stability_threshold: float = 0.1,
                 min_growth_periods: int = 2):
        """
        Args:
            emerging_threshold_low: Max baseline volume for emerging trends
            emerging_threshold_growth: Min growth rate (as ratio) for emerging
            viral_threshold: Min growth rate (as ratio) for viral
            stability_threshold: Max abs(growth rate) for stable
            min_growth_periods: Min consecutive periods for rising trend
        """
        self.emerging_threshold_low = emerging_threshold_low
        self.emerging_threshold_growth = emerging_threshold_growth
        self.viral_threshold = viral_threshold
        self.stability_threshold = stability_threshold
        self.min_growth_periods = min_growth_periods

    def classify(self,
                 volume: float,
                 baseline_volume: float,
                 growth_rate: float,
                 growth_history: Optional[List[float]] = None) -> TrendClassification:
        """
        Classify trend category.

        Args:
            volume: Current volume
            baseline_volume: Average volume in baseline period
            growth_rate: Current growth rate (as ratio, e.g., 1.5 = 150%)
            growth_history: List of recent growth rates (for trend analysis)

        Returns:
            TrendClassification
        """
        criteria_met = []

        # Check for VIRAL (highest priority)
        if growth_rate >= self.viral_threshold:
            category = TrendCategory.VIRAL
            criteria_met.append(f"Growth rate {growth_rate:.1%} >= {self.viral_threshold:.0%}")
            confidence = min(1.0, growth_rate / (self.viral_threshold * 2))

        # Check for EMERGING
        elif (baseline_volume < self.emerging_threshold_low and
              growth_rate >= self.emerging_threshold_growth):
            category = TrendCategory.EMERGING
            criteria_met.append(f"Low baseline ({baseline_volume:.0f} < {self.emerging_threshold_low})")
            criteria_met.append(f"High growth ({growth_rate:.1%} >= {self.emerging_threshold_growth:.0%})")
            confidence = 0.8

        # Check for RISING
        elif growth_rate > 0:
            # Check growth history if available
            consecutive_growth = self._count_consecutive_growth(growth_history) if growth_history else 1

            if consecutive_growth >= self.min_growth_periods:
                category = TrendCategory.RISING
                criteria_met.append(f"Positive growth for {consecutive_growth} periods")
                confidence = 0.7
            else:
                # Single period growth - still rising but lower confidence
                category = TrendCategory.RISING
                criteria_met.append("Positive growth")
                confidence = 0.5

        # Check for STABLE
        elif abs(growth_rate) <= self.stability_threshold:
            category = TrendCategory.STABLE
            criteria_met.append(f"Low variance (|{growth_rate:.1%}| <= {self.stability_threshold:.0%})")
            confidence = 0.7

        # DECLINING
        elif growth_rate < 0:
            category = TrendCategory.DECLINING
            criteria_met.append(f"Negative growth ({growth_rate:.1%})")

            # Check for sustained decline
            consecutive_decline = self._count_consecutive_decline(growth_history) if growth_history else 1
            if consecutive_decline >= self.min_growth_periods:
                criteria_met.append(f"Declining for {consecutive_decline} periods")
                confidence = 0.8
            else:
                confidence = 0.6

        else:
            category = TrendCategory.UNKNOWN
            confidence = 0.0

        # Compute early warning score for EMERGING trends
        early_warning_score = None
        if category == TrendCategory.EMERGING:
            early_warning_score = self._compute_early_warning_score(
                volume, baseline_volume, growth_rate
            )

        return TrendClassification(
            topic_id=0,  # Set by caller
            topic_label="",  # Set by caller
            category=category,
            confidence=confidence,
            timestamp=datetime.now(),
            criteria_met=criteria_met,
            volume=volume,
            baseline_volume=baseline_volume,
            growth_rate=growth_rate,
            consecutive_growth_periods=self._count_consecutive_growth(growth_history) if growth_history else 0,
            early_warning_score=early_warning_score
        )

    def _count_consecutive_growth(self, growth_history: Optional[List[float]]) -> int:
        """Count consecutive periods of positive growth"""
        if not growth_history:
            return 0

        count = 0
        for growth in reversed(growth_history):
            if growth > 0:
                count += 1
            else:
                break
        return count

    def _count_consecutive_decline(self, growth_history: Optional[List[float]]) -> int:
        """Count consecutive periods of negative growth"""
        if not growth_history:
            return 0

        count = 0
        for growth in reversed(growth_history):
            if growth < 0:
                count += 1
            else:
                break
        return count

    def _compute_early_warning_score(self,
                                     volume: float,
                                     baseline_volume: float,
                                     growth_rate: float) -> float:
        """
        Compute early warning score for emerging trends.

        Higher score = more likely to become a major trend.

        Formula combines:
        - Growth magnitude
        - Acceleration from baseline
        - Current momentum

        Returns score [0, 1]
        """
        # Growth magnitude component
        growth_component = min(growth_rate / 3.0, 0.5)  # Cap at 300% growth

        # Acceleration component (how much faster than baseline)
        if baseline_volume > 0:
            acceleration = (volume - baseline_volume) / baseline_volume
            accel_component = min(acceleration / 5.0, 0.3)  # Cap at 500% acceleration
        else:
            accel_component = 0.3  # Max if starting from zero

        # Momentum component (current absolute volume)
        momentum_component = min(volume / 100.0, 0.2)  # Cap at 100 posts

        score = growth_component + accel_component + momentum_component
        return min(score, 1.0)

    def batch_classify(self,
                      volumes: List[float],
                      baseline_volumes: List[float],
                      growth_rates: List[float]) -> List[TrendClassification]:
        """
        Classify multiple trends.

        Args:
            volumes: List of current volumes
            baseline_volumes: List of baseline volumes
            growth_rates: List of growth rates

        Returns:
            List of TrendClassification
        """
        classifications = []

        for vol, baseline, growth in zip(volumes, baseline_volumes, growth_rates):
            classification = self.classify(vol, baseline, growth)
            classifications.append(classification)

        return classifications


def classify_trend(volume: float,
                  baseline_volume: float,
                  growth_rate: float) -> str:
    """
    Convenience function for quick trend classification.

    Args:
        volume: Current volume
        baseline_volume: Baseline volume
        growth_rate: Growth rate (as ratio)

    Returns:
        Category name as string
    """
    classifier = EmergingTrendClassifier()
    result = classifier.classify(volume, baseline_volume, growth_rate)
    return result.category.value


# Testing
if __name__ == "__main__":
    print("Trend Classification Tests:\n")

    classifier = EmergingTrendClassifier()

    test_cases = [
        ("Emerging Trend", 25, 8, 2.12, [0.5, 0.8, 2.12]),
        ("Viral Trend", 850, 100, 7.5, [1.2, 3.5, 7.5]),
        ("Rising Trend", 180, 120, 0.50, [0.2, 0.3, 0.5]),
        ("Stable Trend", 500, 490, 0.02, [0.01, -0.01, 0.02]),
        ("Declining Trend", 80, 150, -0.47, [-0.2, -0.3, -0.47]),
    ]

    for name, volume, baseline, growth, history in test_cases:
        result = classifier.classify(volume, baseline, growth, history)

        print(f"{name}:")
        print(f"  Category: {result.category.value}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Volume: {volume} (baseline: {baseline})")
        print(f"  Growth: {growth:.1%}")
        print(f"  Criteria:")
        for criterion in result.criteria_met:
            print(f"    • {criterion}")
        if result.early_warning_score is not None:
            print(f"  Early Warning Score: {result.early_warning_score:.3f}")
        print()

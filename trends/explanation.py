"""
Explainable Trend Detection

Generates human-readable explanations for trend classifications.
Shows which measurable features contributed to the classification.

This is critical for research transparency and user trust.
"""

from typing import Dict, List
from .trend_score import TrendScore
from .emerging import TrendClassification, TrendCategory


class TrendExplainer:
    """
    Generate explanations for trend classifications.

    Explanations are based on actual measured features,
    not LLM-generated speculation.
    """

    def explain_trend_score(self, result: TrendScore) -> str:
        """
        Explain how composite trend score was computed.

        Args:
            result: TrendScore with component breakdown

        Returns:
            Human-readable explanation string
        """
        lines = [
            f"Trend Score: {result.score:.3f}",
            "",
            "Component Breakdown:",
            f"  • Volume:     {result.volume_component:.3f} (weight: α)",
            f"  • Growth:     {result.growth_component:.3f} (weight: β)",
            f"  • Engagement: {result.engagement_component:.3f} (weight: γ)",
            f"  • Novelty:    {result.novelty_component:.3f} (weight: δ)",
            "",
            "Underlying Features:",
            f"  • Post Volume: {result.features.volume:.0f} (normalized: {result.features.volume_normalized:.2f})",
            f"  • Growth Rate: {result.features.growth_rate:+.1f}%",
            f"  • Engagement: {result.features.engagement:.0f} (normalized: {result.features.engagement_normalized:.2f})",
            f"  • Days Since First: {result.features.days_since_first}",
            f"  • Novelty Score: {result.features.novelty_score:.3f}",
        ]

        return "\n".join(lines)

    def explain_classification(self, classification: TrendClassification) -> str:
        """
        Explain trend category classification.

        Args:
            classification: TrendClassification result

        Returns:
            Human-readable explanation
        """
        lines = [
            f"Classification: {classification.category.value}",
            f"Confidence: {classification.confidence:.2f}",
            "",
            "Reasons:",
        ]

        for criterion in classification.criteria_met:
            lines.append(f"  • {criterion}")

        lines.extend([
            "",
            "Metrics:",
            f"  • Current Volume: {classification.volume:.0f}",
            f"  • Baseline Volume: {classification.baseline_volume:.0f}",
            f"  • Growth Rate: {classification.growth_rate:+.1%}",
        ])

        if classification.consecutive_growth_periods > 0:
            lines.append(f"  • Consecutive Growth Periods: {classification.consecutive_growth_periods}")

        if classification.early_warning_score is not None:
            lines.extend([
                "",
                f"Early Warning Score: {classification.early_warning_score:.3f}",
                "(Higher score = more likely to become major trend)"
            ])

        return "\n".join(lines)

    def explain_trend_full(self,
                          topic_label: str,
                          score_result: TrendScore,
                          classification: TrendClassification,
                          representative_posts: List[str] = None) -> str:
        """
        Generate complete explanation combining score and classification.

        Args:
            topic_label: Topic name
            score_result: TrendScore result
            classification: TrendClassification result
            representative_posts: Optional sample posts

        Returns:
            Complete explanation string
        """
        lines = [
            f"═══════════════════════════════════════════════",
            f"Topic: {topic_label}",
            f"═══════════════════════════════════════════════",
            "",
            f"Category: {classification.category.value}",
            f"Trend Score: {score_result.score:.3f}",
            f"Confidence: {classification.confidence:.2f}",
            "",
            "Why is this trending?",
            ""
        ]

        # Add key reasons
        reasons = []

        # Volume change
        if classification.growth_rate > 50:
            reasons.append(f"• Post volume increased by {classification.growth_rate:+.0f}%")
        elif classification.growth_rate < -30:
            reasons.append(f"• Post volume decreased by {abs(classification.growth_rate):.0f}%")

        # Engagement
        if score_result.features.engagement_growth > 50:
            reasons.append(f"• Engagement increased by {score_result.features.engagement_growth:+.0f}%")

        # Novelty
        if score_result.features.novelty_score > 0.7:
            reasons.append(f"• Topic is relatively new (novelty: {score_result.features.novelty_score:.2f})")

        # Velocity
        if score_result.features.velocity > 20:
            reasons.append(f"• High growth velocity ({score_result.features.velocity:.1f}% avg)")

        # Cross-source presence
        if score_result.features.cross_source_presence > 3:
            reasons.append(f"• Appeared across {score_result.features.cross_source_presence} sources")

        # Unique authors
        if score_result.features.unique_authors > 20:
            reasons.append(f"• {score_result.features.unique_authors} unique authors")

        if reasons:
            lines.extend(reasons)
        else:
            lines.append("• No significant changes detected")

        lines.extend([
            "",
            "Detailed Metrics:",
            f"  Volume: {classification.volume:.0f} (baseline: {classification.baseline_volume:.0f})",
            f"  Growth: {classification.growth_rate:+.1%}",
            f"  Engagement: {score_result.features.engagement:.0f}",
            f"  Velocity: {score_result.features.velocity:.1f}%",
            f"  Acceleration: {score_result.features.acceleration:+.1f}%",
            f"  Days Active: {score_result.features.days_since_first}",
        ])

        if classification.early_warning_score is not None:
            lines.extend([
                "",
                f"⚠️  Early Warning Score: {classification.early_warning_score:.3f}",
                "   (Likely to continue growing)"
            ])

        # Add representative posts if available
        if representative_posts:
            lines.extend([
                "",
                "Representative Posts:",
                ""
            ])
            for i, post in enumerate(representative_posts[:3], 1):
                preview = post[:100] + "..." if len(post) > 100 else post
                lines.append(f"  {i}. {preview}")

        return "\n".join(lines)

    def explain_comparison(self,
                          baseline_score: float,
                          proposed_score: float,
                          topic_label: str) -> str:
        """
        Explain why proposed method scores differently than baseline.

        Args:
            baseline_score: Baseline method score
            proposed_score: Proposed method score
            topic_label: Topic name

        Returns:
            Comparison explanation
        """
        difference = proposed_score - baseline_score
        percent_change = (difference / baseline_score * 100) if baseline_score > 0 else 0

        lines = [
            f"Topic: {topic_label}",
            "",
            f"Baseline Score (frequency only): {baseline_score:.3f}",
            f"Proposed Score (composite):      {proposed_score:.3f}",
            f"Difference:                      {difference:+.3f} ({percent_change:+.1f}%)",
            "",
        ]

        if difference > 0.1:
            lines.append("The proposed method ranks this higher because it considers:")
            lines.append("  • Growth rate (not just volume)")
            lines.append("  • Engagement signals")
            lines.append("  • Topic novelty")
        elif difference < -0.1:
            lines.append("The proposed method ranks this lower because:")
            lines.append("  • High volume but low growth")
            lines.append("  • Low engagement relative to volume")
            lines.append("  • Established topic (low novelty)")
        else:
            lines.append("Both methods rank this similarly.")

        return "\n".join(lines)


def explain_trend(topic_label: str,
                 score: float,
                 category: str,
                 volume: float,
                 growth_rate: float,
                 engagement: float) -> str:
    """
    Convenience function for quick trend explanation.

    Args:
        topic_label: Topic name
        score: Trend score
        category: Trend category
        volume: Current volume
        growth_rate: Growth rate (%)
        engagement: Total engagement

    Returns:
        Explanation string
    """
    lines = [
        f"Topic '{topic_label}' classified as {category}",
        f"Trend Score: {score:.3f}",
        "",
        "Key Metrics:",
        f"  • Volume: {volume:.0f}",
        f"  • Growth: {growth_rate:+.1f}%",
        f"  • Engagement: {engagement:.0f}",
    ]

    return "\n".join(lines)


# Testing
if __name__ == "__main__":
    from datetime import datetime
    from .trend_score import TrendFeatures, TrendScorer
    from .emerging import EmergingTrendClassifier

    print("Trend Explanation Tests:\n")

    # Create sample scenario
    features = TrendFeatures(
        volume=25,
        volume_normalized=0.3,
        baseline_volume=8,
        growth_rate=173.0,
        growth_absolute=17,
        velocity=45.2,
        acceleration=12.5,
        engagement=890,
        engagement_normalized=0.42,
        engagement_growth=128.0,
        days_since_first=2,
        novelty_score=0.84,
        unique_authors=18,
        cross_source_presence=4,
        timestamp=datetime.now()
    )

    # Compute score
    scorer = TrendScorer()
    score_result = scorer.compute_score(features)
    score_result.topic_id = 1
    score_result.topic_label = "AI Coding Agents"

    # Classify
    classifier = EmergingTrendClassifier()
    classification = classifier.classify(
        volume=features.volume,
        baseline_volume=features.baseline_volume,
        growth_rate=features.growth_rate / 100,  # Convert to ratio
        growth_history=[0.5, 0.8, 1.73]
    )
    classification.topic_id = 1
    classification.topic_label = "AI Coding Agents"

    # Generate explanations
    explainer = TrendExplainer()

    print(explainer.explain_trend_full(
        "AI Coding Agents",
        score_result,
        classification,
        representative_posts=[
            "GitHub Copilot is transforming how I write code",
            "Just tried Claude Code - this is the future of development",
            "AI coding assistants are becoming essential tools"
        ]
    ))

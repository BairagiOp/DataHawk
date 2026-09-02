"""
Evaluation Metrics

Comprehensive metrics for evaluating the research system:
- Topic quality metrics
- Trend detection metrics
- Classification metrics
- Ranking metrics
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter


@dataclass
class TopicMetrics:
    """Topic discovery evaluation metrics"""
    coherence: float
    diversity: float
    coverage: float
    silhouette: float
    method: str


@dataclass
class TrendMetrics:
    """Trend detection evaluation metrics"""
    precision: float
    recall: float
    f1: float
    early_detection_rate: float
    false_positive_rate: float
    method: str


@dataclass
class ClassificationMetrics:
    """General classification metrics"""
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray
    method: str


class MetricsCalculator:
    """
    Calculate evaluation metrics for the research system.

    Supports metrics for:
    - Topic quality (RQ1)
    - Sentiment accuracy (RQ2)
    - Trend scoring (RQ3-RQ5)
    - Forecasting (RQ6)
    - Ablation studies (RQ7)
    """

    def __init__(self):
        pass

    # Topic Metrics (RQ1)

    def compute_topic_coherence(self,
                               topic_words: List[List[str]],
                               method: str = 'npmi') -> float:
        """
        Compute topic coherence score.

        Higher = more coherent topics
        (This is a simplified version; full NPMI requires corpus statistics)

        Args:
            topic_words: List of word lists per topic
            method: Coherence method

        Returns:
            Coherence score
        """
        # Simplified: measure average pairwise overlap
        if len(topic_words) == 0:
            return 0.0

        coherence_scores = []

        for words in topic_words:
            if len(words) < 2:
                coherence_scores.append(0.0)
                continue

            # Count unique words (low overlap = more diverse/incoherent)
            unique_ratio = len(set(words)) / len(words)
            coherence_scores.append(unique_ratio)

        return np.mean(coherence_scores)

    def compute_topic_diversity(self,
                                topic_words: List[List[str]],
                                top_n: int = 10) -> float:
        """
        Compute topic diversity (how different topics are).

        Higher = more diverse topics

        Args:
            topic_words: List of word lists per topic
            top_n: Consider top N words per topic

        Returns:
            Diversity score (0-1)
        """
        if len(topic_words) < 2:
            return 1.0

        # Take top N words per topic
        topic_words = [words[:top_n] for words in topic_words]

        # Count unique words across all topics
        all_words = [word for words in topic_words for word in words]
        total_words = len(all_words)
        unique_words = len(set(all_words))

        # Diversity = unique words / total words
        diversity = unique_words / total_words if total_words > 0 else 0.0

        return diversity

    # Classification Metrics (RQ2, RQ4)

    def compute_classification_metrics(self,
                                      y_true: np.ndarray,
                                      y_pred: np.ndarray,
                                      method: str = 'unknown') -> ClassificationMetrics:
        """
        Compute classification metrics.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            method: Method name

        Returns:
            ClassificationMetrics object
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have same length")

        # Accuracy
        accuracy = np.mean(y_true == y_pred)

        # Precision, Recall, F1 (binary classification)
        # Assumes positive class = 1
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Confusion matrix
        confusion = np.array([[tn, fp], [fn, tp]])

        return ClassificationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            confusion_matrix=confusion,
            method=method
        )

    # Trend Detection Metrics (RQ3-RQ5)

    def compute_trend_detection_metrics(self,
                                       y_true: np.ndarray,
                                       y_pred: np.ndarray,
                                       detection_times_true: Optional[List[int]] = None,
                                       detection_times_pred: Optional[List[int]] = None,
                                       method: str = 'unknown') -> TrendMetrics:
        """
        Compute trend detection metrics.

        Args:
            y_true: Ground truth (1=trend, 0=not trend)
            y_pred: Predictions
            detection_times_true: When trends actually started
            detection_times_pred: When we detected them
            method: Method name

        Returns:
            TrendMetrics object
        """
        # Basic classification metrics
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # Early detection rate (if timestamps provided)
        early_detection_rate = 0.0
        if detection_times_true and detection_times_pred:
            early_detections = sum(
                1 for t_true, t_pred in zip(detection_times_true, detection_times_pred)
                if t_pred <= t_true
            )
            early_detection_rate = early_detections / len(detection_times_true)

        return TrendMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            early_detection_rate=early_detection_rate,
            false_positive_rate=fpr,
            method=method
        )

    # Ranking Metrics

    def compute_ndcg(self,
                    y_true: np.ndarray,
                    y_scores: np.ndarray,
                    k: int = 10) -> float:
        """
        Compute Normalized Discounted Cumulative Gain (NDCG@k).

        Measures ranking quality for trend scoring.

        Args:
            y_true: Ground truth relevance scores
            y_scores: Predicted scores
            k: Consider top k items

        Returns:
            NDCG@k score (0-1)
        """
        if len(y_true) != len(y_scores):
            raise ValueError("Arrays must have same length")

        if len(y_true) == 0:
            return 0.0

        # Sort by predicted scores
        order = np.argsort(y_scores)[::-1][:k]
        y_true_sorted = y_true[order]

        # DCG (Discounted Cumulative Gain)
        dcg = np.sum(
            (2 ** y_true_sorted - 1) / np.log2(np.arange(2, len(y_true_sorted) + 2))
        )

        # IDCG (Ideal DCG - sort by true relevance)
        ideal_order = np.argsort(y_true)[::-1][:k]
        y_true_ideal = y_true[ideal_order]
        idcg = np.sum(
            (2 ** y_true_ideal - 1) / np.log2(np.arange(2, len(y_true_ideal) + 2))
        )

        # NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0

        return ndcg

    def compute_spearman_correlation(self,
                                    y_true: np.ndarray,
                                    y_pred: np.ndarray) -> float:
        """
        Compute Spearman's rank correlation.

        Measures how well predicted rankings match true rankings.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            Correlation coefficient (-1 to 1)
        """
        if len(y_true) != len(y_pred):
            raise ValueError("Arrays must have same length")

        if len(y_true) < 2:
            return 0.0

        # Convert to ranks
        rank_true = np.argsort(np.argsort(y_true))
        rank_pred = np.argsort(np.argsort(y_pred))

        # Pearson correlation of ranks
        correlation = np.corrcoef(rank_true, rank_pred)[0, 1]

        return correlation


# Testing
if __name__ == "__main__":
    print("Evaluation Metrics Tests:\n")

    np.random.seed(42)
    calculator = MetricsCalculator()

    # Test 1: Topic metrics
    topic_words = [
        ['machine', 'learning', 'model', 'training', 'data'],
        ['neural', 'network', 'deep', 'learning', 'ai'],
        ['database', 'query', 'sql', 'server', 'storage']
    ]

    coherence = calculator.compute_topic_coherence(topic_words)
    diversity = calculator.compute_topic_diversity(topic_words)

    print(f"1. Topic Metrics:")
    print(f"   Coherence: {coherence:.3f}")
    print(f"   Diversity: {diversity:.3f}")

    # Test 2: Classification metrics
    y_true = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 1, 1, 0, 1, 0])

    class_metrics = calculator.compute_classification_metrics(y_true, y_pred, method='Test')

    print(f"\n2. Classification Metrics:")
    print(f"   Accuracy:  {class_metrics.accuracy:.3f}")
    print(f"   Precision: {class_metrics.precision:.3f}")
    print(f"   Recall:    {class_metrics.recall:.3f}")
    print(f"   F1:        {class_metrics.f1:.3f}")

    # Test 3: Trend detection metrics
    trend_true = np.array([0, 0, 1, 1, 1, 0, 1, 1, 0, 0])
    trend_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 0])

    trend_metrics = calculator.compute_trend_detection_metrics(
        trend_true, trend_pred, method='Test'
    )

    print(f"\n3. Trend Detection Metrics:")
    print(f"   Precision: {trend_metrics.precision:.3f}")
    print(f"   Recall:    {trend_metrics.recall:.3f}")
    print(f"   F1:        {trend_metrics.f1:.3f}")
    print(f"   FPR:       {trend_metrics.false_positive_rate:.3f}")

    # Test 4: Ranking metrics
    relevance_true = np.array([3, 2, 1, 0, 3, 2, 1, 0])
    scores_pred = np.array([0.9, 0.7, 0.5, 0.1, 0.8, 0.6, 0.4, 0.2])

    ndcg = calculator.compute_ndcg(relevance_true, scores_pred, k=5)
    spearman = calculator.compute_spearman_correlation(relevance_true, scores_pred)

    print(f"\n4. Ranking Metrics:")
    print(f"   NDCG@5:   {ndcg:.3f}")
    print(f"   Spearman: {spearman:.3f}")

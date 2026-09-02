"""
Topic Coherence Evaluation

Evaluates the quality of discovered topics using coherence metrics.
This is critical for RQ1: Does semantic clustering produce better topics?
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import Counter
import itertools


class CoherenceEvaluator:
    """
    Evaluate topic coherence using multiple metrics.

    Coherence measures semantic consistency within topics.
    Higher coherence = better topic quality.
    """

    def __init__(self):
        self.word_cooccurrence = None

    def evaluate_cluster_coherence(self,
                                   cluster_texts: List[str],
                                   top_k_words: int = 10) -> Dict[str, float]:
        """
        Evaluate coherence of a single cluster.

        Args:
            cluster_texts: Texts in cluster
            top_k_words: Number of top words to use

        Returns:
            Dict with coherence metrics
        """
        if not cluster_texts or len(cluster_texts) < 2:
            return {
                'coherence': 0.0,
                'avg_similarity': 0.0,
                'keyword_overlap': 0.0
            }

        # Extract top keywords
        keywords = self._extract_top_keywords(cluster_texts, top_k_words)

        # Compute coherence metrics
        coherence = self._compute_npmi_coherence(keywords, cluster_texts)
        avg_similarity = self._compute_avg_text_similarity(cluster_texts)
        keyword_overlap = self._compute_keyword_overlap(cluster_texts)

        return {
            'coherence': coherence,
            'avg_similarity': avg_similarity,
            'keyword_overlap': keyword_overlap,
            'combined_score': (coherence + avg_similarity + keyword_overlap) / 3
        }

    def evaluate_all_clusters(self,
                             cluster_info: Dict[int, Dict],
                             embeddings: Optional[np.ndarray] = None) -> Dict:
        """
        Evaluate coherence for all clusters.

        Args:
            cluster_info: Dict from TopicClusterer.get_cluster_info()
            embeddings: Optional embeddings for similarity computation

        Returns:
            Dict with overall and per-cluster metrics
        """
        cluster_scores = {}
        all_coherences = []

        for cluster_id, info in cluster_info.items():
            if cluster_id == -1:  # Skip noise
                continue

            scores = self.evaluate_cluster_coherence(info['texts'])
            cluster_scores[cluster_id] = scores
            all_coherences.append(scores['combined_score'])

        # Overall metrics
        overall = {
            'mean_coherence': np.mean(all_coherences) if all_coherences else 0.0,
            'median_coherence': np.median(all_coherences) if all_coherences else 0.0,
            'min_coherence': np.min(all_coherences) if all_coherences else 0.0,
            'max_coherence': np.max(all_coherences) if all_coherences else 0.0,
            'std_coherence': np.std(all_coherences) if all_coherences else 0.0
        }

        return {
            'overall': overall,
            'per_cluster': cluster_scores,
            'n_clusters': len(cluster_scores)
        }

    def _extract_top_keywords(self, texts: List[str], top_k: int) -> List[str]:
        """Extract top keywords from texts"""
        import re

        all_words = []
        stop_words = {
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or',
            'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by', 'this',
            'that', 'from', 'are', 'was', 'were', 'been', 'have', 'has'
        }

        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            words = [w for w in words if len(w) > 3 and w not in stop_words]
            all_words.extend(words)

        counts = Counter(all_words)
        return [word for word, count in counts.most_common(top_k)]

    def _compute_npmi_coherence(self,
                                keywords: List[str],
                                texts: List[str]) -> float:
        """
        Compute Normalized Pointwise Mutual Information (NPMI) coherence.

        NPMI measures how often word pairs co-occur vs appearing independently.
        Range: [-1, 1] where 1 = perfect coherence
        """
        if len(keywords) < 2:
            return 0.0

        # Count word occurrences and co-occurrences
        word_counts = Counter()
        cooccurrence_counts = Counter()
        n_docs = len(texts)

        for text in texts:
            text_lower = text.lower()
            words_in_text = set()

            # Count word occurrences
            for word in keywords:
                if word in text_lower:
                    word_counts[word] += 1
                    words_in_text.add(word)

            # Count co-occurrences
            for w1, w2 in itertools.combinations(sorted(words_in_text), 2):
                cooccurrence_counts[(w1, w2)] += 1

        # Compute NPMI for each word pair
        npmi_scores = []
        for w1, w2 in itertools.combinations(keywords, 2):
            p_w1 = word_counts[w1] / n_docs if n_docs > 0 else 0
            p_w2 = word_counts[w2] / n_docs if n_docs > 0 else 0
            p_w1_w2 = cooccurrence_counts[(w1, w2)] / n_docs if n_docs > 0 else 0

            if p_w1 > 0 and p_w2 > 0 and p_w1_w2 > 0:
                pmi = np.log(p_w1_w2 / (p_w1 * p_w2))
                npmi = pmi / (-np.log(p_w1_w2))
                npmi_scores.append(npmi)

        return np.mean(npmi_scores) if npmi_scores else 0.0

    def _compute_avg_text_similarity(self, texts: List[str]) -> float:
        """
        Compute average pairwise text similarity using word overlap.

        Returns score [0, 1]
        """
        if len(texts) < 2:
            return 0.0

        # Convert texts to word sets
        word_sets = []
        for text in texts:
            words = set(text.lower().split())
            word_sets.append(words)

        # Compute pairwise Jaccard similarity
        similarities = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                if word_sets[i] and word_sets[j]:
                    intersection = len(word_sets[i] & word_sets[j])
                    union = len(word_sets[i] | word_sets[j])
                    similarity = intersection / union if union > 0 else 0.0
                    similarities.append(similarity)

        return np.mean(similarities) if similarities else 0.0

    def _compute_keyword_overlap(self, texts: List[str]) -> float:
        """
        Compute how many texts share common keywords.

        Returns score [0, 1]
        """
        if len(texts) < 2:
            return 0.0

        # Extract keywords from each text
        all_keywords = []
        for text in texts:
            keywords = self._extract_top_keywords([text], 5)
            all_keywords.append(set(keywords))

        # Compute average overlap
        overlaps = []
        for i in range(len(all_keywords)):
            for j in range(i + 1, len(all_keywords)):
                if all_keywords[i] and all_keywords[j]:
                    intersection = len(all_keywords[i] & all_keywords[j])
                    union = len(all_keywords[i] | all_keywords[j])
                    overlap = intersection / union if union > 0 else 0.0
                    overlaps.append(overlap)

        return np.mean(overlaps) if overlaps else 0.0


def compute_cluster_coherence(cluster_texts: List[str]) -> float:
    """
    Convenience function for coherence evaluation.

    Args:
        cluster_texts: Texts in cluster

    Returns:
        Coherence score [0, 1]
    """
    evaluator = CoherenceEvaluator()
    result = evaluator.evaluate_cluster_coherence(cluster_texts)
    return result['combined_score']


# Testing
if __name__ == "__main__":
    print("Topic Coherence Evaluation Tests:\n")

    # Test Case 1: Coherent cluster (similar topics)
    coherent_cluster = [
        "AI agents are transforming software development",
        "Autonomous coding assistants help developers write code faster",
        "LLM-powered development tools are becoming essential",
        "GitHub Copilot uses AI to suggest code completions",
        "Machine learning models assist in code generation"
    ]

    # Test Case 2: Incoherent cluster (random topics)
    incoherent_cluster = [
        "The weather is nice today in California",
        "Machine learning algorithms are improving rapidly",
        "Pizza is my favorite food for dinner",
        "Stock market trends show interesting patterns",
        "Cats make great pets for apartment living"
    ]

    evaluator = CoherenceEvaluator()

    print("Coherent Cluster:")
    coherent_scores = evaluator.evaluate_cluster_coherence(coherent_cluster)
    for metric, value in coherent_scores.items():
        print(f"  {metric}: {value:.3f}")
    print()

    print("Incoherent Cluster:")
    incoherent_scores = evaluator.evaluate_cluster_coherence(incoherent_cluster)
    for metric, value in incoherent_scores.items():
        print(f"  {metric}: {value:.3f}")
    print()

    print("Expected: Coherent cluster should have higher scores")

"""
Baseline Topic Discovery Methods

Implements traditional topic modeling approaches for comparison:
1. TF-IDF + K-Means clustering
2. Latent Dirichlet Allocation (LDA)

These serve as baselines for RQ1.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class TfidfKMeansBaseline:
    """
    Baseline: TF-IDF vectorization + K-Means clustering
    """

    def __init__(self, n_clusters: int = 10, max_features: int = 1000):
        """
        Args:
            n_clusters: Number of topics
            max_features: Max features for TF-IDF
        """
        self.n_clusters = n_clusters
        self.max_features = max_features
        self.vectorizer = None
        self.kmeans = None
        self.feature_names = None

    def fit_predict(self, texts: List[str]) -> Tuple[np.ndarray, Dict]:
        """
        Fit model and predict clusters.

        Args:
            texts: List of texts

        Returns:
            (cluster_labels, topic_info)
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans

        # TF-IDF vectorization
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )

        tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()

        # K-Means clustering
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10
        )

        labels = self.kmeans.fit_predict(tfidf_matrix)

        # Extract top terms per cluster
        topic_info = self._get_topic_terms()

        return labels, topic_info

    def _get_topic_terms(self, n_terms: int = 10) -> Dict[int, List[str]]:
        """Get top terms for each cluster"""
        if self.kmeans is None:
            return {}

        topic_terms = {}
        cluster_centers = self.kmeans.cluster_centers_

        for cluster_id in range(self.n_clusters):
            # Get indices of top terms
            top_indices = cluster_centers[cluster_id].argsort()[-n_terms:][::-1]
            top_terms = [self.feature_names[i] for i in top_indices]
            topic_terms[cluster_id] = top_terms

        return topic_terms


class LDABaseline:
    """
    Baseline: Latent Dirichlet Allocation (LDA)
    """

    def __init__(self,
                 n_topics: int = 10,
                 max_features: int = 1000,
                 alpha: float = 0.1,
                 beta: float = 0.01):
        """
        Args:
            n_topics: Number of topics
            max_features: Max vocabulary size
            alpha: Document-topic density (lower = fewer topics per doc)
            beta: Topic-word density (lower = fewer words per topic)
        """
        self.n_topics = n_topics
        self.max_features = max_features
        self.alpha = alpha
        self.beta = beta
        self.vectorizer = None
        self.lda = None
        self.feature_names = None

    def fit_predict(self, texts: List[str]) -> Tuple[np.ndarray, Dict]:
        """
        Fit LDA and assign documents to dominant topics.

        Args:
            texts: List of texts

        Returns:
            (dominant_topic_labels, topic_info)
        """
        try:
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.decomposition import LatentDirichletAllocation

            # Count vectorization (LDA uses raw counts, not TF-IDF)
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                stop_words='english',
                min_df=2,
                max_df=0.95
            )

            doc_term_matrix = self.vectorizer.fit_transform(texts)
            self.feature_names = self.vectorizer.get_feature_names_out()

            # LDA
            self.lda = LatentDirichletAllocation(
                n_components=self.n_topics,
                doc_topic_prior=self.alpha,
                topic_word_prior=self.beta,
                random_state=42,
                max_iter=20,
                learning_method='batch'
            )

            # Fit and get document-topic distributions
            doc_topic_dist = self.lda.fit_transform(doc_term_matrix)

            # Assign each document to its dominant topic
            labels = np.argmax(doc_topic_dist, axis=1)

            # Extract top terms per topic
            topic_info = self._get_topic_terms()

            return labels, topic_info

        except ImportError:
            print("Warning: scikit-learn not fully available")
            # Return random assignments as fallback
            labels = np.random.randint(0, self.n_topics, len(texts))
            return labels, {}

    def _get_topic_terms(self, n_terms: int = 10) -> Dict[int, List[str]]:
        """Get top terms for each topic"""
        if self.lda is None:
            return {}

        topic_terms = {}

        for topic_id, topic_dist in enumerate(self.lda.components_):
            # Get indices of top terms
            top_indices = topic_dist.argsort()[-n_terms:][::-1]
            top_terms = [self.feature_names[i] for i in top_indices]
            topic_terms[topic_id] = top_terms

        return topic_terms

    def get_perplexity(self, texts: List[str]) -> float:
        """Compute perplexity (lower is better)"""
        if self.lda is None or self.vectorizer is None:
            return float('inf')

        doc_term_matrix = self.vectorizer.transform(texts)
        return self.lda.perplexity(doc_term_matrix)


def run_baseline_comparison(texts: List[str],
                           n_topics: int = 10) -> Dict:
    """
    Run all baseline methods and compare.

    Args:
        texts: List of texts
        n_topics: Number of topics

    Returns:
        Dict with results from all methods
    """
    results = {}

    # TF-IDF + K-Means
    print("Running TF-IDF + K-Means...")
    tfidf_kmeans = TfidfKMeansBaseline(n_clusters=n_topics)
    tfidf_labels, tfidf_topics = tfidf_kmeans.fit_predict(texts)

    results['tfidf_kmeans'] = {
        'labels': tfidf_labels,
        'topics': tfidf_topics,
        'method': 'TF-IDF + K-Means'
    }

    # LDA
    print("Running LDA...")
    lda = LDABaseline(n_topics=n_topics)
    lda_labels, lda_topics = lda.fit_predict(texts)

    results['lda'] = {
        'labels': lda_labels,
        'topics': lda_topics,
        'method': 'LDA',
        'perplexity': lda.get_perplexity(texts)
    }

    return results


# Testing
if __name__ == "__main__":
    print("Baseline Topic Discovery Tests:\n")

    # Sample texts
    sample_texts = [
        # Topic 1: AI/ML
        "Machine learning models are improving rapidly",
        "Deep learning neural networks achieve state of the art",
        "AI agents are transforming software development",
        "Natural language processing has made huge progress",
        "Computer vision models can now recognize objects accurately",

        # Topic 2: Climate
        "Climate change is accelerating faster than predicted",
        "Global warming is causing extreme weather events",
        "Renewable energy adoption is increasing worldwide",
        "Carbon emissions need to be reduced urgently",
        "Sea levels are rising due to melting ice caps",

        # Topic 3: Technology
        "New smartphone features include advanced cameras",
        "5G networks are rolling out across major cities",
        "Cloud computing services are becoming essential",
        "Cybersecurity threats are evolving constantly",
        "Internet of Things devices are everywhere now"
    ] * 10  # Repeat for better clustering

    # Run baselines
    results = run_baseline_comparison(sample_texts, n_topics=3)

    print("\nResults:\n")

    # TF-IDF + K-Means
    print("TF-IDF + K-Means Topics:")
    for topic_id, terms in results['tfidf_kmeans']['topics'].items():
        print(f"  Topic {topic_id}: {', '.join(terms[:5])}")
    print()

    # LDA
    print("LDA Topics:")
    for topic_id, terms in results['lda']['topics'].items():
        print(f"  Topic {topic_id}: {', '.join(terms[:5])}")
    print(f"  Perplexity: {results['lda']['perplexity']:.2f}")
    print()

    # Cluster sizes
    print("Cluster Sizes:")
    for method_name, method_results in results.items():
        labels = method_results['labels']
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  {method_results['method']}: {dict(zip(unique, counts))}")

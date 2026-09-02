"""
Topic Clustering using Semantic Embeddings

Implements:
- HDBSCAN (Hierarchical Density-Based Spatial Clustering)
- K-Means clustering
- UMAP dimensionality reduction
- Cluster quality evaluation

This is a KEY RESEARCH COMPONENT for RQ1.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ClusterResult:
    """
    Result of topic clustering.
    """
    cluster_labels: np.ndarray  # Cluster ID for each post (-1 = noise)
    n_clusters: int  # Number of clusters found
    noise_count: int  # Number of noise points
    silhouette_score: float  # Cluster separation metric [-1, 1]

    # Cluster metadata
    cluster_sizes: Dict[int, int]  # cluster_id -> count
    cluster_centers: Optional[np.ndarray] = None  # For K-Means only

    # Method used
    method: str = "hdbscan"

    # Reduced embeddings (for visualization)
    reduced_embeddings: Optional[np.ndarray] = None


class TopicClusterer:
    """
    Cluster posts into semantic topics using embeddings.

    Supports multiple clustering algorithms for comparison.
    """

    def __init__(self,
                 method: str = 'hdbscan',
                 n_clusters: Optional[int] = None,
                 min_cluster_size: int = 50,
                 min_samples: int = 10,
                 use_umap: bool = True,
                 umap_n_components: int = 10):
        """
        Args:
            method: 'hdbscan' or 'kmeans'
            n_clusters: Number of clusters (for K-Means)
            min_cluster_size: Min size for HDBSCAN
            min_samples: Min samples for HDBSCAN
            use_umap: Whether to use UMAP dimensionality reduction
            umap_n_components: UMAP dimensions
        """
        self.method = method
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.use_umap = use_umap
        self.umap_n_components = umap_n_components

        self.clusterer = None
        self.umap_reducer = None

    def fit_predict(self, embeddings: np.ndarray) -> ClusterResult:
        """
        Cluster embeddings into topics.

        Args:
            embeddings: Array of embeddings (n_posts, embedding_dim)

        Returns:
            ClusterResult with labels and metadata
        """
        if embeddings.shape[0] == 0:
            raise ValueError("No embeddings provided")

        # Optional: Dimensionality reduction with UMAP
        reduced_embeddings = None
        if self.use_umap and embeddings.shape[1] > self.umap_n_components:
            reduced_embeddings = self._reduce_dimensions(embeddings)
            clustering_input = reduced_embeddings
        else:
            clustering_input = embeddings

        # Cluster
        if self.method == 'hdbscan':
            result = self._cluster_hdbscan(clustering_input)
        elif self.method == 'kmeans':
            result = self._cluster_kmeans(clustering_input)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Add reduced embeddings for visualization
        result.reduced_embeddings = reduced_embeddings

        return result

    def _reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce dimensionality using UMAP"""
        try:
            import umap

            self.umap_reducer = umap.UMAP(
                n_components=self.umap_n_components,
                n_neighbors=15,
                min_dist=0.0,
                metric='cosine',
                random_state=42
            )

            reduced = self.umap_reducer.fit_transform(embeddings)
            return reduced

        except ImportError:
            print("Warning: umap-learn not installed. Install with: pip install umap-learn")
            print("Continuing without dimensionality reduction.")
            return embeddings

    def _cluster_hdbscan(self, embeddings: np.ndarray) -> ClusterResult:
        """Cluster using HDBSCAN"""
        try:
            import hdbscan

            self.clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric='euclidean',
                cluster_selection_method='eom'
            )

            labels = self.clusterer.fit_predict(embeddings)

            # Count clusters and noise
            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            noise_count = np.sum(labels == -1)

            # Cluster sizes
            cluster_sizes = {}
            for label in unique_labels:
                if label != -1:
                    cluster_sizes[label] = np.sum(labels == label)

            # Silhouette score (excluding noise)
            if n_clusters > 1:
                non_noise_mask = labels != -1
                if np.sum(non_noise_mask) > 1:
                    silhouette = silhouette_score(
                        embeddings[non_noise_mask],
                        labels[non_noise_mask]
                    )
                else:
                    silhouette = 0.0
            else:
                silhouette = 0.0

            return ClusterResult(
                cluster_labels=labels,
                n_clusters=n_clusters,
                noise_count=noise_count,
                silhouette_score=silhouette,
                cluster_sizes=cluster_sizes,
                method='hdbscan'
            )

        except ImportError:
            print("Warning: hdbscan not installed. Install with: pip install hdbscan")
            print("Falling back to K-Means.")
            return self._cluster_kmeans(embeddings)

    def _cluster_kmeans(self, embeddings: np.ndarray) -> ClusterResult:
        """Cluster using K-Means"""
        if self.n_clusters is None:
            # Estimate n_clusters using elbow method
            self.n_clusters = self._estimate_n_clusters(embeddings)

        self.clusterer = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10
        )

        labels = self.clusterer.fit_predict(embeddings)

        # Cluster sizes
        unique_labels = set(labels)
        cluster_sizes = {}
        for label in unique_labels:
            cluster_sizes[label] = np.sum(labels == label)

        # Silhouette score
        if self.n_clusters > 1:
            silhouette = silhouette_score(embeddings, labels)
        else:
            silhouette = 0.0

        return ClusterResult(
            cluster_labels=labels,
            n_clusters=self.n_clusters,
            noise_count=0,  # K-Means doesn't have noise
            silhouette_score=silhouette,
            cluster_sizes=cluster_sizes,
            cluster_centers=self.clusterer.cluster_centers_,
            method='kmeans'
        )

    def _estimate_n_clusters(self,
                            embeddings: np.ndarray,
                            max_clusters: int = 50) -> int:
        """Estimate optimal number of clusters using elbow method"""
        if embeddings.shape[0] < 100:
            return min(5, embeddings.shape[0] // 20)

        # Try different k values
        k_range = range(2, min(max_clusters, embeddings.shape[0] // 50))
        inertias = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)
            kmeans.fit(embeddings)
            inertias.append(kmeans.inertia_)

        # Find elbow point (simple heuristic)
        # Use the point where the rate of decrease slows significantly
        if len(inertias) > 2:
            diffs = np.diff(inertias)
            second_diffs = np.diff(diffs)

            # Find the point with the largest second derivative
            elbow_idx = np.argmax(second_diffs) + 2
            return list(k_range)[elbow_idx]
        else:
            return 5  # Default

    def get_cluster_info(self, result: ClusterResult, texts: List[str]) -> Dict:
        """
        Get detailed information about each cluster.

        Args:
            result: ClusterResult from fit_predict
            texts: Original texts (same order as embeddings)

        Returns:
            Dict mapping cluster_id -> cluster info
        """
        cluster_info = {}

        for cluster_id in result.cluster_sizes.keys():
            # Get posts in this cluster
            mask = result.cluster_labels == cluster_id
            cluster_texts = [texts[i] for i in range(len(texts)) if mask[i]]

            cluster_info[cluster_id] = {
                'cluster_id': cluster_id,
                'size': result.cluster_sizes[cluster_id],
                'texts': cluster_texts,
                'sample_texts': cluster_texts[:5]  # First 5 for preview
            }

        return cluster_info


def cluster_topics(embeddings: np.ndarray,
                  method: str = 'hdbscan',
                  n_clusters: Optional[int] = None) -> ClusterResult:
    """
    Convenience function for topic clustering.

    Args:
        embeddings: Semantic embeddings
        method: Clustering method
        n_clusters: Number of clusters (for K-Means)

    Returns:
        ClusterResult
    """
    clusterer = TopicClusterer(method=method, n_clusters=n_clusters)
    return clusterer.fit_predict(embeddings)


# Testing
if __name__ == "__main__":
    print("Topic Clustering Tests:\n")

    # Generate sample embeddings
    np.random.seed(42)

    # Create 3 clusters with some noise
    cluster1 = np.random.randn(100, 10) + np.array([5, 5, 0, 0, 0, 0, 0, 0, 0, 0])
    cluster2 = np.random.randn(80, 10) + np.array([0, 0, 5, 5, 0, 0, 0, 0, 0, 0])
    cluster3 = np.random.randn(60, 10) + np.array([0, 0, 0, 0, 5, 5, 0, 0, 0, 0])
    noise = np.random.randn(20, 10) * 3

    embeddings = np.vstack([cluster1, cluster2, cluster3, noise])

    # Test HDBSCAN
    print("Testing HDBSCAN:")
    clusterer_hdbscan = TopicClusterer(method='hdbscan', min_cluster_size=30)
    result_hdbscan = clusterer_hdbscan.fit_predict(embeddings)

    print(f"  Clusters found: {result_hdbscan.n_clusters}")
    print(f"  Noise points: {result_hdbscan.noise_count}")
    print(f"  Silhouette score: {result_hdbscan.silhouette_score:.3f}")
    print(f"  Cluster sizes: {result_hdbscan.cluster_sizes}")
    print()

    # Test K-Means
    print("Testing K-Means (k=3):")
    clusterer_kmeans = TopicClusterer(method='kmeans', n_clusters=3)
    result_kmeans = clusterer_kmeans.fit_predict(embeddings)

    print(f"  Clusters: {result_kmeans.n_clusters}")
    print(f"  Silhouette score: {result_kmeans.silhouette_score:.3f}")
    print(f"  Cluster sizes: {result_kmeans.cluster_sizes}")

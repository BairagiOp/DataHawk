"""
Semantic Embeddings for Social Media Text

Uses sentence-transformers for generating semantic embeddings.
Supports batch processing and caching for efficiency.
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
import hashlib
import pickle
import os


class EmbeddingModel:
    """
    Generate semantic embeddings for text using sentence-transformers.

    Features:
    - Batch processing
    - Optional caching
    - Cosine similarity computation
    - Multiple model support
    """

    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 cache_dir: Optional[str] = None,
                 batch_size: int = 32):
        """
        Args:
            model_name: Sentence transformer model name
            cache_dir: Directory for caching embeddings (optional)
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.model = None
        self.embedding_dim = None
        self._load_model()

        # Initialize cache
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def _load_model(self):
        """Load sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)

            # Get embedding dimension
            test_embedding = self.model.encode(["test"], show_progress_bar=False)
            self.embedding_dim = test_embedding.shape[1]

        except ImportError:
            print("Warning: sentence-transformers not installed.")
            print("Install with: pip install sentence-transformers")
            self.model = None

    def encode(self,
               texts: List[str],
               normalize: bool = True,
               show_progress: bool = False) -> np.ndarray:
        """
        Generate embeddings for texts.

        Args:
            texts: List of texts to encode
            normalize: Normalize embeddings to unit length
            show_progress: Show progress bar

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])

        if not self.model:
            raise RuntimeError("Sentence transformer model not available")

        # Check cache if enabled
        if self.cache_dir:
            embeddings = self._load_from_cache(texts)
            if embeddings is not None:
                return embeddings

        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )

        # Cache if enabled
        if self.cache_dir:
            self._save_to_cache(texts, embeddings)

        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """Encode single text"""
        return self.encode([text])[0]

    def compute_similarity(self,
                          embedding1: np.ndarray,
                          embedding2: np.ndarray,
                          metric: str = 'cosine') -> float:
        """
        Compute similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding
            metric: 'cosine' or 'euclidean'

        Returns:
            Similarity score
        """
        if metric == 'cosine':
            return self._cosine_similarity(embedding1, embedding2)
        elif metric == 'euclidean':
            return -np.linalg.norm(embedding1 - embedding2)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def compute_similarity_matrix(self,
                                  embeddings: np.ndarray,
                                  metric: str = 'cosine') -> np.ndarray:
        """
        Compute pairwise similarity matrix.

        Args:
            embeddings: Array of embeddings (n, dim)
            metric: Similarity metric

        Returns:
            Similarity matrix (n, n)
        """
        if metric == 'cosine':
            from sklearn.metrics.pairwise import cosine_similarity
            return cosine_similarity(embeddings)
        elif metric == 'euclidean':
            from sklearn.metrics.pairwise import euclidean_distances
            return -euclidean_distances(embeddings)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def find_similar(self,
                     query_embedding: np.ndarray,
                     embeddings: np.ndarray,
                     top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Find most similar embeddings to query.

        Args:
            query_embedding: Query embedding
            embeddings: Database of embeddings (n, dim)
            top_k: Number of results to return

        Returns:
            List of (index, similarity) tuples
        """
        # Compute similarities
        similarities = []
        for i, emb in enumerate(embeddings):
            sim = self.compute_similarity(query_embedding, emb)
            similarities.append((i, sim))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        return f"{self.model_name}_{text_hash}"

    def _load_from_cache(self, texts: List[str]) -> Optional[np.ndarray]:
        """Load embeddings from cache"""
        if not self.cache_dir:
            return None

        embeddings = []
        for text in texts:
            cache_key = self._get_cache_key(text)
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")

            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    emb = pickle.load(f)
                    embeddings.append(emb)
            else:
                # Cache miss - can't use partial cache
                return None

        return np.array(embeddings)

    def _save_to_cache(self, texts: List[str], embeddings: np.ndarray):
        """Save embeddings to cache"""
        if not self.cache_dir:
            return

        for text, emb in zip(texts, embeddings):
            cache_key = self._get_cache_key(text)
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")

            with open(cache_path, 'wb') as f:
                pickle.dump(emb, f)


class SemanticSearch:
    """
    Semantic search using embeddings.
    """

    def __init__(self, embedding_model: EmbeddingModel):
        """
        Args:
            embedding_model: EmbeddingModel instance
        """
        self.embedding_model = embedding_model
        self.corpus_texts: List[str] = []
        self.corpus_embeddings: Optional[np.ndarray] = None

    def index(self, texts: List[str]):
        """
        Index a corpus of texts.

        Args:
            texts: List of texts to index
        """
        self.corpus_texts = texts
        self.corpus_embeddings = self.embedding_model.encode(texts)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, str, float]]:
        """
        Search for similar texts.

        Args:
            query: Query text
            top_k: Number of results

        Returns:
            List of (index, text, similarity) tuples
        """
        if self.corpus_embeddings is None:
            raise RuntimeError("Corpus not indexed. Call index() first.")

        # Encode query
        query_embedding = self.embedding_model.encode_single(query)

        # Find similar
        results = self.embedding_model.find_similar(
            query_embedding,
            self.corpus_embeddings,
            top_k=top_k
        )

        # Add texts
        return [
            (idx, self.corpus_texts[idx], sim)
            for idx, sim in results
        ]


# Convenience functions
def generate_embeddings(texts: List[str],
                       model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """
    Generate embeddings for texts.

    Args:
        texts: List of texts
        model_name: Model name

    Returns:
        numpy array of embeddings
    """
    model = EmbeddingModel(model_name=model_name)
    return model.encode(texts)


def compute_similarity(text1: str, text2: str) -> float:
    """
    Compute semantic similarity between two texts.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Cosine similarity [0, 1]
    """
    model = EmbeddingModel()
    embeddings = model.encode([text1, text2])
    return model.compute_similarity(embeddings[0], embeddings[1])


# Testing
if __name__ == "__main__":
    # Test with sample texts
    test_texts = [
        "AI agents are transforming software development",
        "Autonomous coding assistants are becoming popular",
        "Machine learning models improve over time",
        "The weather is nice today",
    ]

    print("Semantic Embeddings Test:\n")

    try:
        model = EmbeddingModel()

        if model.model:
            # Generate embeddings
            print(f"Generating embeddings using {model.model_name}...")
            embeddings = model.encode(test_texts)
            print(f"Shape: {embeddings.shape}")
            print(f"Embedding dimension: {model.embedding_dim}")
            print()

            # Compute similarities
            print("Pairwise Similarities:")
            for i, text1 in enumerate(test_texts):
                for j, text2 in enumerate(test_texts):
                    if i < j:
                        sim = model.compute_similarity(embeddings[i], embeddings[j])
                        print(f"  '{text1[:30]}...' <-> '{text2[:30]}...': {sim:.3f}")
            print()

            # Test semantic search
            search = SemanticSearch(model)
            search.index(test_texts)

            query = "AI coding tools"
            print(f"Query: '{query}'")
            print("Top Results:")
            results = search.search(query, top_k=3)
            for idx, text, similarity in results:
                print(f"  {similarity:.3f}: {text}")
        else:
            print("sentence-transformers not available for testing")

    except Exception as e:
        print(f"Error: {e}")

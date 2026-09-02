"""
Duplicate and Near-Duplicate Detection for Social Media Posts

Implements:
1. Exact duplicate detection (hash-based, O(n))
2. Near-duplicate detection (embedding similarity)
3. MinHash LSH for efficient large-scale deduplication
"""

import hashlib
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import re


class DuplicateDetector:
    """
    Detect exact and near-duplicate posts.

    Features:
    - Exact duplicate detection via hashing
    - Near-duplicate detection via text similarity
    - Efficient processing for large datasets
    """

    def __init__(self,
                 similarity_threshold: float = 0.95,
                 normalize_whitespace: bool = True,
                 normalize_case: bool = True,
                 remove_urls: bool = True,
                 remove_mentions: bool = True):
        """
        Args:
            similarity_threshold: Threshold for near-duplicate (0-1)
            normalize_whitespace: Normalize whitespace before hashing
            normalize_case: Convert to lowercase before hashing
            remove_urls: Remove URLs before comparison
            remove_mentions: Remove mentions before comparison
        """
        self.similarity_threshold = similarity_threshold
        self.normalize_whitespace = normalize_whitespace
        self.normalize_case = normalize_case
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions

        self.seen_hashes: Set[str] = set()
        self.hash_to_posts: Dict[str, List] = defaultdict(list)

    def normalize_text(self, text: str) -> str:
        """Normalize text for deduplication"""
        if not text:
            return ""

        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http[s]?://\S+', '', text)

        # Remove mentions
        if self.remove_mentions:
            text = re.sub(r'@\w+', '', text)

        # Normalize whitespace
        if self.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text)

        # Normalize case
        if self.normalize_case:
            text = text.lower()

        return text.strip()

    def compute_hash(self, text: str) -> str:
        """Compute hash of normalized text"""
        normalized = self.normalize_text(text)
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()

    def is_exact_duplicate(self, text: str) -> Tuple[bool, str]:
        """
        Check if text is an exact duplicate.

        Args:
            text: Text to check

        Returns:
            (is_duplicate, hash_value)
        """
        text_hash = self.compute_hash(text)
        is_dup = text_hash in self.seen_hashes
        return (is_dup, text_hash)

    def add_to_seen(self, text: str, post_id: str = None):
        """Add text to seen set"""
        text_hash = self.compute_hash(text)
        self.seen_hashes.add(text_hash)
        if post_id:
            self.hash_to_posts[text_hash].append(post_id)

    def find_exact_duplicates(self, texts: List[str]) -> Dict[str, List[int]]:
        """
        Find all exact duplicates in a list of texts.

        Args:
            texts: List of texts to check

        Returns:
            Dictionary mapping text -> list of indices with that text
        """
        hash_to_indices = defaultdict(list)

        for idx, text in enumerate(texts):
            text_hash = self.compute_hash(text)
            hash_to_indices[text_hash].append(idx)

        # Filter to only duplicates and map hash keys to first text occurrence
        duplicates = {}
        for h, indices in hash_to_indices.items():
            if len(indices) > 1:
                # Key is the text of the first index in the duplicate list
                key = texts[indices[0]]
                duplicates[key] = indices

        return duplicates

    def find_near_duplicates_simple(self, texts: List[str]) -> List[Tuple[int, int, float]]:
        """
        Find near-duplicates using simple Jaccard similarity on word sets.

        Note: This is O(n^2) and only suitable for small datasets.
        For large datasets, use MinHash LSH instead.

        Args:
            texts: List of texts

        Returns:
            List of (idx1, idx2, similarity) tuples
        """
        near_duplicates = []
        n = len(texts)

        # Convert texts to word sets
        word_sets = []
        for text in texts:
            normalized = self.normalize_text(text)
            words = set(normalized.split())
            word_sets.append(words)

        # Compare all pairs
        for i in range(n):
            for j in range(i + 1, n):
                similarity = self._jaccard_similarity(word_sets[i], word_sets[j])
                if similarity >= self.similarity_threshold:
                    near_duplicates.append((i, j, similarity))

        return near_duplicates

    def _jaccard_similarity(self, set1: Set, set2: Set) -> float:
        """Compute Jaccard similarity between two sets"""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def filter_duplicates(self, texts: List[str], keep_first: bool = True) -> List[int]:
        """
        Get indices of texts to keep (removing duplicates).

        Args:
            texts: List of texts
            keep_first: If True, keep first occurrence; else keep last

        Returns:
            List of indices to keep
        """
        seen_hashes = set()
        keep_indices = []

        iterator = enumerate(texts) if keep_first else reversed(list(enumerate(texts)))

        for idx, text in iterator:
            text_hash = self.compute_hash(text)
            if text_hash not in seen_hashes:
                keep_indices.append(idx)
                seen_hashes.add(text_hash)

        if not keep_first:
            keep_indices.reverse()

        return sorted(keep_indices)

    def deduplicate_posts(self, posts: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Deduplicate a list of post dictionaries.

        Args:
            posts: List of post dicts with 'text' field

        Returns:
            (unique_posts, duplicate_posts)
        """
        seen_hashes = set()
        unique_posts = []
        duplicate_posts = []

        for post in posts:
            text = post.get('text', '')
            text_hash = self.compute_hash(text)

            if text_hash not in seen_hashes:
                unique_posts.append(post)
                seen_hashes.add(text_hash)
            else:
                duplicate_posts.append(post)

        return unique_posts, duplicate_posts

    def get_duplicate_stats(self, texts: List[str]) -> Dict:
        """
        Get statistics about duplicates in dataset.

        Returns:
            {
                "total": int,
                "unique": int,
                "duplicates": int,
                "duplicate_ratio": float,
                "duplicate_groups": int
            }
        """
        duplicates = self.find_exact_duplicates(texts)

        total = len(texts)
        duplicate_groups = len(duplicates)
        duplicate_count = sum(len(indices) - 1 for indices in duplicates.values())
        unique_count = total - duplicate_count

        return {
            "total": total,
            "unique": unique_count,
            "duplicates": duplicate_count,
            "duplicate_ratio": duplicate_count / total if total > 0 else 0.0,
            "duplicate_groups": duplicate_groups
        }

    def reset(self):
        """Reset seen hashes (for fresh deduplication)"""
        self.seen_hashes.clear()
        self.hash_to_posts.clear()


class NearDuplicateDetectorAdvanced:
    """
    Near-duplicate detection using text embeddings.

    Requires: sentence-transformers or similar embedding model
    Uses cosine similarity on embeddings.
    """

    def __init__(self,
                 similarity_threshold: float = 0.95,
                 model_name: str = 'all-MiniLM-L6-v2'):
        """
        Args:
            similarity_threshold: Cosine similarity threshold
            model_name: Sentence transformer model name
        """
        self.similarity_threshold = similarity_threshold
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        except ImportError:
            print("Warning: sentence-transformers not installed.")
            print("Install with: pip install sentence-transformers")
            self.model = None

    def find_near_duplicates(self, texts: List[str]) -> List[Tuple[int, int, float]]:
        """
        Find near-duplicates using embedding similarity.

        Args:
            texts: List of texts

        Returns:
            List of (idx1, idx2, similarity) tuples
        """
        if not self.model:
            raise RuntimeError("Sentence transformer model not available")

        # Generate embeddings
        embeddings = self.model.encode(texts, convert_to_numpy=True)

        # Compute pairwise cosine similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(embeddings)

        # Find pairs above threshold
        near_duplicates = []
        n = len(texts)

        for i in range(n):
            for j in range(i + 1, n):
                similarity = similarities[i][j]
                if similarity >= self.similarity_threshold:
                    near_duplicates.append((i, j, similarity))

        return near_duplicates


# Convenience functions
def remove_exact_duplicates(texts: List[str]) -> List[str]:
    """Remove exact duplicates, keeping first occurrence"""
    detector = DuplicateDetector()
    keep_indices = detector.filter_duplicates(texts, keep_first=True)
    return [texts[i] for i in keep_indices]


def count_duplicates(texts: List[str]) -> int:
    """Count number of duplicate texts"""
    detector = DuplicateDetector()
    stats = detector.get_duplicate_stats(texts)
    return stats['duplicates']


# Testing
if __name__ == "__main__":
    test_texts = [
        "This is a test post",
        "This is a test post",  # Exact duplicate
        "This   is   a   test   post",  # Near duplicate (whitespace)
        "THIS IS A TEST POST",  # Near duplicate (case)
        "This is a different post",
        "Another unique post here",
        "This is a test post",  # Another duplicate
    ]

    detector = DuplicateDetector()

    # Test exact duplicates
    print("Exact Duplicate Detection:")
    duplicates = detector.find_exact_duplicates(test_texts)
    for hash_val, indices in duplicates.items():
        print(f"  Hash {hash_val[:8]}... appears at indices: {indices}")
        print(f"    Text: {test_texts[indices[0]]}")

    # Test deduplication
    print("\nDeduplication:")
    unique_indices = detector.filter_duplicates(test_texts)
    print(f"  Original count: {len(test_texts)}")
    print(f"  Unique count: {len(unique_indices)}")
    print(f"  Kept indices: {unique_indices}")

    # Test statistics
    print("\nDuplicate Statistics:")
    stats = detector.get_duplicate_stats(test_texts)
    for key, value in stats.items():
        print(f"  {key}: {value}")

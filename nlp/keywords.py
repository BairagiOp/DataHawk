"""
Keyword and Hashtag Extraction for Social Media

Implements multiple methods:
- TF-IDF
- KeyBERT (embedding-based)
- YAKE (statistical)
- Hashtag analysis
- Co-occurrence analysis
"""

from typing import List, Dict, Tuple, Set, Optional
from collections import Counter, defaultdict
import re


class KeywordExtractor:
    """
    Extract keywords and hashtags from social media text.

    Methods:
    - TF-IDF (baseline)
    - KeyBERT (embedding-based)
    - YAKE (statistical)
    - Hashtag extraction
    """

    def __init__(self, method: str = 'tfidf'):
        """
        Args:
            method: 'tfidf', 'keybert', or 'yake'
        """
        self.method = method
        self._init_extractors()

    def _init_extractors(self):
        """Initialize keyword extraction models"""
        if self.method == 'keybert':
            try:
                from keybert import KeyBERT
                self.keybert = KeyBERT()
            except ImportError:
                print("Warning: keybert not installed. Install with: pip install keybert")
                self.keybert = None

        if self.method == 'yake':
            try:
                import yake
                self.yake_extractor = yake.KeywordExtractor()
            except ImportError:
                print("Warning: yake not installed. Install with: pip install yake")
                self.yake_extractor = None

    def extract(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Extract keywords from text.

        Args:
            text: Text to analyze
            top_k: Number of keywords to return

        Returns:
            List of (keyword, score) tuples
        """
        if not text or not isinstance(text, str):
            return []

        if self.method == 'tfidf':
            return self._extract_tfidf(text, top_k)
        elif self.method == 'keybert':
            return self._extract_keybert(text, top_k)
        elif self.method == 'yake':
            return self._extract_yake(text, top_k)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _extract_tfidf(self, text: str, top_k: int) -> List[Tuple[str, float]]:
        """Extract keywords using simple word frequency"""
        # Simple word tokenization
        words = re.findall(r'\b\w+\b', text.lower())

        # Remove short words and common stop words
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by'}
        words = [w for w in words if len(w) > 3 and w not in stop_words]

        # Count frequency
        counts = Counter(words)

        # Return top k
        return counts.most_common(top_k)

    def _extract_keybert(self, text: str, top_k: int) -> List[Tuple[str, float]]:
        """Extract keywords using KeyBERT"""
        if not self.keybert:
            raise RuntimeError("KeyBERT not available")

        keywords = self.keybert.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            top_n=top_k
        )
        return keywords

    def _extract_yake(self, text: str, top_k: int) -> List[Tuple[str, float]]:
        """Extract keywords using YAKE"""
        if not self.yake_extractor:
            raise RuntimeError("YAKE not available")

        keywords = self.yake_extractor.extract_keywords(text)
        return keywords[:top_k]

    def extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text"""
        hashtags = re.findall(r'#(\w+)', text)
        return [h.lower() for h in hashtags]

    def extract_mentions(self, text: str) -> List[str]:
        """Extract mentions from text"""
        mentions = re.findall(r'@(\w+)', text)
        return [m.lower() for m in mentions]


class HashtagAnalyzer:
    """
    Analyze hashtag usage and co-occurrence.
    """

    def __init__(self):
        self.hashtag_counts: Counter = Counter()
        self.cooccurrence: defaultdict = defaultdict(Counter)

    def add_post(self, text: str):
        """Add a post to hashtag analysis"""
        hashtags = re.findall(r'#(\w+)', text.lower())

        # Count hashtags
        self.hashtag_counts.update(hashtags)

        # Update co-occurrence
        unique_hashtags = list(set(hashtags))
        for i, h1 in enumerate(unique_hashtags):
            for h2 in unique_hashtags[i+1:]:
                self.cooccurrence[h1][h2] += 1
                self.cooccurrence[h2][h1] += 1

    def get_top_hashtags(self, n: int = 10) -> List[Tuple[str, int]]:
        """Get top N hashtags by frequency"""
        return self.hashtag_counts.most_common(n)

    def get_cooccurring(self, hashtag: str, n: int = 5) -> List[Tuple[str, int]]:
        """Get top N hashtags that co-occur with given hashtag"""
        if hashtag not in self.cooccurrence:
            return []
        return self.cooccurrence[hashtag].most_common(n)

    def get_cooccurrence_network(self, min_count: int = 2) -> Dict[str, List[str]]:
        """
        Get hashtag co-occurrence network.

        Args:
            min_count: Minimum co-occurrence count

        Returns:
            Dict mapping hashtag -> list of co-occurring hashtags
        """
        network = {}
        for h1 in self.cooccurrence:
            related = [
                h2 for h2, count in self.cooccurrence[h1].items()
                if count >= min_count
            ]
            if related:
                network[h1] = related
        return network


class KeywordAnalyzer:
    """
    Analyze keyword trends over time.
    """

    def __init__(self):
        self.keyword_history: List[Tuple[str, Set[str]]] = []  # (timestamp, keywords)

    def add_snapshot(self, timestamp: str, keywords: List[str]):
        """Add keyword snapshot at timestamp"""
        self.keyword_history.append((timestamp, set(keywords)))

    def get_growth_rate(self, keyword: str) -> List[Tuple[str, int]]:
        """Get keyword mention count over time"""
        timeline = []
        for timestamp, keywords in self.keyword_history:
            count = 1 if keyword in keywords else 0
            timeline.append((timestamp, count))
        return timeline

    def get_trending_keywords(self, window_size: int = 2) -> List[str]:
        """
        Get keywords with increasing frequency.

        Args:
            window_size: Number of snapshots to compare

        Returns:
            List of trending keywords
        """
        if len(self.keyword_history) < window_size + 1:
            return []

        # Compare recent window to previous window
        recent = set()
        for _, keywords in self.keyword_history[-window_size:]:
            recent.update(keywords)

        previous = set()
        for _, keywords in self.keyword_history[-(window_size*2):-window_size]:
            previous.update(keywords)

        # Keywords that appear more in recent window
        trending = recent - previous
        return list(trending)


# Convenience functions
def extract_keywords(text: str, method: str = 'tfidf', top_k: int = 5) -> List[str]:
    """
    Quick keyword extraction returning just the keywords.

    Args:
        text: Text to analyze
        method: Extraction method
        top_k: Number of keywords

    Returns:
        List of keywords
    """
    extractor = KeywordExtractor(method=method)
    results = extractor.extract(text, top_k=top_k)
    return [keyword for keyword, score in results]


def extract_hashtags(text: str) -> List[str]:
    """Extract hashtags from text"""
    return re.findall(r'#(\w+)', text.lower())


def count_hashtags(texts: List[str]) -> Counter:
    """Count hashtag frequencies across texts"""
    all_hashtags = []
    for text in texts:
        hashtags = extract_hashtags(text)
        all_hashtags.extend(hashtags)
    return Counter(all_hashtags)


# Testing
if __name__ == "__main__":
    # Test keyword extraction
    test_text = """
    AI agents are transforming software development. Machine learning and
    artificial intelligence are becoming more accessible. #AI #MachineLearning
    """

    print("Keyword Extraction Tests:\n")

    extractor = KeywordExtractor(method='tfidf')
    keywords = extractor.extract(test_text, top_k=5)
    print(f"Keywords: {keywords}")

    hashtags = extractor.extract_hashtags(test_text)
    print(f"Hashtags: {hashtags}")
    print()

    # Test hashtag analysis
    analyzer = HashtagAnalyzer()
    test_posts = [
        "AI and machine learning #AI #ML #Tech",
        "Deep learning tutorial #AI #DeepLearning",
        "Tech news today #Tech #News",
        "Machine learning resources #ML #Resources",
    ]

    for post in test_posts:
        analyzer.add_post(post)

    print("Hashtag Analysis:")
    print(f"Top hashtags: {analyzer.get_top_hashtags(5)}")
    print(f"Co-occurring with #AI: {analyzer.get_cooccurring('ai', 3)}")

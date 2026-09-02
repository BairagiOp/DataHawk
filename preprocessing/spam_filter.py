"""
Spam and Noise Detection for Social Media Posts

Implements heuristic-based spam detection:
- Excessive repetition
- High URL-to-text ratio
- Promotional keyword density
- Malformed content
- Frequency abuse (same author posting repeatedly)
"""

import re
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict


class SpamFilter:
    """
    Detect spam and low-quality content in social media posts.

    Uses heuristic scoring rather than claiming definitive bot detection.
    Returns spam likelihood score [0, 1].
    """

    def __init__(self,
                 spam_threshold: float = 0.6,
                 url_ratio_threshold: float = 0.5,
                 repetition_threshold: float = 0.7,
                 promotional_keywords_threshold: int = 3,
                 max_author_frequency: int = 5):
        """
        Args:
            spam_threshold: Score above which post is considered spam
            url_ratio_threshold: Max ratio of URL characters to total
            repetition_threshold: Max ratio of repeated n-grams
            promotional_keywords_threshold: Max promotional keywords
            max_author_frequency: Max posts from same author in window
        """
        self.spam_threshold = spam_threshold
        self.url_ratio_threshold = url_ratio_threshold
        self.repetition_threshold = repetition_threshold
        self.promotional_keywords_threshold = promotional_keywords_threshold
        self.max_author_frequency = max_author_frequency

        # Promotional keywords (expand as needed)
        self.promotional_keywords = {
            'buy', 'sale', 'discount', 'offer', 'deal', 'promo', 'code',
            'free', 'win', 'prize', 'contest', 'giveaway', 'click here',
            'limited time', 'act now', 'order now', 'shop now', 'subscribe',
            'follow for follow', 'f4f', 'follow back', 'followback',
            'dm for', 'link in bio', 'check out my', 'visit my'
        }

        # URL pattern
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )

        # Track author frequency
        self.author_post_counts: defaultdict = defaultdict(int)

    def compute_spam_score(self, text: str, author_id: Optional[str] = None) -> Dict:
        """
        Compute spam likelihood score for a post.

        Args:
            text: Post text
            author_id: Optional author identifier

        Returns:
            {
                "spam_score": float,  # [0, 1]
                "is_spam": bool,
                "features": {
                    "repetition_score": float,
                    "url_ratio": float,
                    "promotional_score": float,
                    "frequency_score": float,
                    "malformed_score": float
                },
                "reasons": List[str]
            }
        """
        if not text or not isinstance(text, str):
            return {
                "spam_score": 0.0,
                "is_spam": False,
                "features": {},
                "reasons": []
            }

        # Compute individual features
        repetition_score = self._compute_repetition_score(text)
        url_ratio = self._compute_url_ratio(text)
        promotional_score = self._compute_promotional_score(text)
        frequency_score = self._compute_frequency_score(author_id)
        malformed_score = self._compute_malformed_score(text)

        # Combine scores (weighted average)
        spam_score = (
            repetition_score * 0.3 +
            url_ratio * 0.2 +
            promotional_score * 0.3 +
            frequency_score * 0.1 +
            malformed_score * 0.1
        )

        # Generate reasons
        reasons = []
        if repetition_score > 0.5:
            reasons.append("High repetition")
        if url_ratio > self.url_ratio_threshold:
            reasons.append("Excessive URLs")
        if promotional_score > 0.5:
            reasons.append("Promotional content")
        if frequency_score > 0.5:
            reasons.append("High posting frequency")
        if malformed_score > 0.5:
            reasons.append("Malformed content")

        return {
            "spam_score": spam_score,
            "is_spam": spam_score > self.spam_threshold,
            "features": {
                "repetition_score": repetition_score,
                "url_ratio": url_ratio,
                "promotional_score": promotional_score,
                "frequency_score": frequency_score,
                "malformed_score": malformed_score
            },
            "reasons": reasons
        }

    def _compute_repetition_score(self, text: str) -> float:
        """
        Compute repetition score based on character and word n-grams,
        and unigram word repetition.

        Returns score [0, 1] where 1 = highly repetitive
        """
        if len(text) < 10:
            return 0.0

        # Character-level repetition
        char_bigrams = [text[i:i+2] for i in range(len(text)-1)]
        if char_bigrams:
            char_counts = Counter(char_bigrams)
            max_char_count = max(char_counts.values())
            char_repetition = max_char_count / len(char_bigrams)
        else:
            char_repetition = 0.0

        # Word-level repetition
        words = text.lower().split()
        if len(words) >= 3:
            word_bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            if word_bigrams:
                word_counts = Counter(word_bigrams)
                max_word_count = max(word_counts.values())
                word_repetition = max_word_count / len(word_bigrams)
            else:
                word_repetition = 0.0
        else:
            word_repetition = 0.0

        # Unigram repetition (fraction of duplicate word occurrences to total words)
        if words:
            word_counts = Counter(words)
            repeated_occurrences = sum(count for w, count in word_counts.items() if count > 1)
            unigram_repetition = repeated_occurrences / len(words)
        else:
            unigram_repetition = 0.0

        # Combine
        return max(char_repetition, word_repetition, unigram_repetition)

    def _compute_url_ratio(self, text: str) -> float:
        """Compute ratio of URL characters to total text"""
        urls = self.url_pattern.findall(text)
        if not urls:
            return 0.0

        url_length = sum(len(url) for url in urls)
        total_length = len(text)

        return url_length / total_length if total_length > 0 else 0.0

    def _compute_promotional_score(self, text: str) -> float:
        """Compute promotional content score"""
        text_lower = text.lower()

        # Count promotional keywords
        keyword_count = sum(
            1 for keyword in self.promotional_keywords
            if keyword in text_lower
        )

        # Normalize to [0, 1]
        # 0 keywords = 0, threshold+ keywords = 1
        score = min(keyword_count / self.promotional_keywords_threshold, 1.0)

        return score

    def _compute_frequency_score(self, author_id: Optional[str]) -> float:
        """Compute score based on posting frequency"""
        if not author_id:
            return 0.0

        # Increment count for this author
        self.author_post_counts[author_id] += 1
        count = self.author_post_counts[author_id]

        # Normalize to [0, 1]
        # 0-max_frequency posts = linear scale
        # max_frequency+ = 1.0
        score = min(count / self.max_author_frequency, 1.0)

        return score

    def _compute_malformed_score(self, text: str) -> float:
        """Compute score for malformed content"""
        if not text:
            return 1.0

        scores = []

        # Check for excessive special characters
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_ratio = special_chars / len(text)
        scores.append(min(special_ratio * 2, 1.0))

        # Check for excessive caps
        if len(text) > 10:
            caps_count = sum(1 for c in text if c.isupper())
            caps_ratio = caps_count / len([c for c in text if c.isalpha()])
            if caps_ratio > 0.7:  # More than 70% caps
                scores.append(caps_ratio)
            else:
                scores.append(0.0)

        # Check for excessive numbers
        numbers = sum(1 for c in text if c.isdigit())
        number_ratio = numbers / len(text)
        scores.append(min(number_ratio * 2, 1.0))

        # Return average of malformed indicators
        return sum(scores) / len(scores) if scores else 0.0

    def filter_spam(self, posts: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter spam from list of posts.

        Args:
            posts: List of post dicts with 'text' and optionally 'author_id_hash'

        Returns:
            (non_spam_posts, spam_posts)
        """
        non_spam = []
        spam = []

        for post in posts:
            text = post.get('text', '')
            author_id = post.get('author_id_hash')

            result = self.compute_spam_score(text, author_id)

            if result['is_spam']:
                # Add spam metadata to post
                post['spam_score'] = result['spam_score']
                post['spam_reasons'] = result['reasons']
                spam.append(post)
            else:
                non_spam.append(post)

        return non_spam, spam

    def reset_author_counts(self):
        """Reset author frequency tracking"""
        self.author_post_counts.clear()

    def get_spam_stats(self, posts: List[Dict]) -> Dict:
        """
        Get statistics about spam in dataset.

        Returns:
            {
                "total": int,
                "spam": int,
                "non_spam": int,
                "spam_ratio": float,
                "avg_spam_score": float
            }
        """
        total = len(posts)
        spam_scores = []
        spam_count = 0

        for post in posts:
            text = post.get('text', '')
            author_id = post.get('author_id_hash')
            result = self.compute_spam_score(text, author_id)

            spam_scores.append(result['spam_score'])
            if result['is_spam']:
                spam_count += 1

        return {
            "total": total,
            "spam": spam_count,
            "non_spam": total - spam_count,
            "spam_ratio": spam_count / total if total > 0 else 0.0,
            "avg_spam_score": sum(spam_scores) / len(spam_scores) if spam_scores else 0.0
        }


# Convenience functions
def is_spam(text: str, threshold: float = 0.6) -> bool:
    """Quick spam check"""
    filter = SpamFilter(spam_threshold=threshold)
    result = filter.compute_spam_score(text)
    return result['is_spam']


def compute_spam_score(text: str) -> float:
    """Get spam score for text"""
    filter = SpamFilter()
    result = filter.compute_spam_score(text)
    return result['spam_score']


# Testing
if __name__ == "__main__":
    spam_filter = SpamFilter()

    test_posts = [
        "This is a normal post about AI",
        "BUY NOW!!! LIMITED TIME OFFER!!! CLICK HERE!!! http://spam.com http://more-spam.com",
        "Check out my new blog post on machine learning",
        "Follow for follow! F4F! Link in bio! DM for collabs!",
        "🔥🔥🔥 AMAZING DEAL 🔥🔥🔥 SALE SALE SALE",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",  # Repetitive
        "Normal informative content here",
    ]

    print("Spam Detection Tests:\n")
    for text in test_posts:
        result = spam_filter.compute_spam_score(text)
        print(f"Text: {text[:60]}")
        print(f"  Spam Score: {result['spam_score']:.3f}")
        print(f"  Is Spam: {result['is_spam']}")
        if result['reasons']:
            print(f"  Reasons: {', '.join(result['reasons'])}")
        print()

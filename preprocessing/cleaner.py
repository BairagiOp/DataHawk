"""
Social Media Text Cleaner

Handles cleaning of social media text while preserving meaningful information.
Extracts hashtags, mentions, URLs, and normalizes text.
"""

import re
import html
import unicodedata
from typing import Dict, List, Tuple
from urllib.parse import urlparse


class SocialMediaCleaner:
    """
    Clean social media text while preserving useful information.

    Features:
    - Extract and preserve hashtags, mentions, URLs
    - Normalize Unicode and handle emojis
    - Normalize repeated characters ("sooo" → "so")
    - Remove HTML entities and excessive whitespace
    - Maintain both original_text and clean_text for traceability
    """

    def __init__(self,
                 remove_urls: bool = False,
                 remove_mentions: bool = False,
                 remove_hashtags: bool = False,
                 normalize_emojis: bool = True,
                 normalize_repetitions: bool = True,
                 max_char_repeat: int = 3):
        """
        Args:
            remove_urls: Replace URLs with [URL] token
            remove_mentions: Replace mentions with [MENTION] token
            remove_hashtags: Replace hashtags with [HASHTAG] token
            normalize_emojis: Convert emojis to text representations
            normalize_repetitions: Normalize repeated characters
            max_char_repeat: Maximum character repetitions allowed
        """
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.normalize_emojis = normalize_emojis
        self.normalize_repetitions = normalize_repetitions
        self.max_char_repeat = max_char_repeat

        # Compile regex patterns
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.mention_pattern = re.compile(r'@[\w]+')
        self.hashtag_pattern = re.compile(r'#[\w]+')
        self.repeated_char_pattern = re.compile(r'(.)\1{' + str(max_char_repeat) + ',}')
        self.whitespace_pattern = re.compile(r'\s+')

    def clean(self, text: str) -> Dict[str, any]:
        """
        Clean social media text and extract metadata.

        Args:
            text: Raw text to clean

        Returns:
            {
                "original_text": str,
                "clean_text": str,
                "hashtags": List[str],
                "mentions": List[str],
                "urls": List[str],
                "has_emoji": bool
            }
        """
        if not text or not isinstance(text, str):
            return {
                "original_text": "",
                "clean_text": "",
                "hashtags": [],
                "mentions": [],
                "urls": [],
                "has_emoji": False
            }

        original_text = text

        # 1. Extract URLs
        urls = self.url_pattern.findall(text)
        if self.remove_urls:
            text = self.url_pattern.sub('[URL]', text)

        # 2. Extract mentions
        mentions = self.mention_pattern.findall(text)
        if self.remove_mentions:
            text = self.mention_pattern.sub('[MENTION]', text)

        # 3. Extract hashtags (before normalization to preserve exact form)
        hashtags = self.hashtag_pattern.findall(text)
        hashtags = [h[1:] for h in hashtags]  # Remove # prefix
        if self.remove_hashtags:
            text = self.hashtag_pattern.sub('[HASHTAG]', text)

        # 4. Normalize Unicode
        text = unicodedata.normalize('NFKC', text)

        # 5. Decode HTML entities
        text = html.unescape(text)

        # 6. Handle emojis
        has_emoji = self._contains_emoji(text)
        if self.normalize_emojis:
            text = self._normalize_emojis(text)

        # 7. Normalize repeated characters
        if self.normalize_repetitions:
            text = self.repeated_char_pattern.sub(r'\1' * self.max_char_repeat, text)

        # 8. Remove excessive whitespace
        text = self.whitespace_pattern.sub(' ', text)

        # 9. Strip leading/trailing whitespace
        text = text.strip()

        return {
            "original_text": original_text,
            "clean_text": text,
            "hashtags": hashtags,
            "mentions": mentions,
            "urls": urls,
            "has_emoji": has_emoji
        }

    def _contains_emoji(self, text: str) -> bool:
        """Check if text contains emoji characters"""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        return bool(emoji_pattern.search(text))

    def _normalize_emojis(self, text: str) -> str:
        """
        Normalize emojis by converting to text or removing.
        For research purposes, we'll convert common emojis to text.
        """
        # Common emoji mappings
        emoji_map = {
            '🔥': ' [FIRE] ',
            '❤️': ' [HEART] ',
            '😂': ' [LAUGHING] ',
            '👍': ' [THUMBS_UP] ',
            '👎': ' [THUMBS_DOWN] ',
            '🎉': ' [CELEBRATION] ',
            '💯': ' [HUNDRED] ',
            '✨': ' [SPARKLES] ',
            '🚀': ' [ROCKET] ',
            '⚡': ' [LIGHTNING] ',
            '💡': ' [BULB] ',
            '🤔': ' [THINKING] ',
            '😊': ' [SMILING] ',
            '😢': ' [CRYING] ',
            '😡': ' [ANGRY] ',
        }

        for emoji, replacement in emoji_map.items():
            text = text.replace(emoji, replacement)

        # Remove remaining emojis
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        text = emoji_pattern.sub('', text)

        return text

    def extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text"""
        hashtags = self.hashtag_pattern.findall(text)
        return [h[1:].lower() for h in hashtags]

    def extract_mentions(self, text: str) -> List[str]:
        """Extract mentions from text"""
        mentions = self.mention_pattern.findall(text)
        return [m[1:].lower() for m in mentions]

    def extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text"""
        return self.url_pattern.findall(text)

    def extract_domains(self, text: str) -> List[str]:
        """Extract domains from URLs in text"""
        urls = self.extract_urls(text)
        domains = []
        for url in urls:
            try:
                parsed = urlparse(url)
                domain = parsed.netloc
                if domain:
                    domains.append(domain)
            except:
                continue
        return domains


class TextNormalizer:
    """Additional text normalization utilities"""

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize all whitespace to single spaces"""
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def remove_special_chars(text: str, keep_alphanumeric_and_spaces: bool = True) -> str:
        """Remove special characters"""
        if keep_alphanumeric_and_spaces:
            return re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text

    @staticmethod
    def lowercase(text: str) -> str:
        """Convert to lowercase"""
        return text.lower()

    @staticmethod
    def remove_numbers(text: str) -> str:
        """Remove all numbers"""
        return re.sub(r'\d+', '', text)

    @staticmethod
    def normalize_repetitions(text: str, max_repeat: int = 3) -> str:
        """Normalize character repetitions"""
        pattern = re.compile(r'(.)\1{' + str(max_repeat) + ',}')
        return pattern.sub(r'\1' * max_repeat, text)


def clean_social_media_post(text: str) -> Dict[str, any]:
    """
    Convenience function for cleaning social media posts with default settings.

    Args:
        text: Raw post text

    Returns:
        Cleaned text and extracted metadata
    """
    cleaner = SocialMediaCleaner(
        remove_urls=False,  # Keep URLs for now
        remove_mentions=False,
        remove_hashtags=False,
        normalize_emojis=True,
        normalize_repetitions=True,
        max_char_repeat=3
    )
    return cleaner.clean(text)


# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_texts = [
        "AI is AMAZING!!! 🔥🔥 #ArtificialIntelligence @OpenAI https://example.com",
        "This is sooo coool!!! Check out this link: http://example.com #AI #ML",
        "😂😂😂 LMAO this is hilarious @user1 @user2",
        "Normal text without special features",
        "Multiple     spaces    and\nnewlines\n\n",
    ]

    cleaner = SocialMediaCleaner()

    for text in test_texts:
        result = cleaner.clean(text)
        print(f"\nOriginal: {result['original_text']}")
        print(f"Cleaned:  {result['clean_text']}")
        print(f"Hashtags: {result['hashtags']}")
        print(f"Mentions: {result['mentions']}")
        print(f"URLs:     {result['urls']}")
        print(f"Has Emoji: {result['has_emoji']}")

"""
Unit Tests for Preprocessing Module

Tests for cleaning, deduplication, spam filtering, and language detection.
"""

import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.cleaner import SocialMediaCleaner
from preprocessing.deduplication import DuplicateDetector
from preprocessing.spam_filter import SpamFilter
from preprocessing.language import LanguageDetector


class TestSocialMediaCleaner(unittest.TestCase):
    """Test text cleaning functionality"""

    def setUp(self):
        self.cleaner = SocialMediaCleaner()

    def test_basic_cleaning(self):
        """Test basic text cleaning"""
        text = "Check out this link! https://example.com #AI @user"
        result = self.cleaner.clean(text)

        self.assertIn('clean_text', result)
        self.assertIn('hashtags', result)
        self.assertIn('urls', result)
        self.assertEqual(len(result['hashtags']), 1)
        self.assertEqual(result['hashtags'][0], 'AI')

    def test_emoji_handling(self):
        """Test emoji normalization"""
        text = "Great work! 👍👍👍"
        result = self.cleaner.clean(text)

        self.assertTrue(result['has_emoji'])

    def test_repeated_characters(self):
        """Test repeated character normalization"""
        text = "Woooooow!!!!! Amazinggggg"
        result = self.cleaner.clean(text)

        # Should normalize repeated characters
        self.assertLess(result['clean_text'].count('o'), text.count('o'))


class TestDuplicateDetector(unittest.TestCase):
    """Test duplicate detection"""

    def setUp(self):
        self.detector = DuplicateDetector()

    def test_exact_duplicates(self):
        """Test exact duplicate detection"""
        texts = [
            "This is a test post",
            "This is another post",
            "This is a test post",  # Duplicate
            "Different content"
        ]

        duplicates = self.detector.find_exact_duplicates(texts)

        self.assertEqual(len(duplicates), 1)
        self.assertIn("This is a test post", duplicates)
        self.assertEqual(len(duplicates["This is a test post"]), 2)

    def test_near_duplicates(self):
        """Test near-duplicate detection"""
        texts = [
            "Machine learning is fascinating",
            "Machine learning is really fascinating",  # Very similar
            "Completely different topic here"
        ]

        duplicates = self.detector.find_near_duplicates_simple(texts)

        # Should find the first two as near-duplicates
        self.assertTrue(len(duplicates) > 0)


class TestSpamFilter(unittest.TestCase):
    """Test spam filtering"""

    def setUp(self):
        self.spam_filter = SpamFilter()

    def test_normal_content(self):
        """Test normal content gets low spam score"""
        text = "AI and machine learning are transforming healthcare"
        result = self.spam_filter.compute_spam_score(text)

        self.assertLess(result['spam_score'], 0.5)

    def test_repetitive_spam(self):
        """Test repetitive spam gets high score"""
        text = "BUY NOW BUY NOW BUY NOW LIMITED TIME OFFER!!!"
        result = self.spam_filter.compute_spam_score(text)

        self.assertGreater(result['spam_score'], 0.5)

    def test_url_heavy_spam(self):
        """Test URL-heavy content"""
        text = "Click here: http://spam1.com http://spam2.com http://spam3.com"
        result = self.spam_filter.compute_spam_score(text)

        self.assertGreater(result['spam_score'], 0.3)


class TestLanguageDetector(unittest.TestCase):
    """Test language detection"""

    def setUp(self):
        self.detector = LanguageDetector()

    def test_english_detection(self):
        """Test English text detection"""
        text = "Machine learning is changing the world"
        lang = self.detector.detect_language(text)

        self.assertEqual(lang, 'english')

    def test_hinglish_detection(self):
        """Test Hinglish (mixed) detection"""
        text = "Aaj main office jaa raha hoon for meeting"
        lang = self.detector.detect_language(text)

        self.assertIn(lang, ['hinglish', 'hindi', 'english'])


if __name__ == '__main__':
    unittest.main()

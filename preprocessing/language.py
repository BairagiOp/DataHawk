"""
Language Detection for Social Media Text

Supports English, Hindi, Hinglish, and other languages.
Uses langdetect library with confidence scoring.
"""

from typing import Dict, Optional, Tuple
import re


class LanguageDetector:
    """
    Detect language of social media posts.

    Supports:
    - English (en)
    - Hindi (hi)
    - Hinglish (mixed Hindi-English)
    - Other languages via langdetect
    - Unknown (detection failure)
    """

    def __init__(self, confidence_threshold: float = 0.9):
        """
        Args:
            confidence_threshold: Minimum confidence to accept detection
        """
        self.confidence_threshold = confidence_threshold
        self._langdetect_available = self._check_langdetect()

        # Devanagari script range for Hindi detection
        self.devanagari_pattern = re.compile(r'[ऀ-ॿ]')
        self.latin_pattern = re.compile(r'[a-zA-Z]')

        # Common Hindi/Hinglish words in Latin transliteration
        self.hinglish_words = {
            'aaj', 'kal', 'parso', 'maine', 'mujhe', 'mera', 'meri', 'mere',
            'tum', 'tume', 'tumhe', 'tumhara', 'tumhari', 'tumhare', 'aap', 'aapka',
            'aapki', 'aapke', 'hume', 'humara', 'humari', 'humare', 'bhi',
            'toh', 'ko', 'se', 'ka', 'ki', 'ke', 'tha', 'thi', 'the', 'ho',
            'hoon', 'raha', 'rahi', 'rahe', 'kar', 'karo', 'karna', 'karke', 'yaar',
            'kya', 'kyun', 'kab', 'kaha', 'aur', 'na', 'ne', 'hi', 'ya', 'ab',
            'aa', 'ja', 'jaa', 'gaya', 'gayi', 'gaye', 'hua', 'hue', 'hai', 'hain',
            'rha', 'rhi', 'rhe', 'kr', 'kra', 'kri',
            'kuch', 'kuchh', 'bahut', 'bohot', 'achha', 'acha', 'sahi', 'galat',
            'fir', 'phir', 'phr', 'jab', 'tab', 'sab', 'ek',
            'teen', 'chaar', 'paanch', 'chah', 'saat', 'aath', 'nau', 'das'
        }

    def _check_langdetect(self) -> bool:
        """Check if langdetect is available"""
        try:
            import langdetect
            return True
        except ImportError:
            print("Warning: langdetect not installed. Install with: pip install langdetect")
            return False

    def detect(self, text: str) -> Dict[str, any]:
        """
        Detect language of text.

        Args:
            text: Text to analyze

        Returns:
            {
                "language": str,  # en, hi, hi-en, other, unknown
                "confidence": float,
                "is_hinglish": bool,
                "script": str  # latin, devanagari, mixed, unknown
            }
        """
        if not text or not isinstance(text, str) or len(text.strip()) < 3:
            return {
                "language": "unknown",
                "confidence": 0.0,
                "is_hinglish": False,
                "script": "unknown"
            }

        # Detect script
        script_info = self._detect_script(text)

        # Check for Hinglish (mixed script)
        if script_info["has_devanagari"] and script_info["has_latin"]:
            return {
                "language": "hi-en",
                "confidence": 0.95,
                "is_hinglish": True,
                "script": "mixed"
            }

        # Devanagari-only -> Hindi
        if script_info["has_devanagari"] and not script_info["has_latin"]:
            return {
                "language": "hi",
                "confidence": 0.95,
                "is_hinglish": False,
                "script": "devanagari"
            }

        # Check for Hinglish in Latin script using vocabulary heuristics
        if script_info["has_latin"] and not script_info["has_devanagari"]:
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
            hinglish_matches = sum(1 for w in words if w in self.hinglish_words)
            if hinglish_matches >= 2:
                return {
                    "language": "hi-en",
                    "confidence": 0.90,
                    "is_hinglish": True,
                    "script": "latin"
                }

        # Use langdetect for other cases
        if self._langdetect_available:
            return self._detect_with_langdetect(text, script_info)
        else:
            # Fallback: assume English if Latin script
            if script_info["has_latin"]:
                return {
                    "language": "en",
                    "confidence": 0.5,
                    "is_hinglish": False,
                    "script": "latin"
                }
            else:
                return {
                    "language": "unknown",
                    "confidence": 0.0,
                    "is_hinglish": False,
                    "script": "unknown"
                }

    def _detect_script(self, text: str) -> Dict[str, any]:
        """Detect which scripts are present in text"""
        has_devanagari = bool(self.devanagari_pattern.search(text))
        has_latin = bool(self.latin_pattern.search(text))

        # Count characters of each type
        devanagari_count = len(self.devanagari_pattern.findall(text))
        latin_count = len(self.latin_pattern.findall(text))
        total_alpha = devanagari_count + latin_count

        devanagari_ratio = devanagari_count / total_alpha if total_alpha > 0 else 0
        latin_ratio = latin_count / total_alpha if total_alpha > 0 else 0

        return {
            "has_devanagari": has_devanagari,
            "has_latin": has_latin,
            "devanagari_count": devanagari_count,
            "latin_count": latin_count,
            "devanagari_ratio": devanagari_ratio,
            "latin_ratio": latin_ratio
        }

    def _detect_with_langdetect(self, text: str, script_info: Dict) -> Dict[str, any]:
        """Detect language using langdetect library"""
        try:
            from langdetect import detect_langs

            # Get language probabilities
            langs = detect_langs(text)

            if not langs:
                return {
                    "language": "unknown",
                    "confidence": 0.0,
                    "is_hinglish": False,
                    "script": script_info.get("script", "unknown")
                }

            # Get top language
            top_lang = langs[0]
            lang_code = top_lang.lang
            confidence = top_lang.prob

            # Map language codes
            if lang_code in ["en", "hi"]:
                return {
                    "language": lang_code,
                    "confidence": confidence,
                    "is_hinglish": False,
                    "script": "devanagari" if lang_code == "hi" else "latin"
                }
            else:
                return {
                    "language": "other",
                    "confidence": confidence,
                    "is_hinglish": False,
                    "script": "other",
                    "detected_language": lang_code
                }

        except Exception as e:
            # Detection failed
            return {
                "language": "unknown",
                "confidence": 0.0,
                "is_hinglish": False,
                "script": "unknown",
                "error": str(e)
            }

    def detect_language(self, text: str) -> str:
        """
        Detect language of text and return descriptive string.

        Returns: "english", "hindi", "hinglish", "other", or "unknown"
        """
        result = self.detect(text)
        mapping = {
            "en": "english",
            "hi": "hindi",
            "hi-en": "hinglish",
            "other": "other",
            "unknown": "unknown"
        }
        return mapping.get(result["language"], "unknown")

    def is_english(self, text: str, threshold: float = 0.8) -> bool:
        """Check if text is English with given confidence"""
        result = self.detect(text)
        return result["language"] == "en" and result["confidence"] >= threshold

    def is_hindi(self, text: str, threshold: float = 0.8) -> bool:
        """Check if text is Hindi with given confidence"""
        result = self.detect(text)
        return result["language"] == "hi" and result["confidence"] >= threshold

    def is_hinglish(self, text: str) -> bool:
        """Check if text is Hinglish (mixed Hindi-English)"""
        result = self.detect(text)
        return result["is_hinglish"]

    def filter_by_language(self, texts: list, languages: list) -> list:
        """
        Filter texts by language.

        Args:
            texts: List of texts
            languages: List of language codes to keep (e.g., ["en", "hi"])

        Returns:
            Filtered list of texts
        """
        filtered = []
        for text in texts:
            result = self.detect(text)
            if result["language"] in languages:
                filtered.append(text)
        return filtered


# Convenience functions
def detect_language(text: str) -> str:
    """
    Quick language detection returning just the language code.

    Returns: "en", "hi", "hi-en", "other", or "unknown"
    """
    detector = LanguageDetector()
    result = detector.detect(text)
    return result["language"]


def is_hinglish(text: str) -> bool:
    """Check if text is Hinglish"""
    detector = LanguageDetector()
    return detector.is_hinglish(text)


# Testing
if __name__ == "__main__":
    detector = LanguageDetector()

    test_cases = [
        "This is a test in English",
        "यह हिंदी में एक परीक्षण है",
        "This is Hinglish मिश्रित text",
        "AI agents are transforming software development",
        "मशीन लर्निंग बहुत interesting है",  # Hinglish
        "",
        "123 456 789",
    ]

    print("Language Detection Tests:\n")
    for text in test_cases:
        result = detector.detect(text)
        print(f"Text: {text[:50]}")
        print(f"  Language: {result['language']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Hinglish: {result['is_hinglish']}")
        print(f"  Script: {result.get('script', 'N/A')}")
        print()

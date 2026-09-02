"""
Emotion Classification for Social Media Posts

Implements LLM-based emotion classification into 7 categories:
- Joy
- Anger
- Sadness
- Fear
- Surprise
- Disgust
- Neutral
"""

from typing import Dict, Optional, List
from enum import Enum


class EmotionLabel(str, Enum):
    """Emotion labels"""
    JOY = "joy"
    ANGER = "anger"
    SADNESS = "sadness"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"


class EmotionClassifier:
    """
    Emotion classification using LLM.

    Classifies text into 7 emotion categories.
    """

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: LLM client for classification
        """
        self.llm_client = llm_client

    def classify(self, text: str) -> Dict[str, any]:
        """
        Classify emotion of text.

        Args:
            text: Text to classify

        Returns:
            {
                "emotion": str,  # joy/anger/sadness/fear/surprise/disgust/neutral
                "confidence": float,  # [0, 1]
                "intensities": dict  # Intensity for each emotion
            }
        """
        if not text or not isinstance(text, str):
            return {
                "emotion": EmotionLabel.NEUTRAL.value,
                "confidence": 0.0,
                "intensities": {}
            }

        if not self.llm_client:
            # Fallback: return neutral
            return {
                "emotion": EmotionLabel.NEUTRAL.value,
                "confidence": 0.0,
                "intensities": {},
                "error": "LLM client not available"
            }

        return self._classify_llm(text)

    def _classify_llm(self, text: str) -> Dict[str, any]:
        """Classify emotion using LLM"""
        prompt = f"""Classify the emotion expressed in this text.

Text: "{text}"

Return JSON with:
- emotion: one of "joy", "anger", "sadness", "fear", "surprise", "disgust", or "neutral"
- confidence: float from 0.0 to 1.0
- intensities: dict with intensity (0.0 to 1.0) for each emotion

Emotion definitions:
- Joy: happiness, excitement, satisfaction, enthusiasm
- Anger: frustration, irritation, outrage, annoyance
- Sadness: disappointment, sorrow, grief, melancholy
- Fear: worry, anxiety, concern, apprehension
- Surprise: astonishment, shock, amazement (positive or negative)
- Disgust: revulsion, distaste, disapproval, contempt
- Neutral: no clear emotion, factual, informational

Return only valid JSON.
"""

        try:
            response = self.llm_client.generate_json(prompt)

            emotion = response.get("emotion", "neutral")
            confidence = float(response.get("confidence", 0.5))
            intensities = response.get("intensities", {})

            # Validate emotion
            if emotion not in [e.value for e in EmotionLabel]:
                emotion = EmotionLabel.NEUTRAL.value

            # Clamp confidence
            confidence = max(0.0, min(1.0, confidence))

            # Validate intensities
            valid_intensities = {}
            for emotion_label in EmotionLabel:
                intensity = intensities.get(emotion_label.value, 0.0)
                valid_intensities[emotion_label.value] = max(0.0, min(1.0, float(intensity)))

            return {
                "emotion": emotion,
                "confidence": confidence,
                "intensities": valid_intensities
            }

        except Exception as e:
            # Fallback to neutral on error
            return {
                "emotion": EmotionLabel.NEUTRAL.value,
                "confidence": 0.0,
                "intensities": {},
                "error": str(e)
            }

    def classify_batch(self, texts: List[str]) -> List[Dict]:
        """Classify emotion for batch of texts"""
        return [self.classify(text) for text in texts]

    def get_dominant_emotion(self, intensities: Dict[str, float]) -> str:
        """Get dominant emotion from intensity scores"""
        if not intensities:
            return EmotionLabel.NEUTRAL.value

        return max(intensities.items(), key=lambda x: x[1])[0]


# Convenience functions
def classify_emotion(text: str, llm_client=None) -> str:
    """
    Quick emotion classification returning just the label.

    Args:
        text: Text to classify
        llm_client: LLM client

    Returns:
        Emotion label: "joy", "anger", "sadness", etc.
    """
    classifier = EmotionClassifier(llm_client=llm_client)
    result = classifier.classify(text)
    return result["emotion"]


def get_emotion_intensities(text: str, llm_client=None) -> Dict[str, float]:
    """
    Get intensity scores for all emotions.

    Returns:
        Dict mapping emotion names to intensity scores [0, 1]
    """
    classifier = EmotionClassifier(llm_client=llm_client)
    result = classifier.classify(text)
    return result.get("intensities", {})


# Testing
if __name__ == "__main__":
    print("Emotion Classification Tests:")
    print("Note: Requires LLM client for actual classification")
    print()

    test_texts = [
        "This is absolutely amazing! I'm so happy! 🎉",
        "This makes me furious! Unacceptable!",
        "I'm feeling really sad about this situation.",
        "This is terrifying. I'm genuinely worried.",
        "Wow! I did not expect that at all!",
        "This is disgusting and offensive.",
        "The meeting is scheduled for 3 PM tomorrow.",
    ]

    for text in test_texts:
        print(f"Text: {text}")
        print(f"  Expected emotion: [needs LLM for classification]")
        print()

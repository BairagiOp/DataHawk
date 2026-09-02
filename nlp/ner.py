"""
Named Entity Recognition for Social Media Posts

Extracts named entities using spaCy:
- PERSON
- ORGANIZATION (ORG)
- LOCATION (GPE, LOC)
- PRODUCT
- EVENT
- And others
"""

from typing import Dict, List, Optional, Tuple
from collections import Counter


class NamedEntityRecognizer:
    """
    Named Entity Recognition using spaCy.

    Extracts entities like persons, organizations, locations, etc.
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Args:
            model_name: spaCy model name
        """
        self.model_name = model_name
        self.nlp = None
        self._load_model()

    def _load_model(self):
        """Load spaCy model"""
        try:
            import spacy
            self.nlp = spacy.load(self.model_name)
        except ImportError:
            print("Warning: spaCy not installed.")
            print("Install with: pip install spacy")
            print(f"Download model with: python -m spacy download {self.model_name}")
        except OSError:
            print(f"Warning: spaCy model '{self.model_name}' not found.")
            print(f"Download with: python -m spacy download {self.model_name}")

    def extract(self, text: str) -> List[Dict[str, any]]:
        """
        Extract named entities from text.

        Args:
            text: Text to analyze

        Returns:
            List of entities:
            [
                {
                    "text": str,
                    "label": str,  # PERSON, ORG, GPE, etc.
                    "start": int,
                    "end": int
                }
            ]
        """
        if not text or not isinstance(text, str):
            return []

        if not self.nlp:
            return []

        try:
            doc = self.nlp(text)

            entities = []
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })

            return entities

        except Exception as e:
            print(f"Error extracting entities: {e}")
            return []

    def extract_by_type(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities grouped by type.

        Args:
            text: Text to analyze

        Returns:
            Dict mapping entity types to lists of entity texts
            {
                "PERSON": ["John", "Mary"],
                "ORG": ["OpenAI", "Google"],
                ...
            }
        """
        entities = self.extract(text)

        grouped = {}
        for entity in entities:
            label = entity["label"]
            text = entity["text"]

            if label not in grouped:
                grouped[label] = []
            grouped[label].append(text)

        return grouped

    def count_entities(self, texts: List[str]) -> Counter:
        """
        Count entity mentions across multiple texts.

        Args:
            texts: List of texts

        Returns:
            Counter of entity texts
        """
        all_entities = []

        for text in texts:
            entities = self.extract(text)
            all_entities.extend([e["text"].lower() for e in entities])

        return Counter(all_entities)

    def count_by_type(self, texts: List[str]) -> Dict[str, Counter]:
        """
        Count entities by type across multiple texts.

        Args:
            texts: List of texts

        Returns:
            Dict mapping entity types to Counters
        """
        by_type = {}

        for text in texts:
            grouped = self.extract_by_type(text)

            for label, entity_texts in grouped.items():
                if label not in by_type:
                    by_type[label] = Counter()

                for entity_text in entity_texts:
                    by_type[label][entity_text.lower()] += 1

        return by_type

    def get_top_entities(self, texts: List[str], n: int = 10) -> List[Tuple[str, int]]:
        """
        Get top N most mentioned entities.

        Args:
            texts: List of texts
            n: Number of top entities to return

        Returns:
            List of (entity, count) tuples
        """
        counts = self.count_entities(texts)
        return counts.most_common(n)


# Convenience functions
def extract_entities(text: str, model_name: str = "en_core_web_sm") -> List[Dict]:
    """
    Quick entity extraction.

    Args:
        text: Text to analyze
        model_name: spaCy model

    Returns:
        List of entity dicts
    """
    ner = NamedEntityRecognizer(model_name=model_name)
    return ner.extract(text)


def extract_persons(text: str) -> List[str]:
    """Extract person names from text"""
    ner = NamedEntityRecognizer()
    grouped = ner.extract_by_type(text)
    return grouped.get("PERSON", [])


def extract_organizations(text: str) -> List[str]:
    """Extract organization names from text"""
    ner = NamedEntityRecognizer()
    grouped = ner.extract_by_type(text)
    return grouped.get("ORG", [])


def extract_locations(text: str) -> List[str]:
    """Extract location names from text"""
    ner = NamedEntityRecognizer()
    grouped = ner.extract_by_type(text)
    locations = []
    locations.extend(grouped.get("GPE", []))  # Countries, cities, states
    locations.extend(grouped.get("LOC", []))  # Non-GPE locations
    return locations


# Testing
if __name__ == "__main__":
    ner = NamedEntityRecognizer()

    test_texts = [
        "Apple CEO Tim Cook announced new AI features in San Francisco.",
        "OpenAI released GPT-4 last year in collaboration with Microsoft.",
        "Elon Musk's Tesla is expanding production in Texas and Germany.",
        "The United Nations held a summit in New York about climate change.",
    ]

    print("Named Entity Recognition Tests:\n")

    if ner.nlp:
        for text in test_texts:
            entities = ner.extract(text)
            print(f"Text: {text}")
            print(f"  Entities:")
            for entity in entities:
                print(f"    {entity['text']} ({entity['label']})")
            print()

        # Test entity counting
        print("\nTop Entities Across All Texts:")
        top_entities = ner.get_top_entities(test_texts, n=5)
        for entity, count in top_entities:
            print(f"  {entity}: {count}")
    else:
        print("spaCy not available for testing")

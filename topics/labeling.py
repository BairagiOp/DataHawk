"""
LLM-Based Topic Labeling

Generates human-readable topic labels from clusters using LLM.
This addresses RQ5: Does LLM labeling improve interpretability?
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import re


@dataclass
class TopicLabel:
    """
    Generated topic label with metadata.
    """
    cluster_id: int
    label: str  # Human-readable topic name
    keywords: List[str]  # Top keywords in cluster
    confidence: float  # LLM confidence in label
    summary: Optional[str] = None  # Optional longer description
    representative_posts: List[str] = None  # Sample posts


class TopicLabeler:
    """
    Generate semantic topic labels using LLM.

    Compares with baseline keyword-based labeling.
    """

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: LLM client for label generation
        """
        self.llm_client = llm_client

    def label_cluster(self,
                     cluster_texts: List[str],
                     cluster_id: int,
                     keywords: Optional[List[str]] = None,
                     max_examples: int = 10) -> TopicLabel:
        """
        Generate label for a cluster.

        Args:
            cluster_texts: Texts in cluster
            cluster_id: Cluster ID
            keywords: Optional pre-extracted keywords
            max_examples: Max posts to show LLM

        Returns:
            TopicLabel
        """
        if not cluster_texts:
            return TopicLabel(
                cluster_id=cluster_id,
                label="Empty Cluster",
                keywords=[],
                confidence=0.0,
                representative_posts=[]
            )

        # Extract keywords if not provided
        if keywords is None:
            keywords = self._extract_keywords_tfidf(cluster_texts)

        # Use LLM if available
        if self.llm_client:
            return self._label_with_llm(
                cluster_texts, cluster_id, keywords, max_examples
            )
        else:
            return self._label_with_keywords(cluster_id, keywords, cluster_texts)

    def _label_with_llm(self,
                       cluster_texts: List[str],
                       cluster_id: int,
                       keywords: List[str],
                       max_examples: int) -> TopicLabel:
        """Generate label using LLM"""
        # Sample representative posts
        sample_texts = cluster_texts[:max_examples]

        # Create prompt
        prompt = f"""These social media posts belong to one topic cluster.

Posts (sample of {len(sample_texts)} from {len(cluster_texts)} total):

"""
        for i, text in enumerate(sample_texts, 1):
            preview = text[:150] + "..." if len(text) > 150 else text
            prompt += f"{i}. {preview}\n"

        prompt += f"""
Top keywords: {', '.join(keywords[:10])}

Generate a concise, descriptive topic label (max 5 words) that captures what these posts are about.

Return JSON with:
- label: Concise topic name (max 5 words)
- summary: One-sentence description of the topic
- confidence: How confident you are in this label (0.0 to 1.0)
- keywords: List of 5-7 key terms that represent this topic

Return only valid JSON.
"""

        try:
            response = self.llm_client.generate_json(prompt)

            label = response.get('label', 'Unknown Topic')
            summary = response.get('summary', '')
            confidence = float(response.get('confidence', 0.5))
            llm_keywords = response.get('keywords', keywords[:7])

            # Validate label length
            if len(label.split()) > 5:
                # Truncate to 5 words
                label = ' '.join(label.split()[:5])

            # Clamp confidence
            confidence = max(0.0, min(1.0, confidence))

            return TopicLabel(
                cluster_id=cluster_id,
                label=label,
                keywords=llm_keywords if llm_keywords else keywords[:7],
                confidence=confidence,
                summary=summary,
                representative_posts=sample_texts[:3]
            )

        except Exception as e:
            print(f"Error in LLM labeling: {e}")
            # Fallback to keyword-based
            return self._label_with_keywords(cluster_id, keywords, cluster_texts)

    def _label_with_keywords(self,
                            cluster_id: int,
                            keywords: List[str],
                            cluster_texts: List[str]) -> TopicLabel:
        """Generate label from keywords (baseline)"""
        if not keywords:
            label = f"Topic {cluster_id}"
        else:
            # Create label from top 3 keywords
            label = ', '.join(keywords[:3])
            if len(label) > 50:
                label = label[:47] + "..."

        return TopicLabel(
            cluster_id=cluster_id,
            label=label,
            keywords=keywords[:7],
            confidence=0.5,
            summary=None,
            representative_posts=cluster_texts[:3] if cluster_texts else []
        )

    def _extract_keywords_tfidf(self, texts: List[str], top_k: int = 10) -> List[str]:
        """Extract keywords using simple word frequency"""
        from collections import Counter
        import re

        # Tokenize all texts
        all_words = []
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            # Filter short words and common stop words
            words = [
                w for w in words
                if len(w) > 3 and w not in {
                    'the', 'this', 'that', 'with', 'from', 'have',
                    'what', 'when', 'where', 'which', 'their', 'there',
                    'these', 'those', 'would', 'could', 'should'
                }
            ]
            all_words.extend(words)

        # Get most common
        counts = Counter(all_words)
        keywords = [word for word, count in counts.most_common(top_k)]

        return keywords

    def label_all_clusters(self,
                          cluster_info: Dict[int, Dict],
                          llm_client=None) -> Dict[int, TopicLabel]:
        """
        Label all clusters.

        Args:
            cluster_info: Dict from TopicClusterer.get_cluster_info()
            llm_client: Optional LLM client

        Returns:
            Dict mapping cluster_id -> TopicLabel
        """
        if llm_client:
            self.llm_client = llm_client

        labels = {}
        for cluster_id, info in cluster_info.items():
            # Skip noise cluster
            if cluster_id == -1:
                continue

            label = self.label_cluster(
                cluster_texts=info['texts'],
                cluster_id=cluster_id
            )
            labels[cluster_id] = label

        return labels


def label_topics(cluster_texts: List[List[str]],
                llm_client=None) -> List[str]:
    """
    Convenience function for topic labeling.

    Args:
        cluster_texts: List of text lists (one per cluster)
        llm_client: LLM client

    Returns:
        List of topic labels
    """
    labeler = TopicLabeler(llm_client=llm_client)

    labels = []
    for i, texts in enumerate(cluster_texts):
        result = labeler.label_cluster(texts, cluster_id=i)
        labels.append(result.label)

    return labels


# Testing
if __name__ == "__main__":
    print("Topic Labeling Tests:\n")

    # Sample clusters
    cluster1_texts = [
        "AI agents are transforming software development",
        "Autonomous coding assistants like GitHub Copilot are amazing",
        "LLM-powered development tools are the future",
        "Claude Code is revolutionizing how we write software"
    ]

    cluster2_texts = [
        "Machine learning models for image recognition",
        "Deep learning advances in computer vision",
        "Neural networks for visual classification",
        "CNN architectures for image processing"
    ]

    cluster3_texts = [
        "Climate change is accelerating faster than predicted",
        "Global warming impacts on weather patterns",
        "Rising sea levels threatening coastal cities",
        "Carbon emissions need to be reduced urgently"
    ]

    # Test keyword-based labeling (baseline)
    labeler = TopicLabeler(llm_client=None)

    print("Baseline (Keyword-Based) Labels:")
    for i, texts in enumerate([cluster1_texts, cluster2_texts, cluster3_texts], 1):
        label = labeler.label_cluster(texts, cluster_id=i)
        print(f"  Cluster {i}: {label.label}")
        print(f"    Keywords: {', '.join(label.keywords[:5])}")
        print()

    print("Note: LLM-based labeling requires LLM client.")
    print("Example LLM labels:")
    print("  Cluster 1: AI Coding Agents & Tools")
    print("  Cluster 2: Deep Learning for Vision")
    print("  Cluster 3: Climate Change & Global Warming")

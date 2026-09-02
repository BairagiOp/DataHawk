"""
Sample Dataset Generator

Generates synthetic social media data for testing and demonstration.

IMPORTANT: This is for development/testing only.
Real research should use ethically collected datasets.
"""

import json
import csv
from datetime import datetime, timedelta
import random
from typing import List, Dict
import os


class SampleDataGenerator:
    """
    Generate synthetic social media posts for testing.

    Creates realistic-looking data with:
    - Multiple topics
    - Trending patterns
    - Various engagement levels
    - Temporal dynamics
    """

    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)

        # Sample topics and keywords
        self.topics = {
            'AI_ML': [
                'machine learning', 'neural networks', 'deep learning', 'AI agents',
                'transformers', 'GPT', 'LLM', 'artificial intelligence', 'model training'
            ],
            'CLIMATE': [
                'climate change', 'global warming', 'renewable energy', 'carbon emissions',
                'sustainability', 'solar power', 'electric vehicles', 'green technology'
            ],
            'TECH': [
                'software development', 'programming', 'coding', 'web development',
                'mobile apps', 'cloud computing', 'DevOps', 'API'
            ],
            'HEALTH': [
                'mental health', 'fitness', 'nutrition', 'healthcare', 'wellness',
                'exercise', 'meditation', 'healthy lifestyle'
            ],
            'ECONOMY': [
                'stock market', 'cryptocurrency', 'bitcoin', 'investing', 'economy',
                'inflation', 'GDP', 'financial markets'
            ]
        }

        # Sentiment templates
        self.positive_templates = [
            "Excited about {topic}! This is game-changing.",
            "Amazing progress in {topic}. The future looks bright!",
            "{topic} is revolutionizing the industry. Love to see it!",
            "Just discovered {topic} and I'm impressed. Great work!",
        ]

        self.negative_templates = [
            "Concerned about {topic}. We need better solutions.",
            "{topic} still has major issues. Disappointed.",
            "Problems with {topic} continue. Not good enough.",
            "Struggling with {topic}. This needs improvement.",
        ]

        self.neutral_templates = [
            "Interesting developments in {topic}.",
            "Learning more about {topic}. Still forming opinions.",
            "{topic} continues to evolve. Watching closely.",
            "Analysis of {topic} trends shows mixed results.",
        ]

    def generate_post(self,
                     topic: str,
                     timestamp: datetime,
                     sentiment: str = 'neutral',
                     engagement_level: str = 'medium') -> Dict:
        """
        Generate a single post.

        Args:
            topic: Topic category
            timestamp: Post timestamp
            sentiment: 'positive', 'negative', or 'neutral'
            engagement_level: 'low', 'medium', 'high', 'viral'

        Returns:
            Post dict
        """
        # Select keywords from topic
        keywords = self.topics.get(topic, ['general topic'])
        topic_keyword = random.choice(keywords)

        # Generate text based on sentiment
        if sentiment == 'positive':
            template = random.choice(self.positive_templates)
        elif sentiment == 'negative':
            template = random.choice(self.negative_templates)
        else:
            template = random.choice(self.neutral_templates)

        text = template.format(topic=topic_keyword)

        # Add hashtags
        hashtag = f"#{topic_keyword.replace(' ', '')}"
        text += f" {hashtag}"

        # Generate engagement based on level
        engagement_multipliers = {
            'low': (1, 10),
            'medium': (10, 100),
            'high': (100, 1000),
            'viral': (1000, 10000)
        }

        min_eng, max_eng = engagement_multipliers[engagement_level]
        likes = random.randint(min_eng, max_eng)
        comments = random.randint(min_eng // 5, max_eng // 5)
        shares = random.randint(min_eng // 10, max_eng // 10)

        post_id = f"{topic}_{timestamp.strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}"

        return {
            'post_id': post_id,
            'text': text,
            'timestamp': timestamp.isoformat(),
            'likes': likes,
            'comments': comments,
            'shares': shares,
            'platform': 'synthetic'
        }

    def generate_trending_topic(self,
                               topic: str,
                               start_date: datetime,
                               duration_days: int = 14) -> List[Dict]:
        """
        Generate a trending topic with realistic growth pattern.

        Args:
            topic: Topic category
            start_date: When trend starts
            duration_days: Trend duration

        Returns:
            List of posts showing trending pattern
        """
        posts = []

        # Trend pattern: slow start, rapid growth, plateau, decline
        for day in range(duration_days):
            current_date = start_date + timedelta(days=day)

            # Volume pattern (sigmoid-like growth)
            if day < 3:
                posts_per_day = random.randint(5, 15)  # Slow start
                engagement = 'low'
            elif day < 7:
                posts_per_day = random.randint(20, 50)  # Growth
                engagement = 'medium'
            elif day < 11:
                posts_per_day = random.randint(50, 100)  # Peak
                engagement = 'high'
            else:
                posts_per_day = random.randint(10, 30)  # Decline
                engagement = 'medium'

            # Generate posts for this day
            for _ in range(posts_per_day):
                timestamp = current_date + timedelta(
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59)
                )

                sentiment = random.choices(
                    ['positive', 'neutral', 'negative'],
                    weights=[0.6, 0.3, 0.1]
                )[0]

                post = self.generate_post(topic, timestamp, sentiment, engagement)
                posts.append(post)

        return posts

    def generate_dataset(self,
                        num_topics: int = 3,
                        posts_per_topic: int = 200,
                        start_date: datetime = None) -> List[Dict]:
        """
        Generate complete dataset with multiple topics.

        Args:
            num_topics: Number of trending topics
            posts_per_topic: Posts per topic
            start_date: Dataset start date

        Returns:
            List of all posts
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)

        all_posts = []

        # Select random topics
        selected_topics = random.sample(list(self.topics.keys()), num_topics)

        for i, topic in enumerate(selected_topics):
            topic_start = start_date + timedelta(days=i * 5)
            posts = self.generate_trending_topic(topic, topic_start)

            # Limit to requested size
            posts = posts[:posts_per_topic]
            all_posts.extend(posts)

        # Sort by timestamp
        all_posts.sort(key=lambda p: p['timestamp'])

        return all_posts

    def save_csv(self, posts: List[Dict], output_path: str):
        """Save posts to CSV file"""
        if not posts:
            print("No posts to save")
            return

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=posts[0].keys())
            writer.writeheader()
            writer.writerows(posts)

        print(f"Saved {len(posts)} posts to {output_path}")

    def save_json(self, posts: List[Dict], output_path: str):
        """Save posts to JSON file"""
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(posts, f, indent=2)

        print(f"Saved {len(posts)} posts to {output_path}")


# Main script
if __name__ == "__main__":
    print("Generating sample datasets...\n")

    generator = SampleDataGenerator(seed=42)

    # Generate small test dataset
    print("1. Small test dataset (100 posts):")
    small_dataset = generator.generate_dataset(
        num_topics=2,
        posts_per_topic=50,
        start_date=datetime(2026, 7, 1)
    )
    generator.save_csv(small_dataset, 'dataset/test_data_small.csv')
    generator.save_json(small_dataset, 'dataset/test_data_small.json')

    # Generate medium dataset
    print("\n2. Medium dataset (500 posts):")
    medium_dataset = generator.generate_dataset(
        num_topics=3,
        posts_per_topic=167,
        start_date=datetime(2026, 7, 1)
    )
    generator.save_csv(medium_dataset, 'dataset/test_data_medium.csv')

    # Generate large dataset
    print("\n3. Large dataset (1000 posts):")
    large_dataset = generator.generate_dataset(
        num_topics=5,
        posts_per_topic=200,
        start_date=datetime(2026, 7, 1)
    )
    generator.save_csv(large_dataset, 'dataset/test_data_large.csv')

    print("\n[OK] Sample datasets generated successfully!")
    print("\nDataset locations:")
    print("  - dataset/test_data_small.csv (100 posts)")
    print("  - dataset/test_data_small.json (100 posts)")
    print("  - dataset/test_data_medium.csv (500 posts)")
    print("  - dataset/test_data_large.csv (1000 posts)")

"""
CSV Data Adapter

Ingests social media data from CSV files.
Supports flexible field mapping and validation.
"""

import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.schema import SocialMediaPost, Platform, create_post_from_raw, validate_post_dict


class CSVAdapter:
    """
    Load social media posts from CSV files.

    Supports flexible field mapping and handles common CSV variations.
    """

    def __init__(self,
                 platform: Platform = Platform.CSV,
                 encoding: str = 'utf-8',
                 validate: bool = True):
        """
        Args:
            platform: Platform type to assign
            encoding: File encoding
            validate: Whether to validate posts
        """
        self.platform = platform
        self.encoding = encoding
        self.validate = validate

    def load(self, csv_path: str) -> List[SocialMediaPost]:
        """
        Load posts from CSV file.

        Args:
            csv_path: Path to CSV file

        Returns:
            List of SocialMediaPost objects
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Read CSV
        try:
            df = pd.read_csv(csv_path, encoding=self.encoding)
        except Exception as e:
            raise ValueError(f"Error reading CSV: {e}")

        if df.empty:
            return []

        # Convert to posts
        posts = []
        errors = []

        for idx, row in df.iterrows():
            try:
                # Convert row to dict
                row_dict = row.to_dict()

                # Validate if enabled
                if self.validate:
                    is_valid, error_msgs = validate_post_dict(row_dict)
                    if not is_valid:
                        errors.append(f"Row {idx}: {', '.join(error_msgs)}")
                        continue

                # Create post
                post = create_post_from_raw(row_dict, platform=self.platform)
                posts.append(post)

            except Exception as e:
                errors.append(f"Row {idx}: {str(e)}")
                continue

        # Report errors
        if errors:
            print(f"Warning: {len(errors)} rows failed to load:")
            for error in errors[:10]:  # Show first 10
                print(f"  {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")

        print(f"Successfully loaded {len(posts)} posts from {csv_path}")
        return posts

    def load_batch(self, csv_paths: List[str]) -> List[SocialMediaPost]:
        """Load posts from multiple CSV files"""
        all_posts = []
        for csv_path in csv_paths:
            posts = self.load(csv_path)
            all_posts.extend(posts)
        return all_posts

    def save(self, posts: List[SocialMediaPost], output_path: str):
        """
        Save posts to CSV file.

        Args:
            posts: List of posts
            output_path: Output CSV path
        """
        # Convert posts to dicts
        data = [post.to_dict(include_derived=True) for post in posts]

        # Create DataFrame
        df = pd.DataFrame(data)

        # Flatten nested dicts
        if 'engagement' in df.columns:
            engagement_df = pd.json_normalize(df['engagement'])
            engagement_df.columns = ['engagement_' + col for col in engagement_df.columns]
            df = pd.concat([df.drop('engagement', axis=1), engagement_df], axis=1)

        # Save
        df.to_csv(output_path, index=False, encoding=self.encoding)
        print(f"Saved {len(posts)} posts to {output_path}")


def load_csv(csv_path: str, platform: Platform = Platform.CSV) -> List[SocialMediaPost]:
    """
    Convenience function for loading CSV.

    Args:
        csv_path: Path to CSV file
        platform: Platform type

    Returns:
        List of posts
    """
    adapter = CSVAdapter(platform=platform)
    return adapter.load(csv_path)


# Testing
if __name__ == "__main__":
    print("CSV Adapter Tests:\n")

    # Create sample CSV for testing
    sample_data = {
        'post_id': ['post1', 'post2', 'post3'],
        'text': [
            'AI agents are transforming software development',
            'Machine learning models are improving rapidly',
            'Deep learning has revolutionized computer vision'
        ],
        'timestamp': ['2026-08-01', '2026-08-02', '2026-08-03'],
        'likes': [50, 30, 75],
        'comments': [10, 5, 15],
        'shares': [5, 2, 8]
    }

    # Create test CSV
    test_csv_path = 'test_data.csv'
    df = pd.DataFrame(sample_data)
    df.to_csv(test_csv_path, index=False)

    # Test loading
    adapter = CSVAdapter()
    posts = adapter.load(test_csv_path)

    print(f"Loaded {len(posts)} posts:")
    for post in posts:
        print(f"  {post.post_id}: {post.text[:50]}...")
        print(f"    Engagement: {post.engagement.total_engagement:.0f}")

    # Clean up
    os.remove(test_csv_path)

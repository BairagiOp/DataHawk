"""
JSON Data Adapter

Ingests social media data from JSON files.
Supports both single JSON objects and arrays of objects.
"""

import json
from typing import List, Dict, Optional, Union
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.schema import SocialMediaPost, Platform, create_post_from_raw, validate_post_dict


class JSONAdapter:
    """
    Load social media posts from JSON files.

    Supports:
    - Single JSON object
    - JSON array of objects
    - JSON Lines (one object per line)
    """

    def __init__(self,
                 platform: Platform = Platform.JSON,
                 validate: bool = True):
        """
        Args:
            platform: Platform type to assign
            validate: Whether to validate posts
        """
        self.platform = platform
        self.validate = validate

    def load(self, json_path: str) -> List[SocialMediaPost]:
        """
        Load posts from JSON file.

        Args:
            json_path: Path to JSON file

        Returns:
            List of SocialMediaPost objects
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        # Determine JSON format
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        # Try to parse as JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Try JSON Lines format
            return self._load_jsonl(json_path)

        # Handle different formats
        if isinstance(data, list):
            # Array of objects
            return self._process_posts(data)
        elif isinstance(data, dict):
            # Check if it's a wrapper with a 'posts' or 'data' key
            if 'posts' in data:
                return self._process_posts(data['posts'])
            elif 'data' in data:
                return self._process_posts(data['data'])
            else:
                # Single object
                return self._process_posts([data])
        else:
            raise ValueError(f"Unexpected JSON format: {type(data)}")

    def _load_jsonl(self, jsonl_path: str) -> List[SocialMediaPost]:
        """Load JSON Lines format (one object per line)"""
        posts_data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    posts_data.append(obj)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}")
                    continue

        return self._process_posts(posts_data)

    def _process_posts(self, posts_data: List[Dict]) -> List[SocialMediaPost]:
        """Process list of post dictionaries"""
        posts = []
        errors = []

        for idx, post_dict in enumerate(posts_data):
            try:
                # Validate if enabled
                if self.validate:
                    is_valid, error_msgs = validate_post_dict(post_dict)
                    if not is_valid:
                        errors.append(f"Post {idx}: {', '.join(error_msgs)}")
                        continue

                # Create post
                post = create_post_from_raw(post_dict, platform=self.platform)
                posts.append(post)

            except Exception as e:
                errors.append(f"Post {idx}: {str(e)}")
                continue

        # Report errors
        if errors:
            print(f"Warning: {len(errors)} posts failed to load:")
            for error in errors[:10]:
                print(f"  {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")

        print(f"Successfully loaded {len(posts)} posts from JSON")
        return posts

    def load_batch(self, json_paths: List[str]) -> List[SocialMediaPost]:
        """Load posts from multiple JSON files"""
        all_posts = []
        for json_path in json_paths:
            posts = self.load(json_path)
            all_posts.extend(posts)
        return all_posts

    def save(self, posts: List[SocialMediaPost], output_path: str, format: str = 'json'):
        """
        Save posts to JSON file.

        Args:
            posts: List of posts
            output_path: Output path
            format: 'json' or 'jsonl'
        """
        # Convert posts to dicts
        data = [post.to_dict(include_derived=True) for post in posts]

        if format == 'json':
            # Standard JSON array
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        elif format == 'jsonl':
            # JSON Lines
            with open(output_path, 'w', encoding='utf-8') as f:
                for post_dict in data:
                    f.write(json.dumps(post_dict, default=str) + '\n')
        else:
            raise ValueError(f"Unknown format: {format}")

        print(f"Saved {len(posts)} posts to {output_path}")


def load_json(json_path: str, platform: Platform = Platform.JSON) -> List[SocialMediaPost]:
    """
    Convenience function for loading JSON.

    Args:
        json_path: Path to JSON file
        platform: Platform type

    Returns:
        List of posts
    """
    adapter = JSONAdapter(platform=platform)
    return adapter.load(json_path)


# Testing
if __name__ == "__main__":
    print("JSON Adapter Tests:\n")

    # Create sample JSON for testing
    sample_data = [
        {
            'post_id': 'post1',
            'text': 'AI agents are transforming software development',
            'timestamp': '2026-08-01T10:00:00Z',
            'likes': 50,
            'comments': 10,
            'shares': 5
        },
        {
            'post_id': 'post2',
            'text': 'Machine learning models are improving rapidly',
            'timestamp': '2026-08-02T11:00:00Z',
            'likes': 30,
            'comments': 5,
            'shares': 2
        }
    ]

    # Test JSON array format
    test_json_path = 'test_data.json'
    with open(test_json_path, 'w') as f:
        json.dump(sample_data, f)

    adapter = JSONAdapter()
    posts = adapter.load(test_json_path)

    print(f"Loaded {len(posts)} posts:")
    for post in posts:
        print(f"  {post.post_id}: {post.text[:50]}...")

    # Test JSON Lines format
    test_jsonl_path = 'test_data.jsonl'
    with open(test_jsonl_path, 'w') as f:
        for obj in sample_data:
            f.write(json.dumps(obj) + '\n')

    posts_jsonl = adapter.load(test_jsonl_path)
    print(f"\nLoaded {len(posts_jsonl)} posts from JSON Lines format")

    # Clean up
    os.remove(test_json_path)
    os.remove(test_jsonl_path)

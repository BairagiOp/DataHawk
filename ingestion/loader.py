"""
Unified Data Loader

Automatically detects file format and loads data from multiple sources.
"""

import os
from typing import List, Union, Optional
from pathlib import Path

from .schema import SocialMediaPost, Platform
from .adapters.csv_adapter import CSVAdapter
from .adapters.json_adapter import JSONAdapter


class DataLoader:
    """
    Unified loader for social media data.

    Automatically detects format and loads data.
    """

    def __init__(self, validate: bool = True):
        """
        Args:
            validate: Whether to validate loaded posts
        """
        self.validate = validate
        self.csv_adapter = CSVAdapter(validate=validate)
        self.json_adapter = JSONAdapter(validate=validate)

    def load(self, file_path: str, platform: Optional[Platform] = None) -> List[SocialMediaPost]:
        """
        Load data from file (auto-detect format).

        Args:
            file_path: Path to data file
            platform: Optional platform type

        Returns:
            List of SocialMediaPost objects
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Detect format from extension
        ext = Path(file_path).suffix.lower()

        if ext == '.csv':
            if platform:
                self.csv_adapter.platform = platform
            return self.csv_adapter.load(file_path)

        elif ext in ['.json', '.jsonl']:
            if platform:
                self.json_adapter.platform = platform
            return self.json_adapter.load(file_path)

        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def load_directory(self,
                      directory: str,
                      recursive: bool = False,
                      extensions: Optional[List[str]] = None) -> List[SocialMediaPost]:
        """
        Load all data files from a directory.

        Args:
            directory: Directory path
            recursive: Search subdirectories
            extensions: File extensions to include (default: .csv, .json, .jsonl)

        Returns:
            List of all posts
        """
        if extensions is None:
            extensions = ['.csv', '.json', '.jsonl']

        all_posts = []

        # Find files
        if recursive:
            files = []
            for ext in extensions:
                files.extend(Path(directory).rglob(f'*{ext}'))
        else:
            files = []
            for ext in extensions:
                files.extend(Path(directory).glob(f'*{ext}'))

        # Load each file
        print(f"Found {len(files)} data files in {directory}")
        for file_path in sorted(files):
            try:
                posts = self.load(str(file_path))
                all_posts.extend(posts)
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
                continue

        print(f"\nTotal posts loaded: {len(all_posts)}")
        return all_posts

    def load_batch(self, file_paths: List[str]) -> List[SocialMediaPost]:
        """Load data from multiple files"""
        all_posts = []
        for file_path in file_paths:
            posts = self.load(file_path)
            all_posts.extend(posts)
        return all_posts


def load_data(file_path: str, platform: Optional[Platform] = None) -> List[SocialMediaPost]:
    """
    Convenience function for loading data.

    Args:
        file_path: Path to data file
        platform: Optional platform type

    Returns:
        List of posts
    """
    loader = DataLoader()
    return loader.load(file_path, platform=platform)


# Testing
if __name__ == "__main__":
    print("Unified Data Loader Tests:\n")

    # This would normally load real files
    # For testing, you would create sample CSV/JSON files

    loader = DataLoader()

    print("Example Usage:")
    print("  loader.load('data.csv')")
    print("  loader.load('data.json')")
    print("  loader.load_directory('dataset/')")
    print("  loader.load_batch(['file1.csv', 'file2.json'])")

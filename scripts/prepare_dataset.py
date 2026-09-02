#!/usr/bin/env python
"""
Dataset Preparation Script

Loads raw CSV/JSON data, validates, cleans, and prepares for experiments.

Usage:
    python scripts/prepare_dataset.py --input dataset/raw.csv --output dataset/prepared.csv
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime
from ingestion.loader import DataLoader
from preprocessing import SocialMediaCleaner, DuplicateDetector, SpamFilter, LanguageDetector


def prepare_dataset(input_path: str, output_path: str, verbose: bool = True):
    """
    Prepare dataset for experiments.

    Steps:
    1. Load data
    2. Clean text
    3. Detect duplicates
    4. Detect spam
    5. Detect language
    6. Save prepared data
    """
    print("="*70)
    print("DATASET PREPARATION")
    print("="*70)

    # Step 1: Load data
    print(f"\n[1/6] Loading data from {input_path}...")
    loader = DataLoader(validate=True)
    try:
        posts = loader.load(input_path)
    except Exception as e:
        print(f"ERROR: Failed to load data: {e}")
        return False

    if not posts:
        print("ERROR: No posts loaded")
        return False

    print(f"✓ Loaded {len(posts)} posts")

    # Step 2: Clean text
    print("\n[2/6] Cleaning text...")
    cleaner = SocialMediaCleaner()
    for post in posts:
        result = cleaner.clean(post.text)
        post.clean_text = result['clean_text']
        post.processed = True
    print(f"✓ Cleaned {len(posts)} posts")

    # Step 3: Detect duplicates
    print("\n[3/6] Detecting duplicates...")
    dedup = DuplicateDetector()
    texts = [p.text for p in posts]
    duplicates = dedup.find_duplicates(texts, threshold=0.95)

    duplicate_count = 0
    for idx_list in duplicates:
        if len(idx_list) > 1:
            # Mark all but first as duplicate
            for idx in idx_list[1:]:
                posts[idx].is_duplicate = True
                duplicate_count += 1

    print(f"✓ Found {duplicate_count} duplicate posts (marked)")

    # Step 4: Detect spam
    print("\n[4/6] Detecting spam/noise...")
    spam_filter = SpamFilter()
    spam_count = 0
    for post in posts:
        score = spam_filter.score_spam(post.text)
        post.spam_score = score
        if score > 0.7:
            post.is_spam = True
            spam_count += 1

    print(f"✓ Identified {spam_count} spam posts (marked)")

    # Step 5: Detect language
    print("\n[5/6] Detecting language...")
    lang_detector = LanguageDetector()
    for post in posts:
        lang = lang_detector.detect(post.text)
        post.language = lang

    lang_dist = {}
    for post in posts:
        lang_dist[post.language.value] = lang_dist.get(post.language.value, 0) + 1

    print(f"✓ Language distribution: {lang_dist}")

    # Step 6: Save prepared data
    print("\n[6/6] Saving prepared data...")

    # Convert to CSV
    data = [post.to_dict(include_derived=True) for post in posts]
    df = pd.DataFrame(data)

    # Flatten nested engagement dict
    if 'engagement' in df.columns:
        engagement_df = pd.json_normalize(df['engagement'])
        engagement_df.columns = ['engagement_' + col for col in engagement_df.columns]
        df = pd.concat([df.drop('engagement', axis=1), engagement_df], axis=1)

    df.to_csv(output_path, index=False)
    print(f"✓ Saved {len(posts)} posts to {output_path}")

    # Print summary
    print("\n" + "="*70)
    print("PREPARATION SUMMARY")
    print("="*70)
    print(f"Total posts: {len(posts)}")
    print(f"Duplicates (marked): {duplicate_count}")
    print(f"Spam (marked): {spam_count}")
    print(f"Clean posts: {len(posts) - duplicate_count - spam_count}")
    print(f"Output: {output_path}")
    print("="*70)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for experiments")
    parser.add_argument("--input", required=True, help="Input CSV or JSON file")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    success = prepare_dataset(args.input, args.output, args.verbose)
    sys.exit(0 if success else 1)

#!/usr/bin/env python
"""
Run Baseline Methods for Comparison

Implements 3 baseline trending methods:
1. Frequency-based (most common keywords)
2. TF-IDF + K-Means clustering
3. Embedding-based clustering (UMAP + K-Means)

Usage:
    python scripts/run_baselines.py --data dataset/test_data_small.csv --output results/baselines.json
"""

import argparse
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import List, Dict
from ingestion.loader import DataLoader
from preprocessing import SocialMediaCleaner
from evaluation.baselines import FrequencyBaseline, TfidfBaseline, EmbeddingBaseline


def run_baselines(data_path: str, output_path: str):
    """Run all baseline methods and save results."""

    print("="*70)
    print("BASELINE METHODS EVALUATION")
    print("="*70)

    # Step 1: Load and preprocess data
    print("\n[1/4] Loading data...")
    loader = DataLoader()
    posts = loader.load(data_path)

    if not posts:
        print("ERROR: No posts loaded")
        return False

    print(f"✓ Loaded {len(posts)} posts")

    # Clean text
    cleaner = SocialMediaCleaner()
    texts = []
    for post in posts:
        result = cleaner.clean(post.text)
        texts.append(result['clean_text'])

    results = {
        'metadata': {
            'dataset': data_path,
            'num_posts': len(posts),
            'timestamp': str(__import__('datetime').datetime.now())
        },
        'baselines': {}
    }

    # Step 2: Frequency baseline
    print("\n[2/4] Running Frequency Baseline...")
    try:
        freq_baseline = FrequencyBaseline()
        freq_result = freq_baseline.rank_topics(texts, top_k=10)
        results['baselines']['frequency'] = {
            'method': 'Frequency-based keyword ranking',
            'top_topics': freq_result,
            'score': 'Frequency count',
            'status': 'success'
        }
        print(f"✓ Found {len(freq_result)} top topics")
    except Exception as e:
        print(f"✗ Frequency baseline failed: {e}")
        results['baselines']['frequency'] = {'status': 'failed', 'error': str(e)}

    # Step 3: TF-IDF baseline
    print("\n[3/4] Running TF-IDF + K-Means Baseline...")
    try:
        tfidf_baseline = TfidfBaseline(n_clusters=5)
        tfidf_result = tfidf_baseline.discover_topics(texts)
        results['baselines']['tfidf_kmeans'] = {
            'method': 'TF-IDF + K-Means clustering',
            'n_clusters': tfidf_result.get('n_clusters', 5),
            'topics': tfidf_result.get('topics', []),
            'silhouette_score': tfidf_result.get('silhouette_score', None),
            'status': 'success'
        }
        print(f"✓ Discovered {tfidf_result.get('n_clusters', 5)} topics")
    except Exception as e:
        print(f"✗ TF-IDF baseline failed: {e}")
        results['baselines']['tfidf_kmeans'] = {'status': 'failed', 'error': str(e)}

    # Step 4: Embedding baseline
    print("\n[4/4] Running Embedding + K-Means Baseline...")
    try:
        emb_baseline = EmbeddingBaseline(n_clusters=5)
        emb_result = emb_baseline.discover_topics(texts)
        results['baselines']['embedding_kmeans'] = {
            'method': 'Embedding-based + K-Means clustering',
            'n_clusters': emb_result.get('n_clusters', 5),
            'topics': emb_result.get('topics', []),
            'silhouette_score': emb_result.get('silhouette_score', None),
            'status': 'success'
        }
        print(f"✓ Discovered {emb_result.get('n_clusters', 5)} topics")
    except Exception as e:
        print(f"✗ Embedding baseline failed: {e}")
        results['baselines']['embedding_kmeans'] = {'status': 'failed', 'error': str(e)}

    # Save results
    print("\n" + "="*70)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"✓ Baseline results saved to {output_path}")
    print("="*70)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline methods for trending")
    parser.add_argument("--data", required=True, help="Input CSV or JSON file")
    parser.add_argument("--output", required=True, help="Output JSON file")

    args = parser.parse_args()

    success = run_baselines(args.data, args.output)
    sys.exit(0 if success else 1)

#!/usr/bin/env python
"""
Run Ablation Study

Tests the impact of each component by removing it and measuring performance degradation.

Components tested:
1. Deduplication
2. Spam filtering
3. Semantic clustering
4. Novelty scoring
5. Temporal analysis

Usage:
    python scripts/run_ablation.py --data dataset/test_data_small.csv --output results/ablation.json
"""

import argparse
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.loader import DataLoader
from evaluation.ablation import AblationStudy


def run_ablation(data_path: str, output_path: str):
    """Run ablation study on all components."""

    print("="*70)
    print("ABLATION STUDY")
    print("="*70)

    # Load data
    print("\n[1/2] Loading data...")
    loader = DataLoader()
    posts = loader.load(data_path)

    if not posts:
        print("ERROR: No posts loaded")
        return False

    print(f"✓ Loaded {len(posts)} posts")

    # Run ablation study
    print("\n[2/2] Running ablation experiments...")

    ablation = AblationStudy()
    results = ablation.run_full_ablation(posts)

    # Save results
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✓ Ablation results saved to {output_path}")
    print("="*70)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument("--data", required=True, help="Input CSV or JSON file")
    parser.add_argument("--output", required=True, help="Output JSON file")

    args = parser.parse_args()

    success = run_ablation(args.data, args.output)
    sys.exit(0 if success else 1)

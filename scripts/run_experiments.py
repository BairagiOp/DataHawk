#!/usr/bin/env python
"""
Run Proposed DataHawk Method Experiments

Implements the complete research pipeline for all 7 research questions.

Usage:
    python scripts/run_experiments.py --data dataset/test_data_small.csv --output results/proposed.json
"""

import argparse
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.run_experiment import ResearchExperiment


def run_experiments(data_path: str, output_path: str):
    """Run complete proposed method pipeline."""

    # Create output directory
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Run experiment
    experiment = ResearchExperiment(data_path)

    try:
        experiment.run_full_experiment()

        # Save results
        with open(output_path, 'w') as f:
            json.dump(experiment.results, f, indent=2, default=str)

        print(f"\n✓ Experiment results saved to {output_path}")
        return True

    except Exception as e:
        print(f"\n✗ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run proposed DataHawk experiments")
    parser.add_argument("--data", required=True, help="Input CSV or JSON file")
    parser.add_argument("--output", required=True, help="Output JSON file")

    args = parser.parse_args()

    success = run_experiments(args.data, args.output)
    sys.exit(0 if success else 1)

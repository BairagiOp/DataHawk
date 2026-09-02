"""
Ablation Study Framework

Systematically evaluate the impact of each component.

RQ7: Which features contribute most to trend detection accuracy?
"""

import numpy as np
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
import itertools


@dataclass
class AblationResult:
    """Result from one ablation configuration"""
    config: Dict[str, Any]
    metric_value: float
    metric_name: str
    components_enabled: List[str]
    components_disabled: List[str]


class AblationStudy:
    """
    Framework for ablation studies.

    Tests system performance with different component combinations:
    - Full system (all components)
    - -Volume (without volume feature)
    - -Growth (without growth feature)
    - -Engagement (without engagement feature)
    - -Novelty (without novelty feature)
    - Minimal (only one feature)
    """

    def __init__(self,
                 components: List[str],
                 evaluation_fn: Callable[[Dict[str, bool]], float],
                 metric_name: str = 'score'):
        """
        Args:
            components: List of component names
            evaluation_fn: Function that takes config dict and returns metric
            metric_name: Name of metric being evaluated
        """
        self.components = components
        self.evaluation_fn = evaluation_fn
        self.metric_name = metric_name
        self.results: List[AblationResult] = []

    def run_full_ablation(self) -> List[AblationResult]:
        """
        Run complete ablation study.

        Tests:
        1. Full system (all components)
        2. Leave-one-out (disable each component individually)
        3. Minimal (only one component at a time)

        Returns:
            List of AblationResult objects
        """
        self.results = []

        # 1. Full system
        print("Running ablation study...")
        print("\n1. Full System:")
        full_config = {comp: True for comp in self.components}
        metric = self.evaluation_fn(full_config)
        result = AblationResult(
            config=full_config,
            metric_value=metric,
            metric_name=self.metric_name,
            components_enabled=self.components.copy(),
            components_disabled=[]
        )
        self.results.append(result)
        print(f"   All components: {metric:.4f}")

        # 2. Leave-one-out
        print("\n2. Leave-One-Out (disable each component):")
        for component in self.components:
            config = {comp: comp != component for comp in self.components}
            metric = self.evaluation_fn(config)

            enabled = [c for c in self.components if config[c]]
            disabled = [c for c in self.components if not config[c]]

            result = AblationResult(
                config=config,
                metric_value=metric,
                metric_name=self.metric_name,
                components_enabled=enabled,
                components_disabled=disabled
            )
            self.results.append(result)

            # Impact = difference from full system
            impact = self.results[0].metric_value - metric
            print(f"   Without {component}: {metric:.4f} (impact: {impact:+.4f})")

        # 3. Minimal (only one component)
        print("\n3. Minimal (only one component):")
        for component in self.components:
            config = {comp: comp == component for comp in self.components}
            metric = self.evaluation_fn(config)

            enabled = [component]
            disabled = [c for c in self.components if c != component]

            result = AblationResult(
                config=config,
                metric_value=metric,
                metric_name=self.metric_name,
                components_enabled=enabled,
                components_disabled=disabled
            )
            self.results.append(result)
            print(f"   Only {component}: {metric:.4f}")

        return self.results

    def run_custom_configs(self, configs: List[Dict[str, bool]]) -> List[AblationResult]:
        """
        Run ablation on custom configurations.

        Args:
            configs: List of component configurations to test

        Returns:
            List of AblationResult objects
        """
        self.results = []

        for config in configs:
            metric = self.evaluation_fn(config)

            enabled = [c for c, v in config.items() if v]
            disabled = [c for c, v in config.items() if not v]

            result = AblationResult(
                config=config,
                metric_value=metric,
                metric_name=self.metric_name,
                components_enabled=enabled,
                components_disabled=disabled
            )
            self.results.append(result)

        return self.results

    def get_component_importance(self) -> Dict[str, float]:
        """
        Compute importance of each component.

        Importance = drop in metric when component is removed

        Returns:
            Dict mapping component name to importance score
        """
        if not self.results:
            raise ValueError("Run ablation study first")

        # Find full system result
        full_result = next(r for r in self.results if len(r.components_disabled) == 0)
        full_metric = full_result.metric_value

        # Find leave-one-out results
        importance = {}
        for component in self.components:
            # Find result where only this component is disabled
            result = next(
                (r for r in self.results
                 if r.components_disabled == [component]),
                None
            )

            if result:
                importance[component] = full_metric - result.metric_value

        return importance

    def print_summary(self):
        """Print formatted ablation summary"""
        if not self.results:
            print("No results yet. Run ablation study first.")
            return

        print("\n" + "="*70)
        print("ABLATION STUDY SUMMARY")
        print("="*70)

        # Component importance
        importance = self.get_component_importance()
        print("\nComponent Importance (impact when removed):")
        print("-" * 70)
        for component, impact in sorted(importance.items(),
                                       key=lambda x: x[1], reverse=True):
            print(f"  {component:<20} {impact:+.4f}")

        # Best configuration
        best_result = max(self.results, key=lambda r: r.metric_value)
        print(f"\nBest Configuration ({self.metric_name}={best_result.metric_value:.4f}):")
        print(f"  Enabled: {', '.join(best_result.components_enabled)}")

        # Worst configuration
        worst_result = min(self.results, key=lambda r: r.metric_value)
        print(f"\nWorst Configuration ({self.metric_name}={worst_result.metric_value:.4f}):")
        print(f"  Enabled: {', '.join(worst_result.components_enabled)}")

        print("="*70)


# Testing
if __name__ == "__main__":
    print("Ablation Study Framework Tests:\n")

    # Simulate trend scoring with different features
    def evaluate_trend_scoring(config: Dict[str, bool]) -> float:
        """
        Simulated evaluation function.

        Pretend we're evaluating trend detection accuracy
        with different features enabled/disabled.
        """
        # Simulate feature contributions
        base_score = 0.5

        feature_weights = {
            'volume': 0.10,
            'growth': 0.15,
            'engagement': 0.12,
            'novelty': 0.08
        }

        score = base_score
        for feature, enabled in config.items():
            if enabled:
                score += feature_weights.get(feature, 0)

        # Add small random noise
        noise = np.random.normal(0, 0.01)
        score += noise

        return score

    # Test 1: Full ablation study
    print("Test 1: Full Ablation Study")
    print("-" * 70)

    components = ['volume', 'growth', 'engagement', 'novelty']

    np.random.seed(42)
    study = AblationStudy(
        components=components,
        evaluation_fn=evaluate_trend_scoring,
        metric_name='F1'
    )

    results = study.run_full_ablation()
    study.print_summary()

    # Test 2: Custom configurations
    print("\n\nTest 2: Custom Configurations")
    print("-" * 70)

    custom_configs = [
        {'volume': True, 'growth': True, 'engagement': False, 'novelty': False},
        {'volume': False, 'growth': False, 'engagement': True, 'novelty': True},
        {'volume': True, 'growth': False, 'engagement': True, 'novelty': False},
    ]

    np.random.seed(42)
    study2 = AblationStudy(
        components=components,
        evaluation_fn=evaluate_trend_scoring,
        metric_name='Accuracy'
    )

    results = study2.run_custom_configs(custom_configs)

    print("\nCustom Configuration Results:")
    for result in results:
        print(f"  {'+'.join(result.components_enabled)}: {result.metric_value:.4f}")

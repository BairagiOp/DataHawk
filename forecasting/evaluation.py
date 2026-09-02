"""
Forecast Evaluation

Metrics for comparing forecasting methods:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- Direction Accuracy (correct trend direction)
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ForecastMetrics:
    """Evaluation metrics for forecasts"""
    mae: float
    rmse: float
    mape: float
    direction_accuracy: float
    method: str


class ForecastEvaluator:
    """
    Evaluate forecast accuracy.

    Compares predictions against ground truth.
    """

    def __init__(self):
        pass

    def evaluate(self,
                y_true: np.ndarray,
                y_pred: np.ndarray,
                method: str = 'unknown') -> ForecastMetrics:
        """
        Compute forecast metrics.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            method: Method name for reporting

        Returns:
            ForecastMetrics object
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have same length")

        if len(y_true) == 0:
            raise ValueError("Cannot evaluate empty arrays")

        # MAE (Mean Absolute Error)
        mae = np.mean(np.abs(y_true - y_pred))

        # RMSE (Root Mean Squared Error)
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        # MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero
        nonzero_mask = y_true != 0
        if nonzero_mask.any():
            mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask])
                                 / y_true[nonzero_mask])) * 100
        else:
            mape = np.inf

        # Direction accuracy (correct trend direction)
        if len(y_true) > 1:
            true_direction = np.sign(np.diff(y_true))
            pred_direction = np.sign(np.diff(y_pred))
            direction_accuracy = np.mean(true_direction == pred_direction)
        else:
            direction_accuracy = np.nan

        return ForecastMetrics(
            mae=mae,
            rmse=rmse,
            mape=mape,
            direction_accuracy=direction_accuracy,
            method=method
        )

    def compare_methods(self,
                       y_true: np.ndarray,
                       predictions: Dict[str, np.ndarray]) -> Dict[str, ForecastMetrics]:
        """
        Compare multiple forecasting methods.

        Args:
            y_true: Ground truth
            predictions: Dict mapping method name to predictions

        Returns:
            Dict mapping method name to metrics
        """
        results = {}

        for method, y_pred in predictions.items():
            metrics = self.evaluate(y_true, y_pred, method=method)
            results[method] = metrics

        return results

    def print_comparison(self, results: Dict[str, ForecastMetrics]):
        """Print formatted comparison table"""
        print("\nForecast Method Comparison:")
        print("-" * 70)
        print(f"{'Method':<20} {'MAE':<10} {'RMSE':<10} {'MAPE':<10} {'Dir. Acc.':<10}")
        print("-" * 70)

        for method, metrics in results.items():
            print(f"{method:<20} {metrics.mae:<10.2f} {metrics.rmse:<10.2f} "
                  f"{metrics.mape:<10.2f} {metrics.direction_accuracy:<10.2%}")

        print("-" * 70)

        # Highlight best method for each metric
        best_mae = min(results.values(), key=lambda m: m.mae).method
        best_rmse = min(results.values(), key=lambda m: m.rmse).method
        best_mape = min(results.values(), key=lambda m: m.mape).method
        best_dir = max(results.values(), key=lambda m: m.direction_accuracy).method

        print(f"\nBest MAE:  {best_mae}")
        print(f"Best RMSE: {best_rmse}")
        print(f"Best MAPE: {best_mape}")
        print(f"Best Direction Accuracy: {best_dir}")


def compute_forecast_metrics(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            method: str = 'unknown') -> ForecastMetrics:
    """
    Convenience function for computing forecast metrics.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        method: Method name

    Returns:
        ForecastMetrics object
    """
    evaluator = ForecastEvaluator()
    return evaluator.evaluate(y_true, y_pred, method=method)


# Testing
if __name__ == "__main__":
    print("Forecast Evaluation Tests:\n")

    np.random.seed(42)

    # Ground truth
    y_true = np.array([100, 110, 105, 115, 120, 118, 125, 130])

    # Simulate predictions from different methods
    naive_pred = np.array([100, 100, 110, 105, 115, 120, 118, 125])  # Last value
    ma_pred = np.array([100, 105, 105, 110, 113, 117, 121, 124])     # Smoothed
    perfect_pred = y_true + np.random.normal(0, 2, len(y_true))      # Nearly perfect

    # Evaluate each method
    evaluator = ForecastEvaluator()

    print("1. Individual Method Evaluation:")
    naive_metrics = evaluator.evaluate(y_true, naive_pred, method='Naive')
    print(f"   Naive: MAE={naive_metrics.mae:.2f}, RMSE={naive_metrics.rmse:.2f}")

    # Compare all methods
    predictions = {
        'Naive': naive_pred,
        'Moving Average': ma_pred,
        'Near Perfect': perfect_pred
    }

    results = evaluator.compare_methods(y_true, predictions)
    evaluator.print_comparison(results)

    # Test direction accuracy specifically
    print("\n2. Direction Accuracy Test:")
    upward_true = np.array([10, 15, 20, 25, 30])
    correct_direction = np.array([10, 16, 21, 26, 31])  # All correct directions
    wrong_direction = np.array([10, 14, 19, 23, 28])    # Close but wrong directions

    correct_metrics = compute_forecast_metrics(upward_true, correct_direction, 'Correct Dir')
    wrong_metrics = compute_forecast_metrics(upward_true, wrong_direction, 'Wrong Dir')

    print(f"   Correct directions: {correct_metrics.direction_accuracy:.0%}")
    print(f"   Wrong directions: {wrong_metrics.direction_accuracy:.0%}")

"""
Baseline Forecasting Methods

Simple forecasting baselines for fair comparison:
- Naive: Repeat last value
- Moving Average: Average of recent window
- Linear Regression: Simple linear trend
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BaselineForecast:
    """Forecast result from baseline method"""
    predictions: np.ndarray
    method: str
    confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None


class NaiveForecaster:
    """
    Naive baseline: repeat last observed value.

    Simple but surprisingly effective baseline for many time series.
    """

    def __init__(self):
        self.last_value = None

    def fit(self, values: np.ndarray):
        """Fit on historical data"""
        if len(values) == 0:
            raise ValueError("Cannot fit on empty data")
        self.last_value = values[-1]
        return self

    def predict(self, steps: int = 1) -> BaselineForecast:
        """
        Predict future values.

        Args:
            steps: Number of steps to forecast

        Returns:
            BaselineForecast with predictions
        """
        if self.last_value is None:
            raise ValueError("Model not fitted")

        predictions = np.full(steps, self.last_value)

        return BaselineForecast(
            predictions=predictions,
            method='naive'
        )


class MovingAverageForecaster:
    """
    Moving average baseline: predict average of recent window.

    Smooths out short-term fluctuations.
    """

    def __init__(self, window: int = 7):
        """
        Args:
            window: Size of moving average window
        """
        self.window = window
        self.recent_values = None

    def fit(self, values: np.ndarray):
        """Fit on historical data"""
        if len(values) == 0:
            raise ValueError("Cannot fit on empty data")

        # Store recent values
        self.recent_values = values[-self.window:]
        return self

    def predict(self, steps: int = 1) -> BaselineForecast:
        """
        Predict future values.

        Args:
            steps: Number of steps to forecast

        Returns:
            BaselineForecast with predictions
        """
        if self.recent_values is None:
            raise ValueError("Model not fitted")

        # Predict mean of recent window
        prediction = np.mean(self.recent_values)
        predictions = np.full(steps, prediction)

        # Simple confidence interval (±1 std)
        std = np.std(self.recent_values)
        lower = predictions - std
        upper = predictions + std

        return BaselineForecast(
            predictions=predictions,
            method='moving_average',
            confidence_intervals=(lower, upper)
        )


class LinearRegressionForecaster:
    """
    Linear regression baseline: fit linear trend.

    Captures simple linear growth/decline patterns.
    """

    def __init__(self):
        self.slope = None
        self.intercept = None
        self.n_points = 0

    def fit(self, values: np.ndarray):
        """Fit linear trend"""
        if len(values) < 2:
            raise ValueError("Need at least 2 points for linear regression")

        self.n_points = len(values)

        # Fit linear regression
        x = np.arange(len(values))

        # Compute slope and intercept using least squares
        x_mean = np.mean(x)
        y_mean = np.mean(values)

        numerator = np.sum((x - x_mean) * (values - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        self.slope = numerator / denominator
        self.intercept = y_mean - self.slope * x_mean

        return self

    def predict(self, steps: int = 1) -> BaselineForecast:
        """
        Predict future values.

        Args:
            steps: Number of steps to forecast

        Returns:
            BaselineForecast with predictions
        """
        if self.slope is None:
            raise ValueError("Model not fitted")

        # Extrapolate linear trend
        future_x = np.arange(self.n_points, self.n_points + steps)
        predictions = self.slope * future_x + self.intercept

        return BaselineForecast(
            predictions=predictions,
            method='linear_regression'
        )


# Testing
if __name__ == "__main__":
    print("Baseline Forecasting Tests:\n")

    # Synthetic time series
    np.random.seed(42)

    # Test 1: Naive on stable series
    stable_series = np.array([10, 11, 10, 12, 11, 10, 11])
    naive = NaiveForecaster()
    naive.fit(stable_series)
    forecast = naive.predict(steps=3)
    print(f"1. Naive Forecast (stable):")
    print(f"   Last value: {stable_series[-1]}")
    print(f"   Predictions: {forecast.predictions}")

    # Test 2: Moving average on noisy series
    noisy_series = np.array([10, 15, 8, 12, 14, 9, 11, 13, 10, 12])
    ma = MovingAverageForecaster(window=3)
    ma.fit(noisy_series)
    forecast = ma.predict(steps=3)
    print(f"\n2. Moving Average Forecast (window=3):")
    print(f"   Recent values: {noisy_series[-3:]}")
    print(f"   Predictions: {forecast.predictions}")
    print(f"   Confidence: [{forecast.confidence_intervals[0][0]:.1f}, {forecast.confidence_intervals[1][0]:.1f}]")

    # Test 3: Linear regression on trending series
    trending_series = np.array([10, 12, 14, 13, 15, 17, 16, 18, 20])
    lr = LinearRegressionForecaster()
    lr.fit(trending_series)
    forecast = lr.predict(steps=3)
    print(f"\n3. Linear Regression Forecast (upward trend):")
    print(f"   Slope: {lr.slope:.2f}")
    print(f"   Predictions: {forecast.predictions}")

    # Test 4: Comparison on same series
    print(f"\n4. Method Comparison (same series):")
    test_series = np.array([100, 105, 110, 108, 115, 120, 118, 125])

    naive_pred = NaiveForecaster().fit(test_series).predict(1).predictions[0]
    ma_pred = MovingAverageForecaster(window=3).fit(test_series).predict(1).predictions[0]
    lr_pred = LinearRegressionForecaster().fit(test_series).predict(1).predictions[0]

    print(f"   Naive:           {naive_pred:.1f}")
    print(f"   Moving Average:  {ma_pred:.1f}")
    print(f"   Linear Reg:      {lr_pred:.1f}")

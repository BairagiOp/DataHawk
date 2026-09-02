"""
Advanced Forecasting Models

Proposed methods using ensemble approaches:
- XGBoost for trend forecasting
- Random Forest for comparison
- Feature engineering from temporal patterns
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import warnings

try:
    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    warnings.warn("XGBoost/sklearn not installed. Install with: pip install xgboost scikit-learn")


@dataclass
class ForecastResult:
    """Result from trend forecasting"""
    predictions: np.ndarray
    confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]]
    method: str
    feature_importance: Optional[Dict[str, float]] = None


class TrendForecaster:
    """
    Advanced trend forecaster using ensemble methods.

    Features:
    - Temporal features (lag, rolling stats, trend)
    - Growth rate features
    - Engagement patterns
    - Cyclical patterns (day of week, hour)

    RQ6: Can ensemble methods predict trend trajectories more accurately
         than statistical baselines?
    """

    def __init__(self,
                 method: str = 'xgboost',
                 lag_features: int = 7,
                 n_estimators: int = 100):
        """
        Args:
            method: 'xgboost' or 'random_forest'
            lag_features: Number of lag features to create
            n_estimators: Number of trees in ensemble
        """
        if not MODELS_AVAILABLE:
            raise ImportError("Install xgboost and scikit-learn first")

        self.method = method
        self.lag_features = lag_features
        self.n_estimators = n_estimators

        # Initialize model
        if method == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif method == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=10,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        self.feature_names = []
        self.history = None

    def _create_features(self, values: np.ndarray) -> np.ndarray:
        """
        Create temporal features from time series.

        Features:
        - Lag features (t-1, t-2, ...)
        - Rolling statistics (mean, std, min, max)
        - Trend features (linear, acceleration)
        - Growth rate
        """
        n = len(values)
        features = []
        self.feature_names = []

        # Lag features
        for i in range(1, min(self.lag_features + 1, n)):
            lag_values = np.concatenate([
                np.full(i, np.nan),
                values[:-i]
            ])
            features.append(lag_values)
            self.feature_names.append(f'lag_{i}')

        # Rolling mean (window=3)
        if n >= 3:
            rolling_mean = np.convolve(values, np.ones(3)/3, mode='same')
            features.append(rolling_mean)
            self.feature_names.append('rolling_mean_3')

        # Rolling std (window=3)
        if n >= 3:
            rolling_std = np.array([
                np.std(values[max(0, i-2):i+1]) if i >= 2 else 0
                for i in range(n)
            ])
            features.append(rolling_std)
            self.feature_names.append('rolling_std_3')

        # Growth rate
        growth_rate = np.zeros(n)
        growth_rate[1:] = (values[1:] - values[:-1]) / (values[:-1] + 1e-6)
        features.append(growth_rate)
        self.feature_names.append('growth_rate')

        # Time index (linear trend)
        time_idx = np.arange(n) / n
        features.append(time_idx)
        self.feature_names.append('time_index')

        # Stack features
        X = np.column_stack(features)

        # Handle NaN values (from lag features)
        # Forward fill
        for col in range(X.shape[1]):
            mask = np.isnan(X[:, col])
            if mask.any():
                first_valid = np.where(~mask)[0]
                if len(first_valid) > 0:
                    X[mask, col] = X[first_valid[0], col]

        return X

    def fit(self, values: np.ndarray):
        """
        Fit forecasting model.

        Args:
            values: Time series values
        """
        if len(values) < self.lag_features + 2:
            raise ValueError(f"Need at least {self.lag_features + 2} points")

        self.history = values.copy()

        # Create features
        X = self._create_features(values)

        # Target is next value (shift by 1)
        y = values

        # Use all but last point for training
        # (last point has no target)
        X_train = X[:-1]
        y_train = y[1:]

        # Fit model
        self.model.fit(X_train, y_train)

        return self

    def predict(self, steps: int = 1) -> ForecastResult:
        """
        Predict future values.

        Args:
            steps: Number of steps to forecast

        Returns:
            ForecastResult with predictions and confidence intervals
        """
        if self.history is None:
            raise ValueError("Model not fitted")

        predictions = []
        current_series = self.history.copy()

        # Iterative forecasting
        for _ in range(steps):
            # Create features from current series
            X = self._create_features(current_series)

            # Predict next value
            next_value = self.model.predict(X[-1:].reshape(1, -1))[0]
            predictions.append(next_value)

            # Append to series for next iteration
            current_series = np.append(current_series, next_value)

        predictions = np.array(predictions)

        # Compute confidence intervals (±1 std from training residuals)
        X_train = self._create_features(self.history)[:-1]
        y_train = self.history[1:]
        train_preds = self.model.predict(X_train)
        residuals = y_train - train_preds
        std = np.std(residuals)

        lower = predictions - 1.96 * std  # 95% CI
        upper = predictions + 1.96 * std

        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance = dict(zip(
                self.feature_names,
                self.model.feature_importances_
            ))
        else:
            importance = None

        return ForecastResult(
            predictions=predictions,
            confidence_intervals=(lower, upper),
            method=self.method,
            feature_importance=importance
        )


# Testing
if __name__ == "__main__":
    if not MODELS_AVAILABLE:
        print("XGBoost/sklearn not installed. Skipping tests.")
        print("Install with: pip install xgboost scikit-learn")
    else:
        print("Advanced Forecasting Tests:\n")

        np.random.seed(42)

        # Test 1: XGBoost on upward trend
        trend_series = np.array([10, 12, 15, 14, 17, 20, 19, 22, 25, 24, 27, 30])

        forecaster = TrendForecaster(method='xgboost', lag_features=3)
        forecaster.fit(trend_series)
        result = forecaster.predict(steps=3)

        print(f"1. XGBoost Forecast (upward trend):")
        print(f"   Historical: {trend_series[-3:]}")
        print(f"   Predictions: {result.predictions}")
        print(f"   Confidence: [{result.confidence_intervals[0][0]:.1f}, {result.confidence_intervals[1][0]:.1f}]")

        if result.feature_importance:
            print(f"   Top features:")
            for feat, imp in sorted(result.feature_importance.items(),
                                   key=lambda x: x[1], reverse=True)[:3]:
                print(f"     {feat}: {imp:.3f}")

        # Test 2: Random Forest comparison
        rf_forecaster = TrendForecaster(method='random_forest', lag_features=3)
        rf_forecaster.fit(trend_series)
        rf_result = rf_forecaster.predict(steps=3)

        print(f"\n2. Random Forest Forecast (same data):")
        print(f"   Predictions: {rf_result.predictions}")

        # Test 3: Seasonal pattern
        seasonal_series = np.array([
            10, 15, 20, 15, 10,  # cycle 1
            12, 17, 22, 17, 12,  # cycle 2
            14, 19, 24, 19, 14   # cycle 3
        ])

        forecaster.fit(seasonal_series)
        result = forecaster.predict(steps=5)

        print(f"\n3. Forecast on Seasonal Pattern:")
        print(f"   Last cycle: {seasonal_series[-5:]}")
        print(f"   Next cycle: {result.predictions}")

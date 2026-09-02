"""
DataHawk Forecasting Module

Time-series forecasting for trend prediction and early warning.
"""

from .baseline import NaiveForecaster, MovingAverageForecaster, LinearRegressionForecaster
from .models import TrendForecaster, ForecastResult
from .evaluation import ForecastEvaluator, compute_forecast_metrics

__all__ = [
    'NaiveForecaster',
    'MovingAverageForecaster',
    'LinearRegressionForecaster',
    'TrendForecaster',
    'ForecastResult',
    'ForecastEvaluator',
    'compute_forecast_metrics',
]

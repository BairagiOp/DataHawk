"""
DataHawk Configuration Settings

Centralized configuration for the research platform.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'dataset'
RESEARCH_DIR = PROJECT_ROOT / 'research'
CACHE_DIR = PROJECT_ROOT / '.cache'

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
RESEARCH_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# API Keys (load from environment)
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

# Research Settings
RANDOM_SEED = int(os.getenv('RANDOM_SEED', 42))
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Trend Scoring Weights (RQ7: Ablation)
ALPHA = float(os.getenv('ALPHA', 0.2))   # Volume weight
BETA = float(os.getenv('BETA', 0.4))     # Growth weight
GAMMA = float(os.getenv('GAMMA', 0.3))   # Engagement weight
DELTA = float(os.getenv('DELTA', 0.1))   # Novelty weight

# Topic Discovery
MIN_CLUSTER_SIZE = int(os.getenv('MIN_CLUSTER_SIZE', 5))
UMAP_DIMENSIONS = int(os.getenv('UMAP_DIMENSIONS', 5))
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

# Trend Detection
TREND_WINDOW_DAYS = 7
MIN_TREND_VOLUME = 10
GROWTH_THRESHOLD = 0.2  # 20% growth

# Emerging Trend Classification
VIRAL_GROWTH_THRESHOLD = 2.0  # 200% growth
STABLE_VARIANCE_THRESHOLD = 0.1
DECLINING_THRESHOLD = -0.1

# Forecasting
FORECAST_HORIZON = int(os.getenv('FORECAST_HORIZON', 7))
LAG_FEATURES = 7
XGBOOST_N_ESTIMATORS = 100

# NLP Settings
SENTIMENT_METHOD = 'vader'  # 'vader' or 'llm'
EMOTION_THRESHOLD = 0.3
NER_MODEL = 'en_core_web_sm'

# Preprocessing
SPAM_THRESHOLD = 0.5
DUPLICATE_SIMILARITY_THRESHOLD = 0.9

# Evaluation
EVALUATION_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1',
    'mae',
    'rmse',
    'mape'
]

# Ethical Guidelines
ENABLE_WEB_SCRAPING = os.getenv('ENABLE_WEB_SCRAPING', 'false').lower() == 'true'
RESPECT_ROBOTS_TXT = os.getenv('RESPECT_ROBOTS_TXT', 'true').lower() == 'true'
RATE_LIMIT_SECONDS = float(os.getenv('RATE_LIMIT_SECONDS', 1.0))

# Logging format
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


def get_settings():
    """
    Get configuration settings as a dictionary.

    Returns:
        dict: All configuration settings
    """
    return {
        'PROJECT_ROOT': PROJECT_ROOT,
        'DATA_DIR': DATA_DIR,
        'RESEARCH_DIR': RESEARCH_DIR,
        'CACHE_DIR': CACHE_DIR,
        'GOOGLE_API_KEY': GOOGLE_API_KEY,
        'OPENAI_API_KEY': OPENAI_API_KEY,
        'RANDOM_SEED': RANDOM_SEED,
        'LOG_LEVEL': LOG_LEVEL,
        'ALPHA': ALPHA,
        'BETA': BETA,
        'GAMMA': GAMMA,
        'DELTA': DELTA,
        'MIN_CLUSTER_SIZE': MIN_CLUSTER_SIZE,
        'UMAP_DIMENSIONS': UMAP_DIMENSIONS,
        'EMBEDDING_MODEL': EMBEDDING_MODEL,
        'TREND_WINDOW_DAYS': TREND_WINDOW_DAYS,
        'MIN_TREND_VOLUME': MIN_TREND_VOLUME,
        'GROWTH_THRESHOLD': GROWTH_THRESHOLD,
        'VIRAL_GROWTH_THRESHOLD': VIRAL_GROWTH_THRESHOLD,
        'STABLE_VARIANCE_THRESHOLD': STABLE_VARIANCE_THRESHOLD,
        'DECLINING_THRESHOLD': DECLINING_THRESHOLD,
        'FORECAST_HORIZON': FORECAST_HORIZON,
        'LAG_FEATURES': LAG_FEATURES,
        'XGBOOST_N_ESTIMATORS': XGBOOST_N_ESTIMATORS,
        'SENTIMENT_METHOD': SENTIMENT_METHOD,
        'EMOTION_THRESHOLD': EMOTION_THRESHOLD,
        'NER_MODEL': NER_MODEL,
        'SPAM_THRESHOLD': SPAM_THRESHOLD,
        'DUPLICATE_SIMILARITY_THRESHOLD': DUPLICATE_SIMILARITY_THRESHOLD,
        'EVALUATION_METRICS': EVALUATION_METRICS,
        'ENABLE_WEB_SCRAPING': ENABLE_WEB_SCRAPING,
        'RESPECT_ROBOTS_TXT': RESPECT_ROBOTS_TXT,
        'RATE_LIMIT_SECONDS': RATE_LIMIT_SECONDS,
        'LOG_FORMAT': LOG_FORMAT,
    }

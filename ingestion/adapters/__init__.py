"""
Ingestion Adapters Module

Adapters for loading data from various sources.
"""

from .csv_adapter import CSVAdapter, load_csv
from .json_adapter import JSONAdapter, load_json

__all__ = [
    'CSVAdapter',
    'load_csv',
    'JSONAdapter',
    'load_json',
]

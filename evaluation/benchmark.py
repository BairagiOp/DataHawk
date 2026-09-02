"""
DataHawk — Benchmark Runner & Experiment Logger

Runs the benchmark dataset through specified methods, computes metrics
against ground truth, and logs everything to SQLite + CSV for analysis.
"""

import csv
import json
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import get_settings
from evaluation.metrics import MetricsCalculator, ExtractionMetrics, SystemMetrics

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkEntry:
    """A single benchmark dataset entry."""
    id: str
    url: str
    category: str
    extraction_task: str
    schema: Dict[str, Any]
    ground_truth: List[Dict[str, Any]]
    cached_html_file: str = ""

    @classmethod
    def from_dict(cls, data: Dict) -> "BenchmarkEntry":
        return cls(
            id=data.get("id", ""),
            url=data.get("url", ""),
            category=data.get("category", ""),
            extraction_task=data.get("extraction_task", ""),
            schema=data.get("schema", {}),
            ground_truth=data.get("ground_truth", []),
            cached_html_file=data.get("cached_html_file", ""),
        )


@dataclass
class ExperimentLog:
    """A single experiment result log entry."""
    experiment_id: str = ""
    timestamp: str = ""
    website: str = ""
    category: str = ""
    method: str = ""
    scraping_strategy: str = ""
    schema_fields: str = ""
    num_fields: int = 0
    success: bool = False
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    exact_match: float = 0.0
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    retry_count: int = 0
    estimated_cost: float = 0.0
    validation_errors: int = 0
    final_confidence: float = 0.0
    html_size: int = 0
    cleaned_size: int = 0
    token_reduction_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class ExperimentLogger:
    """
    Logs experiment results to SQLite and CSV for research analysis.
    """

    def __init__(self, db_path: Optional[str] = None, csv_path: Optional[str] = None):
        settings = get_settings()
        self.db_path = db_path or settings.experiment_db_path
        self.csv_path = csv_path or settings.experiment_csv_path

        # Ensure directories exist
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)

        self._init_db()

    def _init_db(self) -> None:
        """Create the experiment log table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS experiment_logs (
                experiment_id TEXT PRIMARY KEY,
                timestamp TEXT,
                website TEXT,
                category TEXT,
                method TEXT,
                scraping_strategy TEXT,
                schema_fields TEXT,
                num_fields INTEGER,
                success INTEGER,
                precision REAL,
                recall REAL,
                f1 REAL,
                exact_match REAL,
                latency_ms REAL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                retry_count INTEGER,
                estimated_cost REAL,
                validation_errors INTEGER,
                final_confidence REAL,
                html_size INTEGER,
                cleaned_size INTEGER,
                token_reduction_pct REAL
            )
        """)
        conn.commit()
        conn.close()

    def log(self, entry: ExperimentLog) -> None:
        """Log a single experiment result."""
        if not entry.experiment_id:
            entry.experiment_id = str(uuid.uuid4())[:8]
        if not entry.timestamp:
            entry.timestamp = datetime.now().isoformat()

        # SQLite
        conn = sqlite3.connect(self.db_path)
        data = entry.to_dict()
        data["success"] = int(data["success"])
        columns = ", ".join(data.keys())
        placeholders = ", ".join("?" * len(data))
        conn.execute(
            f"INSERT OR REPLACE INTO experiment_logs ({columns}) VALUES ({placeholders})",
            list(data.values()),
        )
        conn.commit()
        conn.close()

        # CSV (append)
        file_exists = os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=entry.to_dict().keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(entry.to_dict())

        logger.info(f"Experiment logged: {entry.experiment_id} — {entry.method}")

    def get_all_logs(self) -> List[Dict[str, Any]]:
        """Retrieve all experiment logs."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM experiment_logs ORDER BY timestamp").fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_logs_by_method(self, method: str) -> List[Dict[str, Any]]:
        """Retrieve logs for a specific method."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM experiment_logs WHERE method = ? ORDER BY timestamp",
            (method,),
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def export_to_csv(self, output_path: str) -> None:
        """Export all logs to a CSV file."""
        logs = self.get_all_logs()
        if not logs:
            logger.warning("No experiment logs to export")
            return

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=logs[0].keys())
            writer.writeheader()
            writer.writerows(logs)

        logger.info(f"Exported {len(logs)} logs to {output_path}")


class BenchmarkRunner:
    """
    Runs benchmark tests across methods and logs results.
    """

    def __init__(
        self,
        experiment_logger: Optional[ExperimentLogger] = None,
    ):
        self.logger_inst = experiment_logger or ExperimentLogger()
        self.metrics = MetricsCalculator()

    def load_dataset(self, dataset_dir: Optional[str] = None) -> List[BenchmarkEntry]:
        """Load benchmark entries from dataset directory."""
        settings = get_settings()
        dataset_dir = dataset_dir or settings.ground_truth_dir

        entries = []
        if not os.path.exists(dataset_dir):
            logger.warning(f"Dataset directory not found: {dataset_dir}")
            return entries

        for filename in sorted(os.listdir(dataset_dir)):
            if filename.endswith(".json"):
                filepath = os.path.join(dataset_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    entries.append(BenchmarkEntry.from_dict(data))
                except Exception as e:
                    logger.error(f"Failed to load {filepath}: {e}")

        logger.info(f"Loaded {len(entries)} benchmark entries from {dataset_dir}")
        return entries

    def load_cached_html(self, entry: BenchmarkEntry) -> Optional[str]:
        """Load cached HTML for a benchmark entry."""
        settings = get_settings()
        if entry.cached_html_file:
            path = os.path.join(settings.dataset_dir, entry.cached_html_file)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
        return None

    def create_experiment_log(
        self,
        entry: BenchmarkEntry,
        method: str,
        extracted_records: List[Dict[str, Any]],
        latency_ms: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        scraping_strategy: str = "",
        retry_count: int = 0,
        validation_errors: int = 0,
        confidence: float = 0.0,
        html_size: int = 0,
        cleaned_size: int = 0,
    ) -> ExperimentLog:
        """Create and log an experiment entry with metrics."""
        schema_fields = list(entry.schema.get("fields", {}).keys())

        metrics = self.metrics.compute_extraction_metrics(
            extracted_records=extracted_records,
            ground_truth_records=entry.ground_truth,
            schema_fields=schema_fields,
        )

        token_reduction = (
            (1 - cleaned_size / html_size) * 100
            if html_size > 0 and cleaned_size > 0 else 0.0
        )

        estimated_cost = (
            (input_tokens / 1_000_000) * 0.10
            + (output_tokens / 1_000_000) * 0.40
        )

        log = ExperimentLog(
            website=entry.url,
            category=entry.category,
            method=method,
            scraping_strategy=scraping_strategy,
            schema_fields=json.dumps(schema_fields),
            num_fields=len(schema_fields),
            success=metrics.f1_score > 0,
            precision=metrics.precision,
            recall=metrics.recall,
            f1=metrics.f1_score,
            exact_match=metrics.exact_match,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            retry_count=retry_count,
            estimated_cost=estimated_cost,
            validation_errors=validation_errors,
            final_confidence=confidence,
            html_size=html_size,
            cleaned_size=cleaned_size,
            token_reduction_pct=token_reduction,
        )

        self.logger_inst.log(log)
        return log

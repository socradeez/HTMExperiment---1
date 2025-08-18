"""Metrics logging for BIO variant."""

import csv
import os
from typing import Dict, Any

FIELDNAMES = [
    "within_column_winner_count",
    "distal_bias_winner_mean",
    "distal_bias_nonwinner_mean",
    "column_precision",
    "column_recall",
    "predicted_columns",
    "notes",
]


def init_metrics(outdir: str) -> str:
    """Initialize metrics CSV and return its path."""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "metrics_bio.csv")
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
    return path


def append_row(csv_path: str, row: Dict[str, Any]) -> None:
    """Append a metrics row, writing headers if file is new."""
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

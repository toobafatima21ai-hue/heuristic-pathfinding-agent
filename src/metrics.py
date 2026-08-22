"""
metrics.py
Collects per-run results into a tabular structure and exports to CSV/JSON.
"""

import json
import csv
import os
from dataclasses import dataclass, asdict


@dataclass
class RunRecord:
    maze_id: str
    rows: int
    cols: int
    obstacle_density: float
    algorithm: str
    success: bool
    path_length: int
    cost: float
    nodes_expanded: int
    runtime_seconds: float


class MetricsLogger:
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.records: list[RunRecord] = []

    def add(self, record: RunRecord):
        self.records.append(record)

    def to_csv(self, filename: str = "run_metrics.csv"):
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(self.records[0]).keys()))
            writer.writeheader()
            for r in self.records:
                writer.writerow(asdict(r))
        return path

    def to_json(self, filename: str = "run_metrics.json"):
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            json.dump([asdict(r) for r in self.records], f, indent=2)
        return path
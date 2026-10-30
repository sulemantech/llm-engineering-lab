"""Reproducibility and experiment-logging helpers (infrastructure, not a learning target)."""

import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch

RESULTS_FILE = Path(__file__).resolve().parents[2] / "experiments" / "results" / "runs.jsonl"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def hardware() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return f"cpu:{platform.processor() or platform.machine()}"


def log_result(run_id: str, stage: int, config: dict, metrics: dict, notes: str = "") -> dict:
    """Append one run to experiments/results/runs.jsonl and return the record."""
    record = {
        "run_id": run_id,
        "stage": stage,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hardware": hardware(),
        "torch": torch.__version__,
        "config": config,
        "metrics": metrics,
        "notes": notes,
    }
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record

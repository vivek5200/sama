"""Temperature-based calibration. Per-head, stored in config.json."""

from dataclasses import dataclass
import torch
import numpy as np


@dataclass
class CalibrationConfig:
    temperature: float = 1.0
    ece_before: float = None
    ece_after: float = None
    n_bins: int = 10
    fitted_on: str = None
    method: str = "power_scaling"  # or "temperature_scaling" for logit-based


def apply_power_scaling(confidence: float, T: float) -> float:
    """Power-scaling for probabilities (used in v0.3+ training)."""
    return float(np.power(np.clip(confidence, 1e-7, 1 - 1e-7), T))


def ece(confs: np.ndarray, corrects: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (confs >= edges[i]) & (confs < edges[i + 1])
        if m.sum() > 0:
            e += m.sum() / len(confs) * abs(confs[m].mean() - corrects[m].mean())
    return float(e)

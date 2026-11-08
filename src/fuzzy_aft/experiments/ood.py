from __future__ import annotations

import hashlib
import json

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def stable_rng(dataset: str, seed: int, mode: str) -> np.random.Generator:
    """Return a perturbation RNG that is stable across Python processes."""
    payload = {
        "base_seed": 0,
        "experiment_name": "ood_detection",
        "dataset_name": dataset,
        "experiment_seed": int(seed),
        "scenario_params": {"mode": mode},
        "realization_id": 0,
    }
    key = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    value = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
    return np.random.default_rng(value % (2**32))


def tail_masks(
    x: np.ndarray, covariate: int = 0, quantile: float = 0.75
) -> tuple[np.ndarray, np.ndarray, float]:
    threshold = float(np.quantile(x[:, covariate], quantile))
    tail = x[:, covariate] >= threshold
    return ~tail, tail, threshold


def perturb(
    x: np.ndarray,
    rng: np.random.Generator,
    noise_sigma: float = 0.5,
    masking_probability: float = 0.3,
    masking_value: float = 2.0,
) -> np.ndarray:
    """Apply the Gaussian-noise and extreme masking protocol from the paper."""
    shifted = np.asarray(x, dtype=float).copy()
    shifted += rng.normal(0.0, noise_sigma, size=shifted.shape)
    mask = rng.random(size=shifted.shape) < masking_probability
    values = rng.choice(
        [-masking_value, masking_value], size=int(mask.sum()), replace=True
    ).astype(float)
    shifted[mask] = values
    return shifted


def ood_metrics(score_id: np.ndarray, score_ood: np.ndarray) -> tuple[float, float]:
    labels = np.concatenate(
        [np.zeros(len(score_id), dtype=int), np.ones(len(score_ood), dtype=int)]
    )
    scores = np.concatenate([score_id, score_ood])
    return float(roc_auc_score(labels, scores)), float(
        average_precision_score(labels, scores)
    )

"""Distance metrics over float32 vectors.

Cosine distance is implemented as ``1 - dot`` on L2-normalized vectors, so the
index normalizes once at insert/query time and then every distance is a single
dot product.
"""

from __future__ import annotations

import numpy as np

Metric = str  # "cosine" | "l2"


def as_f32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalize the last axis. Zero vectors are left as zeros."""
    x = as_f32(x)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    norm = np.where(norm == 0.0, 1.0, norm)
    return x / norm


def prepare(x: np.ndarray, metric: Metric) -> np.ndarray:
    """Pre-process vectors for storage given the metric."""
    x = as_f32(x)
    return normalize(x) if metric == "cosine" else x


def distances(query: np.ndarray, matrix: np.ndarray, metric: Metric) -> np.ndarray:
    """Distance from a single ``query`` (d,) to each row of ``matrix`` (n, d).

    For ``cosine`` both inputs are assumed already normalized.
    """
    if matrix.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    if metric == "cosine":
        return 1.0 - matrix @ query
    diff = matrix - query
    return np.sqrt(np.einsum("ij,ij->i", diff, diff))

"""Exact (brute-force) nearest-neighbor search.

Used as the ground truth to measure the approximate index's recall, and as a
perfectly correct fallback for small datasets.
"""

from __future__ import annotations

import numpy as np

from .distance import Metric, distances, prepare


class BruteForceIndex:
    def __init__(self, dim: int, metric: Metric = "cosine") -> None:
        self.dim = dim
        self.metric = metric
        self._vectors = np.empty((0, dim), dtype=np.float32)
        self._ids: list = []

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, vectors: np.ndarray, ids: list) -> None:
        vectors = prepare(np.atleast_2d(vectors), self.metric)
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape[1]}")
        if len(ids) != vectors.shape[0]:
            raise ValueError("len(ids) must match number of vectors")
        self._vectors = np.vstack([self._vectors, vectors])
        self._ids.extend(ids)

    def search(self, query: np.ndarray, k: int) -> list[tuple]:
        if len(self) == 0:
            return []
        q = prepare(np.atleast_2d(query), self.metric)[0]
        d = distances(q, self._vectors, self.metric)
        k = min(k, len(self._ids))
        # argpartition for the k smallest, then sort those k.
        idx = np.argpartition(d, k - 1)[:k]
        idx = idx[np.argsort(d[idx], kind="stable")]
        return [(self._ids[i], float(d[i])) for i in idx]

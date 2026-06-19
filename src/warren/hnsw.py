"""A from-scratch HNSW (Hierarchical Navigable Small World) index.

HNSW (Malkov & Yashunin, 2016) is the graph-based algorithm behind most
production vector databases. It builds a multi-layer proximity graph: upper
layers are sparse "express lanes" for coarse navigation, the bottom layer is
dense for fine-grained search. A query greedily descends the layers, then runs
a best-first beam search (controlled by ``ef``) at the bottom.

This implementation favors readability and correctness over raw speed (it is
pure Python + NumPy, not a SIMD C kernel). Its quality is measured, not
asserted — see ``benchmarks/bench.py`` for recall@k against exact search.

Key knobs:
    M               max neighbors per node per layer (bottom layer uses 2*M)
    ef_construction beam width while building (higher = better graph, slower)
    ef_search       beam width while querying (higher = better recall, slower)
"""

from __future__ import annotations

import heapq
import math
import random

import numpy as np

from .distance import Metric, distances, prepare


class HNSW:
    def __init__(
        self,
        dim: int,
        *,
        metric: Metric = "cosine",
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        seed: int = 42,
    ) -> None:
        if metric not in ("cosine", "l2"):
            raise ValueError(f"unknown metric {metric!r}")
        self.dim = dim
        self.metric = metric
        self.m = m
        self.m_max0 = 2 * m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._ml = 1.0 / math.log(m) if m > 1 else 1.0
        self._rng = random.Random(seed)
        # Counts distance computations; reset it to profile a single search.
        self.dist_calls = 0

        self._vectors = np.empty((0, dim), dtype=np.float32)
        self._ids: list = []
        # _graph[node][layer] -> set of neighbor node indices
        self._graph: list[list[set[int]]] = []
        self._levels: list[int] = []
        self._entry: int | None = None
        self._top: int = -1

    def __len__(self) -> int:
        return len(self._ids)

    # ---- distance helpers -------------------------------------------------
    def _dist_to_many(self, query: np.ndarray, idxs: np.ndarray) -> np.ndarray:
        self.dist_calls += len(idxs)
        return distances(query, self._vectors[idxs], self.metric)

    def _dist_pair(self, a: int, b: int) -> float:
        self.dist_calls += 1
        return float(distances(self._vectors[a], self._vectors[b][None, :], self.metric)[0])

    # ---- core graph search ------------------------------------------------
    def _search_layer(
        self, query: np.ndarray, entry_points: list[int], ef: int, layer: int
    ) -> list[tuple[float, int]]:
        """Best-first search within one layer. Returns a max-heap of (-dist, idx)."""
        visited = set(entry_points)
        ep = np.asarray(entry_points)
        ep_d = self._dist_to_many(query, ep)

        candidates: list[tuple[float, int]] = []  # min-heap by dist
        results: list[tuple[float, int]] = []  # max-heap by -dist (top = farthest)
        for d, node in zip(ep_d, entry_points, strict=False):
            heapq.heappush(candidates, (float(d), node))
            heapq.heappush(results, (-float(d), node))

        while candidates:
            d, c = heapq.heappop(candidates)
            farthest = -results[0][0]
            if d > farthest:
                break
            neighbors = [n for n in self._graph[c][layer] if n not in visited]
            if not neighbors:
                continue
            visited.update(neighbors)
            nd = self._dist_to_many(query, np.asarray(neighbors))
            for dist_e, e in zip(nd, neighbors, strict=False):
                dist_e = float(dist_e)
                farthest = -results[0][0]
                if dist_e < farthest or len(results) < ef:
                    heapq.heappush(candidates, (dist_e, e))
                    heapq.heappush(results, (-dist_e, e))
                    if len(results) > ef:
                        heapq.heappop(results)
        return results

    def _select_neighbors(
        self, candidates: list[tuple[float, int]], m: int
    ) -> list[int]:
        """HNSW neighbor-selection heuristic: prefer diverse neighbors.

        Accept a candidate only if it is closer to the query than to any
        already-selected neighbor; this keeps the graph navigable instead of
        clustering all edges in one direction. Falls back to nearest-first to
        fill ``m`` slots (keepPrunedConnections).
        """
        ordered = sorted(candidates)  # ascending by distance to query
        selected: list[tuple[float, int]] = []
        for d, e in ordered:
            if len(selected) >= m:
                break
            keep = True
            for _, s in selected:
                if self._dist_pair(e, s) < d:
                    keep = False
                    break
            if keep:
                selected.append((d, e))
        if len(selected) < m:
            chosen = {e for _, e in selected}
            for d, e in ordered:
                if len(selected) >= m:
                    break
                if e not in chosen:
                    selected.append((d, e))
                    chosen.add(e)
        return [e for _, e in selected]

    # ---- public API -------------------------------------------------------
    def add(self, vectors: np.ndarray, ids: list | None = None) -> None:
        vectors = prepare(np.atleast_2d(vectors), self.metric)
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape[1]}")
        if ids is None:
            ids = list(range(len(self._ids), len(self._ids) + vectors.shape[0]))
        if len(ids) != vectors.shape[0]:
            raise ValueError("len(ids) must match number of vectors")
        for vec, ext_id in zip(vectors, ids, strict=True):
            self._insert(vec, ext_id)

    def _insert(self, vec: np.ndarray, ext_id) -> None:
        node = len(self._ids)
        level = int(-math.log(max(self._rng.random(), 1e-12)) * self._ml)

        self._vectors = np.vstack([self._vectors, vec[None, :]])
        self._ids.append(ext_id)
        self._graph.append([set() for _ in range(level + 1)])
        self._levels.append(level)

        if self._entry is None:
            self._entry = node
            self._top = level
            return

        ep = [self._entry]
        # Greedy descent through layers above the new node's top level.
        for layer in range(self._top, level, -1):
            w = self._search_layer(vec, ep, ef=1, layer=layer)
            ep = [max(w)[1]]  # nearest (max -dist == smallest dist)

        for layer in range(min(self._top, level), -1, -1):
            w = self._search_layer(vec, ep, self.ef_construction, layer)
            cand = [(-negd, i) for negd, i in w]
            m_max = self.m_max0 if layer == 0 else self.m
            neighbors = self._select_neighbors(cand, m_max)
            for e in neighbors:
                self._graph[node][layer].add(e)
                self._graph[e][layer].add(node)
                if len(self._graph[e][layer]) > m_max:
                    econn = [(self._dist_pair(e, n), n) for n in self._graph[e][layer]]
                    self._graph[e][layer] = set(self._select_neighbors(econn, m_max))
            ep = [i for _, i in cand] or ep

        if level > self._top:
            self._entry = node
            self._top = level

    def search(self, query: np.ndarray, k: int = 10, ef: int | None = None) -> list[tuple]:
        """Return the ``k`` approximate nearest neighbors as ``(id, distance)``."""
        if self._entry is None:
            return []
        ef = ef or max(self.ef_search, k)
        q = prepare(np.atleast_2d(query), self.metric)[0]

        ep = [self._entry]
        for layer in range(self._top, 0, -1):
            w = self._search_layer(q, ep, ef=1, layer=layer)
            ep = [max(w)[1]]
        w = self._search_layer(q, ep, ef, layer=0)

        found = sorted((-negd, i) for negd, i in w)[:k]
        return [(self._ids[i], dist) for dist, i in found]

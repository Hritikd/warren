"""Save and load a warren HNSW index to a single ``.npz`` file."""

from __future__ import annotations

import json

import numpy as np

from .hnsw import HNSW


def save_index(index: HNSW, path: str) -> None:
    meta = {
        "dim": index.dim,
        "metric": index.metric,
        "m": index.m,
        "ef_construction": index.ef_construction,
        "ef_search": index.ef_search,
        "ids": index._ids,
        "levels": index._levels,
        "entry": index._entry,
        "top": index._top,
        # graph[node][layer] -> list of neighbor ids (sets are not JSON-able)
        "graph": [[sorted(layer) for layer in node] for node in index._graph],
    }
    np.savez_compressed(path, vectors=index._vectors, meta=np.array(json.dumps(meta)))


def load_index(path: str) -> HNSW:
    if not path.endswith(".npz"):
        path = path + ".npz"
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))

    index = HNSW(
        dim=meta["dim"],
        metric=meta["metric"],
        m=meta["m"],
        ef_construction=meta["ef_construction"],
        ef_search=meta["ef_search"],
    )
    index._vectors = data["vectors"].astype(np.float32)
    index._ids = meta["ids"]
    index._levels = meta["levels"]
    index._entry = meta["entry"]
    index._top = meta["top"]
    index._graph = [[set(layer) for layer in node] for node in meta["graph"]]
    return index

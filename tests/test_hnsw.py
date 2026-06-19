import numpy as np
import pytest

from warren import HNSW, BruteForceIndex


def _data(n, dim, seed=0, clusters=8):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 1, size=(clusters, dim))
    assign = rng.integers(0, clusters, size=n)
    return (centers[assign] + rng.normal(0, 0.3, size=(n, dim))).astype(np.float32)


def test_empty_search_returns_empty():
    assert HNSW(4).search(np.zeros(4), k=5) == []


def test_returns_k_results():
    idx = HNSW(16, seed=1)
    idx.add(_data(200, 16))
    assert len(idx.search(_data(1, 16, seed=99)[0], k=10)) == 10


def test_results_sorted_by_distance():
    idx = HNSW(16, seed=1)
    idx.add(_data(300, 16))
    res = idx.search(_data(1, 16, seed=5)[0], k=10)
    dists = [d for _, d in res]
    assert dists == sorted(dists)


def test_high_recall_against_bruteforce():
    """The whole point: HNSW must closely match exact search."""
    dim = 32
    data = _data(1000, dim, seed=3, clusters=15)
    ids = list(range(len(data)))

    exact = BruteForceIndex(dim, metric="cosine")
    exact.add(data, ids)
    ann = HNSW(dim, metric="cosine", m=16, ef_construction=200, seed=3)
    ann.add(data, ids)

    queries = _data(50, dim, seed=123, clusters=15)
    total = 0.0
    for q in queries:
        truth = {i for i, _ in exact.search(q, 10)}
        got = {i for i, _ in ann.search(q, 10, ef=100)}
        total += len(truth & got) / 10
    recall = total / len(queries)
    assert recall >= 0.90, f"recall too low: {recall:.3f}"


def test_determinism_same_seed():
    data = _data(200, 16, seed=2)
    q = _data(1, 16, seed=7)[0]
    a = HNSW(16, seed=11)
    a.add(data)
    b = HNSW(16, seed=11)
    b.add(data)
    assert a.search(q, 10) == b.search(q, 10)


def test_custom_ids_round_trip():
    idx = HNSW(8, seed=1)
    labels = [f"doc-{i}" for i in range(100)]
    idx.add(_data(100, 8), labels)
    res = idx.search(_data(1, 8, seed=4)[0], k=5)
    assert all(isinstance(i, str) and i.startswith("doc-") for i, _ in res)


def test_l2_metric_works():
    dim = 16
    data = _data(300, dim, seed=8)
    exact = BruteForceIndex(dim, metric="l2")
    exact.add(data, list(range(len(data))))
    ann = HNSW(dim, metric="l2", seed=8)
    ann.add(data, list(range(len(data))))
    q = _data(1, dim, seed=44)[0]
    truth = {i for i, _ in exact.search(q, 10)}
    got = {i for i, _ in ann.search(q, 10, ef=100)}
    assert len(truth & got) / 10 >= 0.8


def test_dim_mismatch_raises():
    with pytest.raises(ValueError):
        HNSW(8).add(np.zeros((1, 4)))


def test_bad_metric_raises():
    with pytest.raises(ValueError):
        HNSW(8, metric="manhattan")

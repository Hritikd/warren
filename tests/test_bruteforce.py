import numpy as np

from warren import BruteForceIndex


def test_returns_exact_nearest():
    data = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    idx = BruteForceIndex(2, metric="l2")
    idx.add(data, ["a", "b", "c"])
    res = idx.search(np.array([0.9, 0.0]), k=1)
    assert res[0][0] == "a"


def test_results_sorted_by_distance():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(50, 8)).astype(np.float32)
    idx = BruteForceIndex(8, metric="cosine")
    idx.add(data, list(range(50)))
    res = idx.search(rng.normal(size=8), k=10)
    dists = [d for _, d in res]
    assert dists == sorted(dists)


def test_k_larger_than_n():
    idx = BruteForceIndex(2, metric="l2")
    idx.add(np.array([[0.0, 0.0], [1.0, 1.0]]), [0, 1])
    assert len(idx.search(np.array([0.0, 0.0]), k=10)) == 2


def test_empty_index_returns_empty():
    assert BruteForceIndex(4).search(np.zeros(4), k=5) == []


def test_dim_mismatch_raises():
    idx = BruteForceIndex(3)
    try:
        idx.add(np.zeros((1, 4)), [0])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

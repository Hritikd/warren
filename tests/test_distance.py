import numpy as np

from warren.distance import distances, normalize, prepare


def test_normalize_unit_length():
    x = np.array([[3.0, 4.0], [1.0, 0.0]])
    n = normalize(x)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0)


def test_normalize_handles_zero_vector():
    n = normalize(np.array([[0.0, 0.0]]))
    assert np.all(np.isfinite(n))


def test_cosine_distance_identical_is_zero():
    a = prepare(np.array([[1.0, 2.0, 3.0]]), "cosine")[0]
    m = prepare(np.array([[1.0, 2.0, 3.0]]), "cosine")
    assert distances(a, m, "cosine")[0] < 1e-6


def test_cosine_distance_orthogonal_is_one():
    a = prepare(np.array([[1.0, 0.0]]), "cosine")[0]
    m = prepare(np.array([[0.0, 1.0]]), "cosine")
    assert abs(distances(a, m, "cosine")[0] - 1.0) < 1e-6


def test_l2_distance_matches_numpy():
    a = np.array([0.0, 0.0], dtype=np.float32)
    m = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    d = distances(a, m, "l2")
    assert abs(d[0] - 5.0) < 1e-5
    assert d[1] < 1e-6


def test_empty_matrix():
    a = np.array([1.0, 2.0], dtype=np.float32)
    assert distances(a, np.empty((0, 2), dtype=np.float32), "l2").shape == (0,)

import numpy as np

from warren import HNSW, load_index, save_index


def _data(n, dim, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, dim)).astype(np.float32)


def test_save_load_round_trip(tmp_path):
    dim = 16
    data = _data(200, dim, seed=1)
    idx = HNSW(dim, metric="cosine", m=8, seed=1)
    idx.add(data, [f"v{i}" for i in range(200)])

    path = str(tmp_path / "index")
    save_index(idx, path)
    loaded = load_index(path)

    q = _data(1, dim, seed=2)[0]
    assert idx.search(q, 10) == loaded.search(q, 10)


def test_loaded_index_preserves_params(tmp_path):
    idx = HNSW(8, metric="l2", m=12, ef_construction=150, ef_search=64, seed=3)
    idx.add(_data(50, 8, seed=3))
    path = str(tmp_path / "ix.npz")
    save_index(idx, path)
    loaded = load_index(path)
    assert (loaded.dim, loaded.metric, loaded.m) == (8, "l2", 12)
    assert (loaded.ef_construction, loaded.ef_search) == (150, 64)
    assert len(loaded) == 50

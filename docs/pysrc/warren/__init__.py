"""warren — a from-scratch HNSW approximate-nearest-neighbor index.

The graph algorithm behind modern vector databases, implemented in readable
pure Python + NumPy, with measured recall against exact search.

    >>> import numpy as np
    >>> from warren import HNSW
    >>> index = HNSW(dim=128, metric="cosine")
    >>> index.add(np.random.rand(1000, 128))
    >>> index.search(np.random.rand(128), k=10)   # -> [(id, distance), ...]
"""

from __future__ import annotations

from .bruteforce import BruteForceIndex
from .hnsw import HNSW
from .io import load_index, save_index

__all__ = ["HNSW", "BruteForceIndex", "save_index", "load_index"]
__version__ = "0.1.0"

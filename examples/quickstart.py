"""Minimal end-to-end example. Run: python examples/quickstart.py"""

import numpy as np

from warren import HNSW, BruteForceIndex

rng = np.random.default_rng(0)
dim = 64

# A toy "embedding" database of 2,000 vectors.
data = rng.normal(size=(2000, dim)).astype(np.float32)
ids = [f"doc-{i}" for i in range(2000)]

index = HNSW(dim, metric="cosine", m=16, ef_construction=200, seed=0)
index.add(data, ids)

query = rng.normal(size=dim).astype(np.float32)

approx = index.search(query, k=5, ef=100)
print("warren (approximate) top-5:")
for doc_id, dist in approx:
    print(f"  {doc_id:<10} dist={dist:.4f}")

# Compare against exact search to see the quality.
exact = BruteForceIndex(dim, metric="cosine")
exact.add(data, ids)
truth = {i for i, _ in exact.search(query, 5)}
got = {i for i, _ in approx}
print(f"\nrecall@5 vs exact search: {len(truth & got) / 5:.2f}")

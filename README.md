<h1 align="center">warren</h1>

<p align="center">
  <b>A from-scratch HNSW approximate-nearest-neighbor index.</b><br/>
  The graph algorithm behind modern vector databases, in readable pure Python + NumPy —
  <i>with recall measured against exact search, not asserted.</i>
</p>

<p align="center">
  <img alt="CI" src="https://github.com/Hritikd/warren/actions/workflows/ci.yml/badge.svg" />
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue.svg" />
  <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green.svg" />
  <img alt="recall@10: 0.99+" src="https://img.shields.io/badge/recall@10-0.99%2B-success.svg" />
</p>

<p align="center">
  <b><a href="https://hritikd.github.io/warren/">▶ Try the live demo</a></b> — move your cursor to query an HNSW index in your browser; watch recall vs exact search update live.
</p>

---

## The problem

Every RAG system, semantic search box, and recommendation feed has the same
core operation: given a query vector, find its nearest neighbors among millions
of stored vectors. Doing this *exactly* means comparing the query to every
vector — O(N) per query, which collapses at scale.

**HNSW** (Hierarchical Navigable Small World, Malkov & Yashunin 2016) is the
algorithm almost every vector database — FAISS, hnswlib, Qdrant, Weaviate,
pgvector — reaches for instead. It builds a layered proximity graph and *walks*
to the answer, touching a tiny fraction of the data per query.

**warren** is that algorithm implemented from first principles: the multi-layer
graph, the greedy descent, the `ef`-controlled beam search, and the
neighbor-selection heuristic — readable, dependency-light, and benchmarked for
recall so you can see it actually works.

## Install

```bash
pip install warren-ann
```

The only runtime dependency is NumPy.

## Quickstart

```python
import numpy as np
from warren import HNSW

index = HNSW(dim=128, metric="cosine", m=16, ef_construction=200)
index.add(np.random.rand(10_000, 128), ids=[f"doc-{i}" for i in range(10_000)])

index.search(np.random.rand(128), k=10, ef=100)
# -> [('doc-8123', 0.0419), ('doc-771', 0.0533), ...]   (id, distance)
```

## How it works

HNSW arranges vectors into a stack of graphs. The top layers are sparse — long
hops for coarse navigation. The bottom layer holds every vector with dense local
connections. A search **descends greedily** through the upper layers to land
near the query, then runs a **best-first beam search** at the bottom:

```
                 ●                       layer 2   (few nodes, long hops)
                / \
       ●───────●───●───────●            layer 1
      /|       |   |       |\
   ●──●──●──●──●───●──●──●──●──●         layer 0   (all nodes, dense)
              ↑ greedy walk → beam search (width = ef)
```

Two design choices from the paper that warren implements faithfully:

* **Exponential layer assignment** — a node's top layer is drawn from a
  geometric distribution (`level = ⌊-ln(U) · 1/ln(M)⌋`), so higher layers are
  exponentially sparser. This is what makes navigation logarithmic.
* **The neighbor-selection heuristic** — when connecting a node, a candidate is
  kept only if it is closer to the node than to any already-chosen neighbor.
  That keeps edges pointing in *diverse* directions instead of clustering, which
  is what keeps the graph navigable. (See `_select_neighbors` in `hnsw.py`.)

`ef` (search) and `ef_construction` (build) are the beam widths that trade
recall for speed. The whole index is seeded and deterministic.

## Does it actually work? (reproducible benchmark)

Built over 4,000 clustered vectors (synthetic Gaussian clusters, which resemble
real embedding distributions), measuring recall against exact brute-force search
on 200 held-out queries:

| ef_search | recall@10 | DB scanned / query | median latency |
|---|---|---|---|
| 10 | **0.988** | **4.5%** | 0.16 ms |
| 50 | **1.000** | 5.8% | 0.37 ms |
| 100 | **1.000** | 6.5% | 0.56 ms |

```bash
python benchmarks/bench.py        # reproduce this table (deterministic)
```

**Reading this honestly:**

* The headline is **recall vs. scan fraction**: warren returns essentially the
  exact neighbors (0.99–1.00 recall@10) while computing the distance to only
  **~5% of the database** per query. That sublinear scan is HNSW's whole point,
  and it's the metric that doesn't depend on the implementation language.
* I deliberately **do not** claim a wall-clock speedup. At these sizes a NumPy
  brute-force search is a single BLAS matmul and is genuinely fast; pure-Python
  graph traversal can't beat it on the clock. The scan fraction is what a
  C/SIMD implementation (hnswlib, FAISS) converts into a real latency win at
  millions of vectors. warren is the *algorithm*, not a FAISS replacement.

## API

```python
HNSW(dim, *, metric="cosine"|"l2", m=16, ef_construction=200, ef_search=50, seed=42)
  .add(vectors, ids=None)              # ids default to 0..n
  .search(query, k=10, ef=None)        # -> [(id, distance), ...]

BruteForceIndex(dim, metric)           # exact search / ground truth
save_index(index, path); load_index(path)   # single-file .npz persistence
```

## Persistence

```python
from warren import save_index, load_index
save_index(index, "my_index")          # -> my_index.npz
index = load_index("my_index.npz")     # graph + vectors restored exactly
```

## Design principles

```text
Correct first, measured always   recall is benchmarked, not assumed
Readable over clever             the algorithm should be legible in the source
Honest about scope               no false speedup claims; scan fraction is the win
Deterministic                    seeded build -> identical graph and results
```

## Limitations & scope

* **Pure Python + NumPy**, so it is not a speed competitor to `hnswlib`/FAISS.
  Use it to *understand* HNSW, for small/medium datasets, or where native
  extensions can't be installed — not to serve a billion vectors.
* Inserts are incremental; there is no deletion or rebalancing yet.
* The benchmark uses synthetic clustered data; recall on your own embeddings
  will vary with their intrinsic dimensionality — reproduce it with your data.

## Development

```bash
make install   # pip install -e ".[dev]"
make test      # pytest  (23 tests, incl. a recall guard-rail)
make lint      # ruff
make bench     # regenerate benchmarks/results.md
```

## License

MIT © Hritik Datta

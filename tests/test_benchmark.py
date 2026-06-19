"""Guard-rail: the benchmark must keep delivering high recall and a real speedup.

If a change quietly degrades the index, CI fails here instead of the README
quietly becoming a lie.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import bench  # noqa: E402


def test_recall_and_speedup():
    # Small, fast config for CI — the published numbers use bench defaults.
    result = bench.run(n=1000, n_queries=50, ef_construction=80)
    # Highest ef should achieve strong recall.
    best_ef = max(result["rows"])
    assert result["rows"][best_ef]["recall"] >= 0.90

    # More search effort should not reduce recall (monotonic-ish).
    efs = sorted(result["rows"])
    recalls = [result["rows"][ef]["recall"] for ef in efs]
    assert recalls[-1] >= recalls[0]

    # HNSW's algorithmic win: it touches only a fraction of the database.
    assert result["rows"][min(efs)]["scan_pct"] < 50.0

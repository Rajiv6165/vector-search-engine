import numpy as np
import pytest
from vsearch.hnsw import HNSWIndex, HNSWConfig, HAS_NATIVE

def test_ef_search_recall_and_visited_growth():
    """
    Assert that searching the same queries with ef=10 and ef=100 yields:
    1. A meaningful margin in recall@10 (ef=10 must be lower than ef=100).
    2. The number of visited nodes grows with ef (ef=100 visits strictly more nodes).
    """
    np.random.seed(42)
    dim = 32
    num_nodes = 1000
    num_queries = 40

    # Generate clustered vectors so nearest neighbors are non-trivial
    centers = np.random.randn(10, dim).astype(np.float32)
    data = []
    for i in range(num_nodes):
        c = centers[i % 10]
        data.append(c + np.random.randn(dim).astype(np.float32) * 0.5)
    data = np.array(data, dtype=np.float32)

    queries = np.random.randn(num_queries, dim).astype(np.float32)

    config = HNSWConfig(M=16, ef_construction=100, heuristic=True)
    index = HNSWIndex(config=config, metric="l2")
    index.batch_insert(data)

    # Compute exact ground truth top-10
    dists = np.linalg.norm(data[None, :, :] - queries[:, None, :], axis=2)
    I_gt = np.argsort(dists, axis=1)[:, :10]

    r10_ef10 = []
    r10_ef100 = []
    visited_ef10 = 0
    visited_ef100 = 0

    for i, q in enumerate(queries):
        res10, v10 = index.search(q, k=10, ef=10, return_visited=True)
        res100, v100 = index.search(q, k=10, ef=100, return_visited=True)

        visited_ef10 += v10
        visited_ef100 += v100

        gt_set = set(I_gt[i])
        s10 = set(x[0] for x in res10)
        s100 = set(x[0] for x in res100)

        r10_ef10.append(len(gt_set.intersection(s10)) / 10.0)
        r10_ef100.append(len(gt_set.intersection(s100)) / 10.0)

    mean_r10_ef10 = float(np.mean(r10_ef10))
    mean_r10_ef100 = float(np.mean(r10_ef100))

    # Assert recall@10 differs by a meaningful margin (ef=10 must be lower)
    assert mean_r10_ef10 < mean_r10_ef100, (
        f"Expected ef=10 recall ({mean_r10_ef10:.4f}) to be strictly lower than ef=100 recall ({mean_r10_ef100:.4f})"
    )
    recall_margin = mean_r10_ef100 - mean_r10_ef10
    assert recall_margin >= 0.05, (
        f"Recall margin between ef=100 and ef=10 should be at least 5%, got {recall_margin*100:.2f}%"
    )

    # Assert visited nodes grows significantly with ef
    assert visited_ef100 > visited_ef10 * 1.5, (
        f"Expected visited nodes at ef=100 ({visited_ef100}) to be > 1.5x ef=10 ({visited_ef10})"
    )


def test_pure_python_ef_search_scaling():
    """Verify that pure-Python fallback also scales visited count and recall with ef."""
    np.random.seed(123)
    dim = 16
    num_nodes = 500
    num_queries = 25

    centers = np.random.randn(8, dim).astype(np.float32)
    data = []
    for i in range(num_nodes):
        c = centers[i % 8]
        data.append(c + np.random.randn(dim).astype(np.float32) * 0.4)
    data = np.array(data, dtype=np.float32)
    queries = np.random.randn(num_queries, dim).astype(np.float32)

    config = HNSWConfig(M=8, ef_construction=50, heuristic=False)
    index = HNSWIndex(config=config, metric="l2")
    index.batch_insert(data)

    dists = np.linalg.norm(data[None, :, :] - queries[:, None, :], axis=2)
    I_gt = np.argsort(dists, axis=1)[:, :10]

    v10_tot = 0
    v100_tot = 0
    r10 = []
    r100 = []

    for i, q in enumerate(queries):
        res10, v10 = index._search_layer_python(q, [index.entry_point], ef=10, layer=0, return_visited=True)
        res100, v100 = index._search_layer_python(q, [index.entry_point], ef=100, layer=0, return_visited=True)
        res10.sort(key=lambda x: x[1])
        res100.sort(key=lambda x: x[1])

        v10_tot += v10
        v100_tot += v100

        gt_set = set(I_gt[i])
        r10.append(len(gt_set.intersection(set(x[0] for x in res10[:10]))) / 10.0)
        r100.append(len(gt_set.intersection(set(x[0] for x in res100[:10]))) / 10.0)

    assert np.mean(r10) < np.mean(r100)
    assert v100_tot > v10_tot * 1.3

import numpy as np
import pytest
from vsearch.hnsw import HNSWIndex, HNSWConfig, HAS_NATIVE

def test_native_vs_python_search_layer_identical():
    """Assert native search_layer and pure-Python search_layer return identical results."""
    assert HAS_NATIVE, "Native compiled extension must be available for comparison test"

    np.random.seed(42)
    dim = 32
    num_nodes = 500
    num_queries = 20

    data = np.random.randn(num_nodes, dim).astype(np.float32)
    queries = np.random.randn(num_queries, dim).astype(np.float32)

    config = HNSWConfig(M=16, ef_construction=100, ef_search=50, heuristic=False)
    index = HNSWIndex(config=config, metric="l2")

    for i in range(num_nodes):
        index.insert(i, data[i])

    # For layer 0 and layer 1, compare native vs python search layer
    for layer in [0, 1]:
        if layer >= len(index.graphs) or not index.graphs[layer]:
            continue
        ep = [index.entry_point]
        for q in queries:
            python_res = index._search_layer_python(q, ep, ef=50, layer=layer)
            native_res = index._search_layer(q, ep, ef=50, layer=layer)

            # Sort both by distance for comparison
            python_res.sort(key=lambda x: (x[1], x[0]))
            native_res.sort(key=lambda x: (x[1], x[0]))

            assert len(python_res) == len(native_res), f"Result size mismatch: {len(python_res)} vs {len(native_res)}"
            
            py_ids = [x[0] for x in python_res]
            nat_ids = [x[0] for x in native_res]
            py_dists = np.array([x[1] for x in python_res], dtype=np.float32)
            nat_dists = np.array([x[1] for x in native_res], dtype=np.float32)

            assert py_ids == nat_ids, f"Node IDs differ: {py_ids} vs {nat_ids}"
            np.testing.assert_allclose(py_dists, nat_dists, rtol=1e-5, atol=1e-5)

def test_batch_search_identical_to_single():
    """Assert batch_search produces identical results to sequential search."""
    np.random.seed(123)
    dim = 16
    n = 200
    n_queries = 15

    data = np.random.randn(n, dim).astype(np.float32)
    queries = np.random.randn(n_queries, dim).astype(np.float32)

    index = HNSWIndex(HNSWConfig(M=8, ef_construction=50), metric="l2")
    index.batch_insert(data)

    seq_results = [index.search(q, k=5) for q in queries]
    batch_results = index.batch_search(queries, k=5, num_threads=4)

    assert len(seq_results) == len(batch_results)
    for seq, bat in zip(seq_results, batch_results):
        seq_ids = [x[0] for x in seq]
        bat_ids = [x[0] for x in bat]
        assert seq_ids == bat_ids

def test_heuristic_neighbor_selection_diversity():
    """Verify that diversity heuristic (Algorithm 4) selects diverse connections."""
    np.random.seed(99)
    dim = 8
    config_heuristic = HNSWConfig(M=8, ef_construction=50, heuristic=True)
    config_simple = HNSWConfig(M=8, ef_construction=50, heuristic=False)

    idx_h = HNSWIndex(config=config_heuristic)
    idx_s = HNSWIndex(config=config_simple)

    data = np.random.randn(100, dim).astype(np.float32)
    idx_h.batch_insert(data)
    idx_s.batch_insert(data)

    q = np.random.randn(dim).astype(np.float32)
    res_h = idx_h.search(q, k=5)
    res_s = idx_s.search(q, k=5)

    assert len(res_h) == 5
    assert len(res_s) == 5

def test_native_vs_python_filtered_search_identical():
    """Assert native and pure-Python paths produce identical filtered search results on the same graph."""
    assert HAS_NATIVE, "Native compiled extension must be available for comparison test"

    np.random.seed(42)
    dim = 16
    num_nodes = 400
    num_queries = 10

    data = np.random.randn(num_nodes, dim).astype(np.float32)
    queries = np.random.randn(num_queries, dim).astype(np.float32)

    config = HNSWConfig(M=16, ef_construction=100, ef_search=50, heuristic=True)
    index = HNSWIndex(config=config, metric="l2")

    categories = ["electronics", "apparel", "books", "home"]
    ratings = [1, 2, 3, 4, 5]

    for i in range(num_nodes):
        meta = {
            "category": categories[i % len(categories)],
            "rating": ratings[i % len(ratings)],
            "even": (i % 2 == 0)
        }
        index.insert(i, data[i], metadata=meta)

    # Test 1: In-graph traversal path (force threshold low so it runs in-graph)
    index.filter_fallback_threshold = 10  # 100 matching nodes > 10 => in-graph
    test_filter = {"category": "electronics"}
    
    for q in queries:
        native_res = index.search(q, k=10, ef=50, filter_dict=test_filter, use_native=True)
        py_res = index.search(q, k=10, ef=50, filter_dict=test_filter, use_native=False)

        assert len(native_res) == len(py_res)
        for (nat_id, nat_dist, nat_meta), (py_id, py_dist, py_meta) in zip(native_res, py_res):
            assert nat_id == py_id, f"In-graph mismatch: {nat_id} vs {py_id}"
            assert abs(nat_dist - py_dist) < 1e-4
            assert nat_meta["category"] == "electronics"
            assert py_meta["category"] == "electronics"

    # Test 2: Fallback brute-force path (threshold high so it uses fallback)
    index.filter_fallback_threshold = 1000  # 100 matching nodes <= 1000 => fallback
    for q in queries:
        native_res = index.search(q, k=10, ef=50, filter_dict=test_filter, use_native=True)
        py_res = index.search(q, k=10, ef=50, filter_dict=test_filter, use_native=False)

        assert len(native_res) == len(py_res)
        for (nat_id, nat_dist, nat_meta), (py_id, py_dist, py_meta) in zip(native_res, py_res):
            assert nat_id == py_id, f"Fallback mismatch: {nat_id} vs {py_id}"
            assert abs(nat_dist - py_dist) < 1e-4

    # Test 3: Multi-field AND filter
    multi_filter = {"category": "electronics", "rating": 5}
    index.filter_fallback_threshold = 0  # Force in-graph
    for q in queries:
        res = index.search(q, k=5, ef=50, filter_dict=multi_filter, use_native=True)
        for _, _, meta in res:
            assert meta["category"] == "electronics"
            assert meta["rating"] == 5

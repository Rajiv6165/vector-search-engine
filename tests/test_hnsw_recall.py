import pytest
import numpy as np
from vsearch.hnsw import HNSWIndex, HNSWConfig
from vsearch.brute_force import ExactKNN

def test_recall():
    """Asserts >85% recall against brute-force on random vectors."""
    np.random.seed(42)
    dim = 64
    num_elements = 2000
    num_queries = 50
    k = 10
    
    # Generate random data
    data = np.random.randn(num_elements, dim).astype(np.float32)
    queries = np.random.randn(num_queries, dim).astype(np.float32)
    
    # Initialize indexes
    config = HNSWConfig(M=16, ef_construction=100, ef_search=50)
    hnsw = HNSWIndex(config=config, metric="l2")
    bf = ExactKNN(metric="l2")
    
    # Insert data
    for i in range(num_elements):
        hnsw.insert(i, data[i])
        bf.add(i, data[i])
        
    recall_sum = 0.0
    for q in queries:
        hnsw_results = hnsw.search(q, k=k)
        bf_results = bf.search(q, k=k)
        
        hnsw_ids = {n_id for n_id, _, _ in hnsw_results}
        bf_ids = {n_id for n_id, _ in bf_results}
        
        intersection = len(hnsw_ids.intersection(bf_ids))
        recall_sum += intersection / float(k)
        
    avg_recall = recall_sum / num_queries
    print(f"Average Recall@{k}: {avg_recall:.4f}")
    assert avg_recall > 0.90, f"Recall {avg_recall} is below the strict 90% threshold."

def test_insert_delete_consistency():
    """Validates graph integrity after deletions."""
    np.random.seed(42)
    dim = 16
    
    hnsw = HNSWIndex(metric="l2")
    
    # Insert 100
    for i in range(100):
        hnsw.insert(i, np.random.randn(dim))
        
    # Delete some internal nodes and the entry point
    nodes_to_delete = [50, 25, 75, hnsw.entry_point]
    
    for n_id in nodes_to_delete:
        if n_id in hnsw.nodes:
            hnsw.delete(n_id)
            
    # Verify no dangling references
    for lc in range(len(hnsw.graphs)):
        for node, neighbors in hnsw.graphs[lc].items():
            assert node in hnsw.nodes, f"Node {node} in graph but not in nodes list"
            for n_id in neighbors:
                assert n_id in hnsw.nodes, f"Dangling reference to deleted node {n_id}"
                assert n_id != node, f"Self-loop detected on node {node}"
                
    # Ensure search still works and doesn't crash
    res = hnsw.search(np.random.randn(dim), k=5)
    assert len(res) == 5

def test_edge_cases():
    """Tests empty index, single vector, k > len(index), and exact duplicate points."""
    hnsw = HNSWIndex(metric="l2")
    dim = 8
    
    # 1. Empty index search
    res = hnsw.search(np.random.randn(dim), k=5)
    assert len(res) == 0
    
    # 2. Single vector index
    v1 = np.random.randn(dim)
    hnsw.insert(0, v1)
    res = hnsw.search(np.random.randn(dim), k=5)
    assert len(res) == 1
    assert res[0][0] == 0
    
    # 3. k > len(index)
    hnsw.insert(1, np.random.randn(dim))
    res = hnsw.search(np.random.randn(dim), k=10)
    assert len(res) == 2
    
    # 4. Duplicate points (should insert fine, distance = 0)
    v_dup = v1.copy()
    hnsw.insert(2, v_dup)
    res = hnsw.search(v1, k=5)
    # The two closest should be 0 and 2 (both distance 0 from v1)
    dists = {d for n_id, d, _ in res if n_id in [0, 2]}
    for d in dists:
        assert np.isclose(d, 0.0, atol=1e-5)
    assert len(res) == 3

def test_filtered_recall_regression():
    """Regression test: filtered search achieves >= 90% recall at 50% and 10% selectivity on 5,000 vectors."""
    np.random.seed(123)
    num_vectors = 5000
    dim = 64
    k = 10
    num_queries = 40

    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    queries = np.random.randn(num_queries, dim).astype(np.float32)

    config = HNSWConfig(M=16, ef_construction=100, ef_search=50)
    index = HNSWIndex(config=config, metric="l2")

    # 50% category ('A' vs 'B') and 10% category ('val_0' through 'val_9')
    metadatas = []
    for i in range(num_vectors):
        meta = {
            "is_even": (i % 2 == 0),         # 50% selectivity
            "cluster_10": f"c_{i % 10}"      # 10% selectivity
        }
        metadatas.append(meta)

    index.batch_insert(vectors, node_ids=list(range(num_vectors)), metadatas=metadatas)

    # 1. Test 50% selectivity (is_even == True)
    matching_50 = np.array([i for i in range(num_vectors) if metadatas[i]["is_even"] is True])
    vecs_50 = vectors[matching_50]

    recalls_50 = []
    for q in queries:
        hnsw_res = index.search(q, k=k, filter_dict={"is_even": True})
        hnsw_ids = {n_id for n_id, _, _ in hnsw_res}

        # Brute force over matching
        dists = np.sum((vecs_50 - q) ** 2, axis=1)
        top_k_idx = np.argsort(dists)[:k]
        gt_ids = set(matching_50[top_k_idx])

        recalls_50.append(len(hnsw_ids.intersection(gt_ids)) / float(k))

    avg_recall_50 = float(np.mean(recalls_50))
    print(f"Filtered Recall @ 50% selectivity: {avg_recall_50:.4f}")
    assert avg_recall_50 >= 0.90, f"50% selectivity recall {avg_recall_50:.4f} fell below 0.90 threshold"

    # 2. Test 10% selectivity (cluster_10 == 'c_0')
    matching_10 = np.array([i for i in range(num_vectors) if metadatas[i]["cluster_10"] == "c_0"])
    vecs_10 = vectors[matching_10]

    recalls_10 = []
    for q in queries:
        hnsw_res = index.search(q, k=k, filter_dict={"cluster_10": "c_0"})
        hnsw_ids = {n_id for n_id, _, _ in hnsw_res}

        # Brute force over matching
        dists = np.sum((vecs_10 - q) ** 2, axis=1)
        top_k_idx = np.argsort(dists)[:k]
        gt_ids = set(matching_10[top_k_idx])

        recalls_10.append(len(hnsw_ids.intersection(gt_ids)) / float(k))

    avg_recall_10 = float(np.mean(recalls_10))
    print(f"Filtered Recall @ 10% selectivity: {avg_recall_10:.4f}")
    assert avg_recall_10 >= 0.90, f"10% selectivity recall {avg_recall_10:.4f} fell below 0.90 threshold"


import threading
import time
import random
import numpy as np
import pytest

from vsearch.hnsw import HNSWIndex, HNSWConfig

def test_concurrency_mixed_workload(tmp_path):
    config = HNSWConfig(M=16, ef_construction=50, ef_search=50)
    index = HNSWIndex(config=config, persist_dir=str(tmp_path))
    
    # Pre-populate index with some nodes
    num_initial = 100
    dim = 16
    for i in range(num_initial):
        vec = np.random.rand(dim).astype(np.float32)
        index.insert(i, vec)
        
    num_threads = 10
    ops_per_thread = 50
    exceptions = []
    
    def worker(thread_id):
        try:
            for i in range(ops_per_thread):
                op = random.choice(["search", "insert", "delete", "search"])
                node_id = num_initial + thread_id * ops_per_thread + i
                vec = np.random.rand(dim).astype(np.float32)
                
                if op == "search":
                    # Randomly pick an existing or new vector to search
                    res = index.search(vec, k=5)
                    assert isinstance(res, list)
                elif op == "insert":
                    # Some vectors may be inserted twice if random overlaps (handled in stress test, here unique)
                    index.insert(node_id, vec)
                elif op == "delete":
                    # Try to delete a random node
                    target = random.randint(0, num_initial - 1)
                    index.delete(target)
        except Exception as e:
            exceptions.append(e)

    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    assert not exceptions, f"Exceptions occurred during concurrent execution: {exceptions}"
    
    # Verify graph state is consistent
    # Just checking it doesn't crash on full search
    for i in range(10):
        vec = np.random.rand(dim).astype(np.float32)
        index.search(vec, k=10)

def test_concurrency_stress_overlap(tmp_path):
    config = HNSWConfig(M=8, ef_construction=30, ef_search=30)
    index = HNSWIndex(config=config, persist_dir=str(tmp_path))
    
    dim = 8
    num_threads = 8
    ops_per_thread = 100
    exceptions = []
    
    def worker():
        try:
            for _ in range(ops_per_thread):
                # High contention on a small set of node IDs
                node_id = random.randint(0, 20)
                vec = np.random.rand(dim).astype(np.float32)
                
                op = random.choice(["insert", "delete"])
                if op == "insert":
                    try:
                        index.insert(node_id, vec)
                    except ValueError:
                        # Expected if node already exists
                        pass
                else:
                    index.delete(node_id)
        except Exception as e:
            exceptions.append(e)

    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    assert not exceptions, f"Exceptions occurred during overlap stress test: {exceptions}"

def test_concurrency_filtered_search(tmp_path):
    """Verify concurrent inserts, deletes, and filtered searches under RWLock."""
    config = HNSWConfig(M=16, ef_construction=50, ef_search=50)
    index = HNSWIndex(config=config, persist_dir=str(tmp_path))

    dim = 16
    num_initial = 150
    departments = ["engineering", "sales", "finance", "hr"]

    for i in range(num_initial):
        vec = np.random.randn(dim).astype(np.float32)
        meta = {
            "dept": departments[i % len(departments)],
            "active": (i % 2 == 0)
        }
        index.insert(i, vec, metadata=meta)

    # Pre-delete a set of nodes to confirm they are never returned by any concurrent search
    permanently_deleted = {10, 20, 30, 40, 50}
    for d_id in permanently_deleted:
        index.delete(d_id)

    num_threads = 8
    ops_per_thread = 50
    exceptions = []

    def worker(thread_id):
        try:
            for i in range(ops_per_thread):
                op = random.choice(["search", "insert", "delete", "search"])
                vec = np.random.randn(dim).astype(np.float32)

                if op == "search":
                    dept = random.choice(departments)
                    filter_dict = {"dept": dept}
                    results = index.search(vec, k=5, filter_dict=filter_dict)
                    assert isinstance(results, list)
                    for n_id, dist, meta in results:
                        assert n_id not in permanently_deleted, f"Permanently deleted node {n_id} returned in search"
                        assert meta.get("dept") == dept, f"Filter violated: expected {dept}, got {meta.get('dept')}"
                elif op == "insert":
                    new_id = num_initial + thread_id * ops_per_thread + i
                    meta = {
                        "dept": random.choice(departments),
                        "active": random.choice([True, False])
                    }
                    index.insert(new_id, vec, metadata=meta)
                elif op == "delete":
                    # Delete dynamic nodes (excluding permanently_deleted check targets)
                    target = random.randint(60, num_initial - 1)
                    index.delete(target)
        except Exception as e:
            exceptions.append(e)

    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert not exceptions, f"Exceptions occurred during concurrent filtered execution: {exceptions}"

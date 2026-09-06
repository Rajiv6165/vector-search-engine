import threading
import time
import random
import numpy as np
import pytest

from src.vsearch.hnsw import HNSWIndex, HNSWConfig

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

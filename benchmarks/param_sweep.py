import time
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from vsearch.hnsw import HNSWIndex, HNSWConfig
from vsearch.brute_force import ExactKNN

def measure_footprint(hnsw: HNSWIndex) -> int:
    """Estimates the memory footprint proxy by counting total directed edges."""
    total_edges = 0
    for layer in hnsw.graphs:
        for neighbors in layer.values():
            total_edges += len(neighbors)
    return total_edges

def run_sweep():
    np.random.seed(42)
    dim = 64
    num_elements = 2000
    num_queries = 200
    k = 10

    data = np.random.randn(num_elements, dim).astype(np.float32)
    queries = np.random.randn(num_queries, dim).astype(np.float32)

    # Brute force baseline for recall
    bf = ExactKNN(metric="l2")
    for i in range(num_elements):
        bf.add(i, data[i])
        
    print("Computing brute-force baseline...")
    bf_results = []
    for q in queries:
        bf_results.append({n_id for n_id, _ in bf.search(q, k=k)})

    Ms = [8, 16, 32]
    ef_constructions = [100, 200, 400]
    ef_searches = [50, 100, 200]
    
    results = []

    print(f"{'M':>4} | {'ef_con':>6} | {'ef_sch':>6} | {'Recall':>7} | {'Insert(s)':>9} | {'Search(ms)':>10} | {'Edges':>8}")
    print("-" * 65)

    for M in Ms:
        for ef_c in ef_constructions:
            config = HNSWConfig(M=M, ef_construction=ef_c)
            hnsw = HNSWIndex(config=config, metric="l2")
            
            start_insert = time.time()
            for i in range(num_elements):
                hnsw.insert(i, data[i])
            insert_time = time.time() - start_insert
            
            edges = measure_footprint(hnsw)
            
            for ef_s in ef_searches:
                start_search = time.time()
                recall_sum = 0.0
                
                for i, q in enumerate(queries):
                    res = hnsw.search(q, k=k, ef=ef_s)
                    hnsw_ids = {n_id for n_id, _ in res}
                    intersection = len(hnsw_ids.intersection(bf_results[i]))
                    recall_sum += intersection / float(k)
                    
                search_time = time.time() - start_search
                avg_recall = recall_sum / num_queries
                avg_search_ms = (search_time / num_queries) * 1000
                
                print(f"{M:>4} | {ef_c:>6} | {ef_s:>6} | {avg_recall:>7.4f} | {insert_time:>9.3f} | {avg_search_ms:>10.3f} | {edges:>8}")
                results.append((M, ef_c, ef_s, avg_recall, insert_time, avg_search_ms, edges))
                
if __name__ == "__main__":
    run_sweep()

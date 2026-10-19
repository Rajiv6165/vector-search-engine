import cProfile
import pstats
import io
import time
import os
import sys
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from vsearch.hnsw import HNSWIndex, HNSWConfig

SIFT_BASE = os.path.join(os.path.dirname(__file__), "data", "sift", "sift_base.fvecs")
SIFT_QUERY = os.path.join(os.path.dirname(__file__), "data", "sift", "sift_query.fvecs")

def load_fvecs(filename, n_max):
    return np.fromfile(filename, dtype=np.float32, count=n_max * 129).reshape(n_max, 129)[:, 1:].copy()

def main():
    print("Loading SIFT data...")
    # Load 10,000 base vectors to profile insert on realistic graph size,
    # and 500 queries to profile search
    N_INDEX = 10_000
    N_SEARCH_INDEX = 10_000
    N_QUERIES = 500
    
    base_vecs = load_fvecs(SIFT_BASE, N_INDEX)
    query_vecs = load_fvecs(SIFT_QUERY, N_QUERIES)
    print(f"Loaded {len(base_vecs)} base vectors and {len(query_vecs)} query vectors.")
    
    config = HNSWConfig(M=16, ef_construction=100, ef_search=50)
    index = HNSWIndex(config=config, metric="l2")
    
    # --- Profile INSERT ---
    print(f"\n--- Profiling insert() on {N_INDEX} SIFT vectors ---")
    pr_insert = cProfile.Profile()
    t0 = time.time()
    pr_insert.enable()
    for i in range(N_INDEX):
        index.insert(i, base_vecs[i])
    pr_insert.disable()
    insert_duration = time.time() - t0
    print(f"Total insert time for {N_INDEX} vectors: {insert_duration:.3f} s ({N_INDEX/insert_duration:.1f} inserts/s)")
    
    s_insert = pstats.Stats(pr_insert).sort_stats("tottime")
    print("\n[Top 20 Functions by tottime - INSERT]:")
    s_insert.print_stats(20)
    
    # --- Profile SEARCH ---
    print(f"\n--- Profiling search() on {N_QUERIES} queries (index size={len(index.nodes)}, ef={config.ef_search}) ---")
    # Warmup
    for i in range(20):
        index.search(query_vecs[i], k=10)
        
    pr_search = cProfile.Profile()
    t0 = time.time()
    pr_search.enable()
    for q in query_vecs:
        index.search(q, k=10)
    pr_search.disable()
    search_duration = time.time() - t0
    print(f"Total search time for {N_QUERIES} queries: {search_duration:.3f} s ({N_QUERIES/search_duration:.1f} QPS)")
    
    s_search = pstats.Stats(pr_search).sort_stats("tottime")
    print("\n[Top 20 Functions by tottime - SEARCH]:")
    s_search.print_stats(20)

if __name__ == "__main__":
    main()

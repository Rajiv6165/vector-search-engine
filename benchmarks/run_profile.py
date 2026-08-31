import cProfile
import pstats
import time
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from vsearch.hnsw import HNSWIndex, HNSWConfig

def profile_hnsw():
    np.random.seed(42)
    dim = 64
    num_elements = 5000
    num_queries = 500
    k = 10

    data = np.random.randn(num_elements, dim).astype(np.float32)
    queries = np.random.randn(num_queries, dim).astype(np.float32)

    config = HNSWConfig(M=16, ef_construction=100, ef_search=50)
    hnsw = HNSWIndex(config=config, metric="l2")

    print(f"Profiling insert() for {num_elements} vectors...")
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time.time()
    for i in range(num_elements):
        hnsw.insert(i, data[i])
    insert_time = time.time() - start_time
    
    profiler.disable()
    
    print(f"Total insert time: {insert_time:.4f} seconds")
    print("--- Insert Profile Top 15 ---")
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(15)

    print(f"\nProfiling search() for {num_queries} queries...")
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time.time()
    for q in queries:
        hnsw.search(q, k=k)
    search_time = time.time() - start_time
    
    profiler.disable()
    
    print(f"Total search time: {search_time:.4f} seconds")
    print("--- Search Profile Top 15 ---")
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(15)

if __name__ == "__main__":
    profile_hnsw()

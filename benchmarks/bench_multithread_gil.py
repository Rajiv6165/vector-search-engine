import os
import sys
import time
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from vsearch.hnsw import HNSWIndex, HNSWConfig

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "sift")
BASE_FILE = os.path.join(DATA_DIR, "sift_base.fvecs")
QUERY_FILE = os.path.join(DATA_DIR, "sift_query.fvecs")

def load_fvecs(filename, n_max):
    return np.fromfile(filename, dtype=np.float32, count=n_max * 129).reshape(n_max, 129)[:, 1:].copy()

def main():
    N_VECS = 20_000
    N_QUERIES = 2_000
    print(f"Loading {N_VECS} vectors and {N_QUERIES} queries...")
    base_vecs = load_fvecs(BASE_FILE, N_VECS)
    # Tile queries to have 2000 queries for statistically reliable timing
    query_vecs = load_fvecs(QUERY_FILE, 1000)
    query_vecs = np.vstack([query_vecs, query_vecs])

    print("Building HNSW index...")
    config = HNSWConfig(M=16, ef_construction=100, ef_search=50, heuristic=True)
    idx = HNSWIndex(config=config, metric="l2")
    idx.batch_insert(base_vecs)
    print("Index built.")

    thread_counts = [1, 2, 4, 8]
    results = {}

    print("\n--- Measuring Batch Search QPS Across Thread Counts (GIL Release) ---")
    for threads in thread_counts:
        # Warmup
        idx.batch_search(query_vecs[:100], k=10, num_threads=threads)

        latencies = []
        # Repeat 3 times for mean & stddev
        runs = []
        for _ in range(3):
            t0 = time.perf_counter()
            idx.batch_search(query_vecs, k=10, num_threads=threads)
            t1 = time.perf_counter()
            runs.append(len(query_vecs) / (t1 - t0))

        mean_qps = float(np.mean(runs))
        std_qps = float(np.std(runs))
        speedup = mean_qps / results.get(1, (mean_qps, 0))[0]
        results[threads] = (mean_qps, std_qps)
        print(f"Threads: {threads:<2} | QPS: {mean_qps:8.1f} ± {std_qps:5.1f} | Speedup vs 1T: {speedup:.2f}x")

    print("\n" + "="*55)
    print("PHASE 5 vs PHASE 9 GIL SCALING COMPARISON")
    print("="*55)
    print("Phase 5 Finding: GIL bottleneck capped multi-thread scaling.")
    print("Phase 9 Result:")
    t1_qps = results[1][0]
    for t in thread_counts:
        qps, std = results[t]
        print(f"  {t} Thread{'s' if t>1 else ' '}: {qps:8.1f} QPS ({qps/t1_qps:.2f}x speedup)")
    print("="*55)

if __name__ == "__main__":
    main()

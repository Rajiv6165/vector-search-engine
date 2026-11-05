import os
import sys
import time
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from vsearch.hnsw import HNSWIndex
from bench_vs_faiss import load_fvecs

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BASE_VECTORS_FILE = os.path.join(DATA_DIR, "sift", "sift_base.fvecs")
QUERY_VECTORS_FILE = os.path.join(DATA_DIR, "sift", "sift_query.fvecs")

def profile_brute_force():
    query_vectors = load_fvecs(QUERY_VECTORS_FILE, 100)
    SNAP_DIR = os.path.join(DATA_DIR, "sift_index_snap")
    index = HNSWIndex.load_snapshot(SNAP_DIR)
    full_matrix = index.nodes.get_matrix()
    num_vectors = full_matrix.shape[0]

    sizes = [1000, 5000, 10000]
    k = 10

    print("=" * 80)
    print("PROFILING BRUTE FORCE AT 1K, 5K, 10K MATCHES (Averaged over 100 queries)")
    print("=" * 80)

    for M in sizes:
        # Generate random indices to simulate matching subset
        np.random.seed(42)
        matching_indices = np.random.choice(num_vectors, size=M, replace=False).astype(np.int32)
        
        # Pre-cached contiguous copy of vectors
        cached_contiguous_vectors = np.ascontiguousarray(full_matrix[matching_indices])

        t_gather_list = []
        t_dist_gathered_list = []
        t_dist_cached_list = []
        t_sort_list = []
        t_total_gathered = []
        t_total_cached = []

        for q in query_vectors:
            q_c = np.ascontiguousarray(q, dtype=np.float32)

            # Measure Row Gather
            t0 = time.perf_counter()
            gathered_matrix = full_matrix[matching_indices]
            t1 = time.perf_counter()
            t_gather = (t1 - t0) * 1e6

            # Measure Distance on Gathered Matrix
            t0 = time.perf_counter()
            diff = gathered_matrix - q_c
            dists = np.sqrt(np.sum(diff * diff, axis=1))
            t1 = time.perf_counter()
            t_dist_gathered = (t1 - t0) * 1e6

            # Measure Distance on Pre-cached Contiguous Matrix
            t0 = time.perf_counter()
            diff_c = cached_contiguous_vectors - q_c
            dists_c = np.sqrt(np.sum(diff_c * diff_c, axis=1))
            t1 = time.perf_counter()
            t_dist_cached = (t1 - t0) * 1e6

            # Measure Top-k Sort
            t0 = time.perf_counter()
            if M > k:
                subidx = np.argpartition(dists, k)[:k]
                top_k = subidx[np.argsort(dists[subidx])]
            else:
                top_k = np.argsort(dists)
            t1 = time.perf_counter()
            t_sort = (t1 - t0) * 1e6

            t_gather_list.append(t_gather)
            t_dist_gathered_list.append(t_dist_gathered)
            t_dist_cached_list.append(t_dist_cached)
            t_sort_list.append(t_sort)
            t_total_gathered.append(t_gather + t_dist_gathered + t_sort)
            t_total_cached.append(t_dist_cached + t_sort)

        avg_gather = np.mean(t_gather_list)
        avg_dist_g = np.mean(t_dist_gathered_list)
        avg_dist_c = np.mean(t_dist_cached_list)
        avg_sort = np.mean(t_sort_list)
        avg_total_g = np.mean(t_total_gathered)
        avg_total_c = np.mean(t_total_cached)

        print(f"\n--- Matching Count: {M} Vectors ({M/num_vectors*100:.1f}% Selectivity) ---")
        print(f"1. Row Gather (matrix[indices]):      {avg_gather:8.2f} us ({avg_gather/avg_total_g*100:5.1f}% of uncached total)")
        print(f"2. Distance Calc (Gathered copy):     {avg_dist_g:8.2f} us ({avg_dist_g/avg_total_g*100:5.1f}% of uncached total)")
        print(f"3. Top-k Sort (argpartition+sort):    {avg_sort:8.2f} us ({avg_sort/avg_total_g*100:5.1f}% of uncached total)")
        print(f"   => TOTAL UNCACHED BRUTE FORCE:     {avg_total_g:8.2f} us ({1e6/avg_total_g:7.1f} QPS)")
        print(f"4. Distance Calc (Cached Contiguous): {avg_dist_c:8.2f} us")
        print(f"   => TOTAL CACHED CONTIGUOUS:        {avg_total_c:8.2f} us ({1e6/avg_total_c:7.1f} QPS)")
        print(f"   => SPEEDUP FROM CONTIGUOUS CACHE:  {avg_total_g/avg_total_c:5.2f}x faster ({avg_gather:8.2f} us gather eliminated)")

if __name__ == "__main__":
    profile_brute_force()


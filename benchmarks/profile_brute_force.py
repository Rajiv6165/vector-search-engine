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

def profile_brute_force_with_warmup(num_reps: int = 5):
    query_vectors = load_fvecs(QUERY_VECTORS_FILE, 100)
    SNAP_DIR = os.path.join(DATA_DIR, "sift_index_snap")
    index = HNSWIndex.load_snapshot(SNAP_DIR)
    full_matrix = index.nodes.get_matrix()
    num_vectors = full_matrix.shape[0]

    sizes = [1000, 5000, 10000]
    k = 10

    print("=" * 110)
    print(f"PROFILING BRUTE FORCE AT 1K, 5K, 10K MATCHES ({num_reps} Repetitions x 100 Queries with Warmup)")
    print("=" * 110)

    for M in sizes:
        np.random.seed(42)
        matching_indices = np.random.choice(num_vectors, size=M, replace=False).astype(np.int32)
        cached_contiguous_vectors = np.ascontiguousarray(full_matrix[matching_indices])

        # Warmup (20 iterations)
        for q in query_vectors[:20]:
            q_c = np.ascontiguousarray(q, dtype=np.float32)
            _ = full_matrix[matching_indices]
            _ = cached_contiguous_vectors - q_c

        gather_times = []
        dist_g_times = []
        dist_c_times = []
        sort_times = []
        total_g_times = []
        total_c_times = []

        for rep in range(num_reps):
            for q in query_vectors:
                q_c = np.ascontiguousarray(q, dtype=np.float32)

                # 1. Row Gather
                t0 = time.perf_counter()
                gathered_matrix = full_matrix[matching_indices]
                t1 = time.perf_counter()
                t_gather = (t1 - t0) * 1e6

                # 2. Distance on Gathered Matrix
                t0 = time.perf_counter()
                diff_g = gathered_matrix - q_c
                dists_g = np.sqrt(np.sum(diff_g * diff_g, axis=1))
                t1 = time.perf_counter()
                t_dist_g = (t1 - t0) * 1e6

                # 3. Distance on Pre-cached Contiguous Matrix
                t0 = time.perf_counter()
                diff_c = cached_contiguous_vectors - q_c
                dists_c = np.sqrt(np.sum(diff_c * diff_c, axis=1))
                t1 = time.perf_counter()
                t_dist_c = (t1 - t0) * 1e6

                # 4. Top-k Sort
                t0 = time.perf_counter()
                if M > k:
                    subidx = np.argpartition(dists_g, k)[:k]
                    _ = subidx[np.argsort(dists_g[subidx])]
                else:
                    _ = np.argsort(dists_g)
                t1 = time.perf_counter()
                t_sort = (t1 - t0) * 1e6

                gather_times.append(t_gather)
                dist_g_times.append(t_dist_g)
                dist_c_times.append(t_dist_c)
                sort_times.append(t_sort)
                total_g_times.append(t_gather + t_dist_g + t_sort)
                total_c_times.append(t_dist_c + t_sort)

        def format_stats(data):
            med = np.median(data)
            p25 = np.percentile(data, 25)
            p75 = np.percentile(data, 75)
            spread = p75 - p25
            return f"{med:8.2f} us (IQR: {spread:6.2f} us, [{p25:7.2f} - {p75:7.2f}])"

        med_total_g = np.median(total_g_times)
        med_total_c = np.median(total_c_times)
        med_gather = np.median(gather_times)
        gather_pct = (med_gather / med_total_g) * 100.0
        speedup = med_total_g / med_total_c

        print(f"\n--- Matching Count: {M} Vectors ({M/num_vectors*100:.1f}% Selectivity, {len(gather_times)} measurements) ---")
        print(f"1. Row Gather (`matrix[indices]`):     {format_stats(gather_times)} | {gather_pct:5.1f}% of uncached")
        print(f"2. Distance Calc (Gathered Matrix):    {format_stats(dist_g_times)}")
        print(f"3. Top-k Selection (argpartition):     {format_stats(sort_times)}")
        print(f"   => TOTAL UNCACHED BRUTE FORCE:      {format_stats(total_g_times)} | {1e6/med_total_g:7.1f} QPS")
        print(f"4. Distance Calc (Cached Contiguous):  {format_stats(dist_c_times)}")
        print(f"   => TOTAL CACHED CONTIGUOUS:         {format_stats(total_c_times)} | {1e6/med_total_c:7.1f} QPS")
        print(f"   => SPEEDUP FROM CONTIGUOUS CACHE:   {speedup:5.2f}x faster ({med_gather:8.2f} us gather eliminated)")

if __name__ == "__main__":
    profile_brute_force_with_warmup(num_reps=5)

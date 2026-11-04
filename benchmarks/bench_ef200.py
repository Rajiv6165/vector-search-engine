import os
import sys
import time
import numpy as np
import faiss

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from vsearch.hnsw import HNSWIndex, HNSWConfig
from bench_vs_faiss import load_fvecs, compute_recall

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BASE_VECTORS_FILE = os.path.join(DATA_DIR, "sift", "sift_base.fvecs")
QUERY_VECTORS_FILE = os.path.join(DATA_DIR, "sift", "sift_query.fvecs")
GT_FILE = os.path.join(DATA_DIR, "sift", "sift_groundtruth.ivecs")

def load_ivecs(filename: str, max_count: int) -> np.ndarray:
    return np.fromfile(filename, dtype=np.int32, count=max_count * 101).reshape(max_count, 101)[:, 1:].copy()

def run_ef200_benchmark():
    print("Loading 100K SIFT base vectors and 1K queries...")
    base_vectors = load_fvecs(BASE_VECTORS_FILE, 100_000)
    query_vectors = load_fvecs(QUERY_VECTORS_FILE, 1_000)
    I_gt = load_ivecs(GT_FILE, 1_000)
    
    # 1. FAISS index
    print("Building FAISS index (M=16, efC=100)...")
    faiss_index = faiss.IndexHNSWFlat(128, 16, faiss.METRIC_L2)
    faiss_index.hnsw.efConstruction = 100
    faiss_index.add(base_vectors)
    faiss_index.hnsw.efSearch = 200
    faiss.omp_set_num_threads(1)
    
    # 2. vsearch (heuristic=True)
    print("Building vsearch index (heuristic=True)...")
    cfg_true = HNSWConfig(M=16, ef_construction=100, heuristic=True)
    vsearch_true = HNSWIndex(config=cfg_true, metric="l2")
    vsearch_true.batch_insert(base_vectors)
    vsearch_true.config.ef_search = 200
    
    # 3. vsearch (heuristic=False)
    print("Building vsearch index (heuristic=False)...")
    cfg_false = HNSWConfig(M=16, ef_construction=100, heuristic=False)
    vsearch_false = HNSWIndex(config=cfg_false, metric="l2")
    vsearch_false.batch_insert(base_vectors)
    vsearch_false.config.ef_search = 200
    
    # Warmup
    print("Warmup...")
    for i in range(50):
        faiss_index.search(query_vectors[i:i+1], 10)
        vsearch_true.search(query_vectors[i], k=10, ef=200)
        vsearch_false.search(query_vectors[i], k=10, ef=200)
        
    num_reps = 5
    f_qps_runs, vt_qps_runs, vf_qps_runs = [], [], []
    f_r10_runs, vt_r10_runs, vf_r10_runs = [], [], []
    
    print(f"\nRunning {num_reps} clean interleaved repetitions at efSearch=200:")
    for rep in range(num_reps):
        # FAISS
        f_lats, f_res = [], []
        for i in range(len(query_vectors)):
            q = query_vectors[i:i+1]
            t0 = time.perf_counter()
            _, I = faiss_index.search(q, 10)
            t1 = time.perf_counter()
            f_lats.append((t1 - t0) * 1000)
            f_res.append(I[0])
        f_qps = len(query_vectors) / (sum(f_lats) / 1000)
        f_r10 = compute_recall(np.array(f_res), I_gt, 10)
        f_qps_runs.append(f_qps)
        f_r10_runs.append(f_r10)
        
        # vsearch H=True
        vt_lats, vt_res = [], []
        for i in range(len(query_vectors)):
            q = query_vectors[i]
            t0 = time.perf_counter()
            res = vsearch_true.search(q, k=10, ef=200)
            t1 = time.perf_counter()
            vt_lats.append((t1 - t0) * 1000)
            vt_res.append([x[0] for x in res])
        vt_qps = len(query_vectors) / (sum(vt_lats) / 1000)
        vt_r10 = compute_recall(np.array(vt_res), I_gt, 10)
        vt_qps_runs.append(vt_qps)
        vt_r10_runs.append(vt_r10)
        
        # vsearch H=False
        vf_lats, vf_res = [], []
        for i in range(len(query_vectors)):
            q = query_vectors[i]
            t0 = time.perf_counter()
            res = vsearch_false.search(q, k=10, ef=200)
            t1 = time.perf_counter()
            vf_lats.append((t1 - t0) * 1000)
            vf_res.append([x[0] for x in res])
        vf_qps = len(query_vectors) / (sum(vf_lats) / 1000)
        vf_r10 = compute_recall(np.array(vf_res), I_gt, 10)
        vf_qps_runs.append(vf_qps)
        vf_r10_runs.append(vf_r10)
        
        print(f"Rep {rep+1}: FAISS: {f_qps:6.1f} QPS (R@10: {f_r10:.4f}) | "
              f"vsearch(H=True): {vt_qps:6.1f} QPS (R@10: {vt_r10:.4f}) | "
              f"vsearch(H=False): {vf_qps:6.1f} QPS (R@10: {vf_r10:.4f})")
              
    def stats(arr):
        med = float(np.median(arr))
        # Exclude runs under 50% of the median as stalled OS outliers
        clean = [x for x in arr if x >= 0.5 * med]
        return med, min(clean), max(clean), min(arr)
        
    f_med, f_min, f_max, f_raw_min = stats(f_qps_runs)
    vt_med, vt_min, vt_max, vt_raw_min = stats(vt_qps_runs)
    vf_med, vf_min, vf_max, vf_raw_min = stats(vf_qps_runs)
    
    print("\n--- Summary ef=200 ---")
    print(f"FAISS: Median={f_med:.1f} QPS [Clean Range: {f_min:.1f} - {f_max:.1f}, Raw Min: {f_raw_min:.1f}] | Recall@10: {np.mean(f_r10_runs)*100:.2f}%")
    print(f"vsearch (H=True): Median={vt_med:.1f} QPS [Clean Range: {vt_min:.1f} - {vt_max:.1f}, Raw Min: {vt_raw_min:.1f}] | Recall@10: {np.mean(vt_r10_runs)*100:.2f}%")
    print(f"vsearch (H=False): Median={vf_med:.1f} QPS [Clean Range: {vf_min:.1f} - {vf_max:.1f}, Raw Min: {vf_raw_min:.1f}] | Recall@10: {np.mean(vf_r10_runs)*100:.2f}%")

if __name__ == "__main__":
    run_ef200_benchmark()

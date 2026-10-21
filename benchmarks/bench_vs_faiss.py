import os
import sys
import time
import numpy as np
import faiss
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from vsearch.hnsw import HNSWIndex, HNSWConfig

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BASE_VECTORS_FILE = os.path.join(DATA_DIR, "sift", "sift_base.fvecs")
QUERY_VECTORS_FILE = os.path.join(DATA_DIR, "sift", "sift_query.fvecs")

NUM_BASE_VECTORS = 100_000
NUM_QUERIES = 1_000
DIM = 128
M = 16
EF_CONSTRUCTION = 100
EF_SEARCH_VALUES = [10, 20, 50, 100, 200]
NUM_REPETITIONS = 5
NUM_BUILDS = 3

def load_fvecs(filename: str, max_count: int) -> np.ndarray:
    return np.fromfile(filename, dtype=np.float32, count=max_count * 129).reshape(max_count, 129)[:, 1:].copy()

def compute_recall(I_approx: np.ndarray, I_gt: np.ndarray, k: int) -> float:
    """Compute average recall@k."""
    recalls = []
    for i in range(len(I_approx)):
        gt_set = set(I_gt[i][:k])
        approx_set = set(I_approx[i][:k])
        hits = len(gt_set.intersection(approx_set))
        recalls.append(hits / k)
    return float(np.mean(recalls))

def benchmark_builds(base_vectors: np.ndarray):
    print("\n" + "="*85)
    print(f"BENCHMARKING BUILD TIMES ({NUM_BUILDS} builds per engine)")
    print("="*85)
    
    # 1. FAISS builds
    faiss_times = []
    faiss_active_index = None
    for b in range(NUM_BUILDS):
        idx = faiss.IndexHNSWFlat(DIM, M, faiss.METRIC_L2)
        idx.hnsw.efConstruction = EF_CONSTRUCTION
        # Strict assertions for FAISS parity
        assert idx.hnsw.efConstruction == EF_CONSTRUCTION, (
            f"FAISS efConstruction mismatch: expected {EF_CONSTRUCTION}, got {idx.hnsw.efConstruction}"
        )
        t0 = time.time()
        idx.add(base_vectors)
        dt = time.time() - t0
        faiss_times.append(dt)
        print(f"FAISS Build {b+1}/{NUM_BUILDS}: {dt:.2f}s | Verified M={M}, efConstruction={idx.hnsw.efConstruction}")
        if b == 0:
            faiss_active_index = idx

    # 2. vsearch (heuristic=True) builds
    vsearch_true_times = []
    vsearch_true_active = None
    for b in range(NUM_BUILDS):
        cfg = HNSWConfig(M=M, ef_construction=EF_CONSTRUCTION, heuristic=True)
        idx = HNSWIndex(config=cfg, metric="l2")
        t0 = time.time()
        idx.batch_insert(base_vectors)
        dt = time.time() - t0
        vsearch_true_times.append(dt)
        print(f"vsearch (heuristic=True) Build {b+1}/{NUM_BUILDS}: {dt:.2f}s ({len(base_vectors)/dt:.1f} vec/s)")
        if b == 0:
            vsearch_true_active = idx

    # 3. vsearch (heuristic=False) build (Ablation)
    vsearch_false_times = []
    cfg_false = HNSWConfig(M=M, ef_construction=EF_CONSTRUCTION, heuristic=False)
    vsearch_false_active = HNSWIndex(config=cfg_false, metric="l2")
    t0 = time.time()
    vsearch_false_active.batch_insert(base_vectors)
    dt = time.time() - t0
    vsearch_false_times.append(dt)
    print(f"vsearch (heuristic=False) Build: {dt:.2f}s ({len(base_vectors)/dt:.1f} vec/s)")

    return {
        "faiss": {"times": faiss_times, "index": faiss_active_index},
        "vsearch_true": {"times": vsearch_true_times, "index": vsearch_true_active},
        "vsearch_false": {"times": vsearch_false_times, "index": vsearch_false_active}
    }

def benchmark_search_interleaved(
    faiss_index: faiss.IndexHNSWFlat,
    vsearch_true: HNSWIndex,
    vsearch_false: HNSWIndex,
    query_vectors: np.ndarray,
    I_gt: np.ndarray
):
    print("\n" + "="*85)
    print("INTERLEAVED SEARCH BENCHMARK (5 repetitions per efSearch, single-threaded)")
    print("="*85)
    
    # Strictly enforce 1 thread for FAISS
    faiss.omp_set_num_threads(1)
    max_threads = faiss.omp_get_max_threads()
    assert max_threads == 1, f"Expected FAISS OpenMP threads == 1, got {max_threads}"
    print(f"FAISS OpenMP max threads verified: {max_threads} (strict single-threaded search)")
    
    results = {
        "faiss": {},
        "vsearch_true": {},
        "vsearch_false": {}
    }
    
    for ef in EF_SEARCH_VALUES:
        print(f"\n>>> Operating Point: efSearch = {ef} <<<")
        
        # Configure and verify parameters
        faiss_index.hnsw.efSearch = ef
        assert faiss_index.hnsw.efSearch == ef, f"FAISS efSearch mismatch: {faiss_index.hnsw.efSearch} != {ef}"
        vsearch_true.config.ef_search = ef
        vsearch_false.config.ef_search = ef
        
        print(f"Asserted params: FAISS efSearch={faiss_index.hnsw.efSearch}, "
              f"vsearch_true efSearch={vsearch_true.config.ef_search}, "
              f"vsearch_false efSearch={vsearch_false.config.ef_search}")
        
        # Warmup all three engines
        for i in range(100):
            faiss_index.search(query_vectors[i:i+1], 10)
            vsearch_true.search(query_vectors[i], k=10, ef=ef)
            vsearch_false.search(query_vectors[i], k=10, ef=ef)
            
        faiss_qps_runs, faiss_r10_runs, faiss_lats_all = [], [], []
        vtrue_qps_runs, vtrue_r10_runs, vtrue_lats_all = [], [], []
        vfalse_qps_runs, vfalse_r10_runs, vfalse_lats_all = [], [], []
        
        # Interleave repetitions across engines
        for rep in range(NUM_REPETITIONS):
            # 1. FAISS (single query calls)
            f_lats, f_res = [], []
            for i in range(NUM_QUERIES):
                q = query_vectors[i:i+1]
                t0 = time.perf_counter()
                D, I = faiss_index.search(q, 10)
                t1 = time.perf_counter()
                f_lats.append((t1 - t0) * 1000)
                f_res.append(I[0])
            f_qps = NUM_QUERIES / (sum(f_lats) / 1000)
            f_r10 = compute_recall(np.array(f_res), I_gt, 10)
            faiss_qps_runs.append(f_qps)
            faiss_r10_runs.append(f_r10)
            faiss_lats_all.extend(f_lats)
            
            # 2. vsearch (heuristic=True)
            vt_lats, vt_res = [], []
            for i in range(NUM_QUERIES):
                q = query_vectors[i]
                t0 = time.perf_counter()
                res = vsearch_true.search(q, k=10, ef=ef)
                t1 = time.perf_counter()
                vt_lats.append((t1 - t0) * 1000)
                vt_res.append([x[0] for x in res])
            vt_qps = NUM_QUERIES / (sum(vt_lats) / 1000)
            vt_r10 = compute_recall(np.array(vt_res), I_gt, 10)
            vtrue_qps_runs.append(vt_qps)
            vtrue_r10_runs.append(vt_r10)
            vtrue_lats_all.extend(vt_lats)

            # 3. vsearch (heuristic=False)
            vf_lats, vf_res = [], []
            for i in range(NUM_QUERIES):
                q = query_vectors[i]
                t0 = time.perf_counter()
                res = vsearch_false.search(q, k=10, ef=ef)
                t1 = time.perf_counter()
                vf_lats.append((t1 - t0) * 1000)
                vf_res.append([x[0] for x in res])
            vf_qps = NUM_QUERIES / (sum(vf_lats) / 1000)
            vf_r10 = compute_recall(np.array(vf_res), I_gt, 10)
            vfalse_qps_runs.append(vf_qps)
            vfalse_r10_runs.append(vf_r10)
            vfalse_lats_all.extend(vf_lats)
            
            print(f"  Rep {rep+1}/{NUM_REPETITIONS} -> "
                  f"FAISS: {f_qps:7.1f} QPS (R@10: {f_r10:.4f}) | "
                  f"vsearch(H=True): {vt_qps:7.1f} QPS (R@10: {vt_r10:.4f}) | "
                  f"vsearch(H=False): {vf_qps:7.1f} QPS (R@10: {vf_r10:.4f})")
            
        # Recall@100 evaluation for ef >= 100
        f_r100, vt_r100, vf_r100 = None, None, None
        if ef >= 100:
            _, I100 = faiss_index.search(query_vectors, 100)
            f_r100 = compute_recall(I100, I_gt, 100)
            
            vt_100_res = [vsearch_true.search(query_vectors[i], k=100, ef=ef) for i in range(NUM_QUERIES)]
            vt_r100 = compute_recall(np.array([[x[0] for x in r] for r in vt_100_res]), I_gt, 100)
            
            vf_100_res = [vsearch_false.search(query_vectors[i], k=100, ef=ef) for i in range(NUM_QUERIES)]
            vf_r100 = compute_recall(np.array([[x[0] for x in r] for r in vf_100_res]), I_gt, 100)

        results["faiss"][ef] = {
            "qps_median": float(np.median(faiss_qps_runs)),
            "qps_min": float(np.min(faiss_qps_runs)),
            "qps_max": float(np.max(faiss_qps_runs)),
            "qps_mean": float(np.mean(faiss_qps_runs)),
            "qps_std": float(np.std(faiss_qps_runs)),
            "r10_median": float(np.median(faiss_r10_runs)),
            "r10_min": float(np.min(faiss_r10_runs)),
            "r10_max": float(np.max(faiss_r10_runs)),
            "r100": f_r100,
            "p50": float(np.percentile(faiss_lats_all, 50)),
            "p95": float(np.percentile(faiss_lats_all, 95)),
            "p99": float(np.percentile(faiss_lats_all, 99)),
            "qps_runs": faiss_qps_runs
        }
        results["vsearch_true"][ef] = {
            "qps_median": float(np.median(vtrue_qps_runs)),
            "qps_min": float(np.min(vtrue_qps_runs)),
            "qps_max": float(np.max(vtrue_qps_runs)),
            "qps_mean": float(np.mean(vtrue_qps_runs)),
            "qps_std": float(np.std(vtrue_qps_runs)),
            "r10_median": float(np.median(vtrue_r10_runs)),
            "r10_min": float(np.min(vtrue_r10_runs)),
            "r10_max": float(np.max(vtrue_r10_runs)),
            "r100": vt_r100,
            "p50": float(np.percentile(vtrue_lats_all, 50)),
            "p95": float(np.percentile(vtrue_lats_all, 95)),
            "p99": float(np.percentile(vtrue_lats_all, 99)),
            "qps_runs": vtrue_qps_runs
        }
        results["vsearch_false"][ef] = {
            "qps_median": float(np.median(vfalse_qps_runs)),
            "qps_min": float(np.min(vfalse_qps_runs)),
            "qps_max": float(np.max(vfalse_qps_runs)),
            "qps_mean": float(np.mean(vfalse_qps_runs)),
            "qps_std": float(np.std(vfalse_qps_runs)),
            "r10_median": float(np.median(vfalse_r10_runs)),
            "r10_min": float(np.min(vfalse_r10_runs)),
            "r10_max": float(np.max(vfalse_r10_runs)),
            "r100": vf_r100,
            "p50": float(np.percentile(vfalse_lats_all, 50)),
            "p95": float(np.percentile(vfalse_lats_all, 95)),
            "p99": float(np.percentile(vfalse_lats_all, 99)),
            "qps_runs": vfalse_qps_runs
        }

    return results

def plot_tradeoff(results, save_path):
    plt.figure(figsize=(10, 6))
    
    faiss_recalls = [results["faiss"][ef]["r10_median"] for ef in EF_SEARCH_VALUES]
    faiss_qps = [results["faiss"][ef]["qps_median"] for ef in EF_SEARCH_VALUES]
    
    vtrue_recalls = [results["vsearch_true"][ef]["r10_median"] for ef in EF_SEARCH_VALUES]
    vtrue_qps = [results["vsearch_true"][ef]["qps_median"] for ef in EF_SEARCH_VALUES]

    vfalse_recalls = [results["vsearch_false"][ef]["r10_median"] for ef in EF_SEARCH_VALUES]
    vfalse_qps = [results["vsearch_false"][ef]["qps_median"] for ef in EF_SEARCH_VALUES]
    
    plt.plot(faiss_recalls, faiss_qps, marker='o', label="FAISS IndexHNSWFlat (C++, 1 thread)", linewidth=2, color="#1f77b4")
    plt.plot(vtrue_recalls, vtrue_qps, marker='s', label="vsearch Native (heuristic=True)", linewidth=2, color="#2ca02c")
    plt.plot(vfalse_recalls, vfalse_qps, marker='^', label="vsearch Native (heuristic=False) [Ablation]", linewidth=2, color="#ff7f0e", linestyle="--")
    
    for i, ef in enumerate(EF_SEARCH_VALUES):
        plt.annotate(f"ef={ef}", (faiss_recalls[i], faiss_qps[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
        plt.annotate(f"ef={ef}", (vtrue_recalls[i], vtrue_qps[i]), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=9)
        plt.annotate(f"ef={ef}", (vfalse_recalls[i], vfalse_qps[i]), textcoords="offset points", xytext=(15, 0), ha='left', fontsize=8, color="#d62728")

    plt.yscale('log')
    plt.xlabel("Recall@10", fontsize=12)
    plt.ylabel("Queries Per Second (QPS) - Log Scale", fontsize=12)
    plt.title(f"Recall@10 vs QPS Tradeoff on SIFT100K (M={M}, efConstruction={EF_CONSTRUCTION})", fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nTradeoff chart saved to {save_path}")

def print_summary_tables(build_data, search_results):
    f_times = build_data["faiss"]["times"]
    vt_times = build_data["vsearch_true"]["times"]
    vf_times = build_data["vsearch_false"]["times"]
    
    print("\n" + "="*95)
    print("1. BUILD TIME SUMMARY (Over 3 builds)")
    print("="*95)
    print(f"FAISS Build Times:              min={min(f_times):.2f}s | median={np.median(f_times):.2f}s | max={max(f_times):.2f}s")
    print(f"vsearch (heuristic=True) Build: min={min(vt_times):.2f}s | median={np.median(vt_times):.2f}s | max={max(vt_times):.2f}s")
    print(f"vsearch (heuristic=False) Build: {vf_times[0]:.2f}s")
    
    speedup_min = 923.6 / max(vt_times)
    speedup_max = 923.6 / min(vt_times)
    print(f"Build Speedup vs Phase 6 (923.6s): {speedup_min:.1f}x to {speedup_max:.1f}x speedup")

    print("\n" + "="*115)
    print("2. ABLATION & COMPARISON TABLE (Recall@10, Recall@100, and QPS: Median [Min, Max] over 5 reps)")
    print("="*115)
    print(f"{'ef':<4} | {'FAISS R@10':<12} | {'vsearch (H=True) R@10':<22} | {'vsearch (H=False) R@10':<23} | {'FAISS QPS (Median)':<20} | {'vsearch (H=True) QPS':<22} | {'vsearch (H=False) QPS':<22}")
    print("-"*115)
    
    for ef in EF_SEARCH_VALUES:
        f = search_results["faiss"][ef]
        vt = search_results["vsearch_true"][ef]
        vf = search_results["vsearch_false"][ef]
        
        f_qps_str = f"{f['qps_median']:6.0f} [{f['qps_min']:5.0f}-{f['qps_max']:5.0f}]"
        vt_qps_str = f"{vt['qps_median']:6.0f} [{vt['qps_min']:5.0f}-{vt['qps_max']:5.0f}]"
        vf_qps_str = f"{vf['qps_median']:6.0f} [{vf['qps_min']:5.0f}-{vf['qps_max']:5.0f}]"
        
        print(f"{ef:<4} | {f['r10_median']*100:5.2f}%       | {vt['r10_median']*100:5.2f}%                | {vf['r10_median']*100:5.2f}%                 | {f_qps_str:<20} | {vt_qps_str:<22} | {vf_qps_str:<22}")

    print("\n" + "="*95)
    print("3. HEAD-TO-HEAD QPS RATIO RANGES (FAISS vs vsearch Native heuristic=True)")
    print("="*95)
    print(f"{'ef':<4} | {'FAISS QPS (Median)':<20} | {'vsearch QPS (Median)':<22} | {'QPS Ratio (FAISS/vsearch)':<25}")
    print("-"*95)
    all_ratios = []
    for ef in EF_SEARCH_VALUES:
        f = search_results["faiss"][ef]
        vt = search_results["vsearch_true"][ef]
        ratios = [f_q / vt_q for f_q, vt_q in zip(f["qps_runs"], vt["qps_runs"])]
        all_ratios.extend(ratios)
        print(f"{ef:<4} | {f['qps_median']:8.1f}             | {vt['qps_median']:8.1f}               | {np.median(ratios):.2f}x [{min(ratios):.2f}x - {max(ratios):.2f}x]")
    print(f"\nOverall FAISS-to-vsearch QPS Ratio across all operating points: {min(all_ratios):.2f}x to {max(all_ratios):.2f}x faster")
    print("="*95)

def main():
    if not os.path.exists(BASE_VECTORS_FILE):
        print(f"Dataset not found at {BASE_VECTORS_FILE}. Run download_dataset.py first.")
        sys.exit(1)
        
    print("Loading vectors from .fvecs files...")
    base_vectors = load_fvecs(BASE_VECTORS_FILE, NUM_BASE_VECTORS)
    query_vectors = load_fvecs(QUERY_VECTORS_FILE, NUM_QUERIES)
    print(f"Loaded {len(base_vectors)} base vectors and {len(query_vectors)} query vectors.")
    
    print("Computing exact ground truth via FAISS IndexFlatL2...")
    exact_index = faiss.IndexFlatL2(DIM)
    exact_index.add(base_vectors)
    _, I_gt = exact_index.search(query_vectors, 100)
    print("Ground truth computed.")
    
    build_data = benchmark_builds(base_vectors)
    search_results = benchmark_search_interleaved(
        build_data["faiss"]["index"],
        build_data["vsearch_true"]["index"],
        build_data["vsearch_false"]["index"],
        query_vectors,
        I_gt
    )
    
    save_path = os.path.join(os.path.dirname(__file__), "faiss_tradeoff.png")
    plot_tradeoff(search_results, save_path)
    print_summary_tables(build_data, search_results)

if __name__ == "__main__":
    main()

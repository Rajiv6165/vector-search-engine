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
NUM_RUNS = 3

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

def benchmark_faiss(base_vectors: np.ndarray, query_vectors: np.ndarray, I_gt: np.ndarray):
    print("\n--- Benchmarking FAISS ---")
    
    faiss_index = faiss.IndexHNSWFlat(DIM, M, faiss.METRIC_L2)
    faiss_index.hnsw.efConstruction = EF_CONSTRUCTION
    
    print("Building FAISS index...")
    t0 = time.time()
    faiss_index.add(base_vectors)
    build_time = time.time() - t0
    print(f"FAISS index built in {build_time:.2f} seconds.")
    
    results = []
    
    for ef in EF_SEARCH_VALUES:
        faiss_index.hnsw.efSearch = ef
        
        # Warmup
        faiss_index.search(query_vectors[:100], 100)
        
        run_qps = []
        run_p50 = []
        run_p95 = []
        run_p99 = []
        run_r10 = []
        run_r100 = []
        
        for run_idx in range(NUM_RUNS):
            latencies = []
            I_results = []
            for i in range(NUM_QUERIES):
                q = query_vectors[i:i+1]
                t0 = time.perf_counter()
                D, I = faiss_index.search(q, 100)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)
                I_results.append(I[0])
                
            I_results = np.array(I_results)
            r10 = compute_recall(I_results, I_gt, 10)
            r100 = compute_recall(I_results, I_gt, 100)
            qps = NUM_QUERIES / (sum(latencies) / 1000)
            
            run_qps.append(qps)
            run_p50.append(np.percentile(latencies, 50))
            run_p95.append(np.percentile(latencies, 95))
            run_p99.append(np.percentile(latencies, 99))
            run_r10.append(r10)
            run_r100.append(r100)
            
        mean_qps, std_qps = float(np.mean(run_qps)), float(np.std(run_qps))
        mean_p50 = float(np.mean(run_p50))
        mean_p95, std_p95 = float(np.mean(run_p95)), float(np.std(run_p95))
        mean_p99 = float(np.mean(run_p99))
        mean_r10 = float(np.mean(run_r10))
        mean_r100 = float(np.mean(run_r100))
        
        print(f"efSearch={ef:<4} | R@10: {mean_r10:.4f} | R@100: {mean_r100:.4f} | QPS: {mean_qps:8.2f} ± {std_qps:5.2f} | p95: {mean_p95:6.3f}ms")
        
        results.append({
            "ef": ef,
            "recall_10": mean_r10,
            "recall_100": mean_r100,
            "qps": mean_qps,
            "std_qps": std_qps,
            "p50": mean_p50,
            "p95": mean_p95,
            "std_p95": std_p95,
            "p99": mean_p99
        })
        
    return {"build_time": build_time, "results": results}

def benchmark_vsearch(base_vectors: np.ndarray, query_vectors: np.ndarray, I_gt: np.ndarray):
    print("\n--- Benchmarking vsearch.HNSWIndex (Phase 9 Native Hot Path) ---")
    
    config = HNSWConfig(M=M, ef_construction=EF_CONSTRUCTION, heuristic=True)
    vsearch_index = HNSWIndex(config=config, metric="l2")
    
    print("Building vsearch index...")
    t0 = time.time()
    for i in range(len(base_vectors)):
        vsearch_index.insert(i, base_vectors[i])
        if (i+1) % 20000 == 0:
            print(f"  Added {i+1} vectors in {time.time()-t0:.1f}s...")
    build_time = time.time() - t0
    print(f"vsearch index built in {build_time:.2f} seconds ({len(base_vectors)/build_time:.1f} vec/s).")
    
    results = []
    
    for ef in EF_SEARCH_VALUES:
        vsearch_index.config.ef_search = ef
        
        # Warmup
        for i in range(100):
            vsearch_index.search(query_vectors[i], k=100)
            
        run_qps = []
        run_p50 = []
        run_p95 = []
        run_p99 = []
        run_r10 = []
        run_r100 = []
        
        for run_idx in range(NUM_RUNS):
            latencies = []
            I_results = []
            for i in range(NUM_QUERIES):
                q = query_vectors[i]
                t0 = time.perf_counter()
                res = vsearch_index.search(q, k=100, ef=ef)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)
                
                approx_ids = [node_id for node_id, _, _ in res]
                while len(approx_ids) < 100:
                    approx_ids.append(-1)
                I_results.append(approx_ids)
                
            I_results = np.array(I_results)
            r10 = compute_recall(I_results, I_gt, 10)
            r100 = compute_recall(I_results, I_gt, 100)
            qps = NUM_QUERIES / (sum(latencies) / 1000)
            
            run_qps.append(qps)
            run_p50.append(np.percentile(latencies, 50))
            run_p95.append(np.percentile(latencies, 95))
            run_p99.append(np.percentile(latencies, 99))
            run_r10.append(r10)
            run_r100.append(r100)
            
        mean_qps, std_qps = float(np.mean(run_qps)), float(np.std(run_qps))
        mean_p50 = float(np.mean(run_p50))
        mean_p95, std_p95 = float(np.mean(run_p95)), float(np.std(run_p95))
        mean_p99 = float(np.mean(run_p99))
        mean_r10 = float(np.mean(run_r10))
        mean_r100 = float(np.mean(run_r100))
        
        print(f"efSearch={ef:<4} | R@10: {mean_r10:.4f} | R@100: {mean_r100:.4f} | QPS: {mean_qps:8.2f} ± {std_qps:5.2f} | p95: {mean_p95:6.3f}ms")
        
        results.append({
            "ef": ef,
            "recall_10": mean_r10,
            "recall_100": mean_r100,
            "qps": mean_qps,
            "std_qps": std_qps,
            "p50": mean_p50,
            "p95": mean_p95,
            "std_p95": std_p95,
            "p99": mean_p99
        })
        
    return {"build_time": build_time, "results": results}

def plot_tradeoff(faiss_results, vsearch_results, save_path):
    plt.figure(figsize=(10, 6))
    
    faiss_recalls = [r["recall_10"] for r in faiss_results["results"]]
    faiss_qps = [r["qps"] for r in faiss_results["results"]]
    
    vsearch_recalls = [r["recall_10"] for r in vsearch_results["results"]]
    vsearch_qps = [r["qps"] for r in vsearch_results["results"]]
    
    plt.plot(faiss_recalls, faiss_qps, marker='o', label="FAISS IndexHNSWFlat (C++)", linewidth=2, color="#1f77b4")
    plt.plot(vsearch_recalls, vsearch_qps, marker='s', label="vsearch HNSWIndex (Phase 9 Native)", linewidth=2, color="#2ca02c")
    
    for i, ef in enumerate(EF_SEARCH_VALUES):
        plt.annotate(f"ef={ef}", (faiss_recalls[i], faiss_qps[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
        plt.annotate(f"ef={ef}", (vsearch_recalls[i], vsearch_qps[i]), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=9)

    plt.yscale('log')
    plt.xlabel("Recall@10", fontsize=12)
    plt.ylabel("Queries Per Second (QPS) - Log Scale", fontsize=12)
    plt.title(f"Recall@10 vs QPS Tradeoff on SIFT100K (M={M}, efConstruction={EF_CONSTRUCTION})", fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nTradeoff chart saved to {save_path}")

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
    
    faiss_res = benchmark_faiss(base_vectors, query_vectors, I_gt)
    vsearch_res = benchmark_vsearch(base_vectors, query_vectors, I_gt)
    
    save_path = os.path.join(os.path.dirname(__file__), "faiss_tradeoff.png")
    plot_tradeoff(faiss_res, vsearch_res, save_path)
    
    print("\n" + "="*85)
    print("FINAL HEAD-TO-HEAD COMPARISON: FAISS vs vsearch (100K SIFT Subset)")
    print("="*85)
    print(f"Index Build Time: FAISS = {faiss_res['build_time']:.2f}s | vsearch = {vsearch_res['build_time']:.2f}s")
    print("-"*85)
    print(f"{'efSearch':<10} | {'Recall@10 (FAISS)':<18} | {'Recall@10 (vsearch)':<19} | {'QPS (FAISS)':<16} | {'QPS (vsearch)':<16}")
    print("-"*85)
    for i in range(len(EF_SEARCH_VALUES)):
        f_item = faiss_res["results"][i]
        v_item = vsearch_res["results"][i]
        print(f"{f_item['ef']:<10} | {f_item['recall_10']*100:<17.2f}% | {v_item['recall_10']*100:<18.2f}% | {f_item['qps']:<16.1f} | {v_item['qps']:<16.1f}")
    print("="*85)

if __name__ == "__main__":
    main()

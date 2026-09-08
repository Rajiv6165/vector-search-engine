import os
import sys
import time
import numpy as np
import faiss
import matplotlib.pyplot as plt
from typing import Tuple, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from vsearch.hnsw import HNSWIndex, HNSWConfig

from download_dataset import read_fvecs

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BASE_VECTORS_FILE = os.path.join(DATA_DIR, "sift", "sift_base.fvecs")
QUERY_VECTORS_FILE = os.path.join(DATA_DIR, "sift", "sift_query.fvecs")

NUM_BASE_VECTORS = 100_000
NUM_QUERIES = 1_000
DIM = 128
M = 16
EF_CONSTRUCTION = 100
EF_SEARCH_VALUES = [10, 20, 50, 100, 200]

def compute_recall(I_approx: np.ndarray, I_gt: np.ndarray, k: int) -> float:
    """Compute average recall@k."""
    recalls = []
    for i in range(len(I_approx)):
        gt_set = set(I_gt[i][:k])
        approx_set = set(I_approx[i][:k])
        # Sometimes exact distance has ties, but standard recall just counts intersections
        hits = len(gt_set.intersection(approx_set))
        recalls.append(hits / k)
    return float(np.mean(recalls))

def benchmark_faiss(base_vectors, query_vectors, I_gt):
    print("\n--- Benchmarking FAISS ---")
    
    # Initialize FAISS Index
    # faiss.IndexHNSWFlat uses L2 distance
    faiss_index = faiss.IndexHNSWFlat(DIM, M, faiss.METRIC_L2)
    faiss_index.hnsw.efConstruction = EF_CONSTRUCTION
    
    # Build Index
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
        
        latencies = []
        I_results = []
        for i in range(NUM_QUERIES):
            q = query_vectors[i:i+1]
            t0 = time.perf_counter()
            D, I = faiss_index.search(q, 100)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000) # ms
            I_results.append(I[0])
            
        I_results = np.array(I_results)
        
        recall_10 = compute_recall(I_results, I_gt, 10)
        recall_100 = compute_recall(I_results, I_gt, 100)
        
        qps = NUM_QUERIES / (sum(latencies) / 1000)
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        
        print(f"efSearch={ef:<4} | R@10: {recall_10:.4f} | R@100: {recall_100:.4f} | QPS: {qps:8.2f} | p95: {p95:6.2f}ms")
        
        results.append({
            "ef": ef,
            "recall_10": recall_10,
            "recall_100": recall_100,
            "qps": qps,
            "p50": p50,
            "p95": p95,
            "p99": p99
        })
        
    return {"build_time": build_time, "results": results}

def benchmark_vsearch(base_vectors, query_vectors, I_gt):
    print("\n--- Benchmarking vsearch.HNSWIndex ---")
    
    # Initialize our index
    config = HNSWConfig(M=M, ef_construction=EF_CONSTRUCTION)
    # We use L2 distance by default for consistency with FAISS HNSWFlat
    vsearch_index = HNSWIndex(config=config)
    
    print("Building vsearch index...")
    t0 = time.time()
    # Batch add if implemented, else loop
    for i in range(len(base_vectors)):
        vsearch_index.insert(i, base_vectors[i])
        if (i+1) % 20000 == 0:
            print(f"  Added {i+1} vectors...")
    build_time = time.time() - t0
    print(f"vsearch index built in {build_time:.2f} seconds.")
    
    results = []
    
    for ef in EF_SEARCH_VALUES:
        vsearch_index.config.ef_search = ef
        
        # Warmup
        for i in range(100):
            vsearch_index.search(query_vectors[i], k=100)
            
        latencies = []
        I_results = []
        for i in range(NUM_QUERIES):
            q = query_vectors[i]
            t0 = time.perf_counter()
            res = vsearch_index.search(q, k=100)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000) # ms
            
            # extract node ids
            approx_ids = [node_id for node_id, _, _ in res]
            # pad if less than 100 (rare but possible at low ef)
            while len(approx_ids) < 100:
                approx_ids.append(-1)
            I_results.append(approx_ids)
            
        I_results = np.array(I_results)
        
        recall_10 = compute_recall(I_results, I_gt, 10)
        recall_100 = compute_recall(I_results, I_gt, 100)
        
        qps = NUM_QUERIES / (sum(latencies) / 1000)
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        
        print(f"efSearch={ef:<4} | R@10: {recall_10:.4f} | R@100: {recall_100:.4f} | QPS: {qps:8.2f} | p95: {p95:6.2f}ms")
        
        results.append({
            "ef": ef,
            "recall_10": recall_10,
            "recall_100": recall_100,
            "qps": qps,
            "p50": p50,
            "p95": p95,
            "p99": p99
        })
        
    return {"build_time": build_time, "results": results}

def plot_tradeoff(faiss_results, vsearch_results, save_path):
    plt.figure(figsize=(10, 6))
    
    # Extract
    faiss_recalls = [r["recall_10"] for r in faiss_results["results"]]
    faiss_qps = [r["qps"] for r in faiss_results["results"]]
    
    vsearch_recalls = [r["recall_10"] for r in vsearch_results["results"]]
    vsearch_qps = [r["qps"] for r in vsearch_results["results"]]
    
    plt.plot(faiss_recalls, faiss_qps, marker='o', label="FAISS IndexHNSWFlat", linewidth=2)
    plt.plot(vsearch_recalls, vsearch_qps, marker='s', label="vsearch HNSWIndex (Ours)", linewidth=2)
    
    # Annotate points with efSearch
    for i, ef in enumerate(EF_SEARCH_VALUES):
        plt.annotate(f"ef={ef}", (faiss_recalls[i], faiss_qps[i]), textcoords="offset points", xytext=(0,10), ha='center')
        plt.annotate(f"ef={ef}", (vsearch_recalls[i], vsearch_qps[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.yscale('log')
    plt.xlabel("Recall@10")
    plt.ylabel("Queries Per Second (QPS) - Log Scale")
    plt.title(f"Recall@10 vs QPS Tradeoff (SIFT {NUM_BASE_VECTORS//1000}K, M={M})")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nTradeoff chart saved to {save_path}")

def main():
    if not os.path.exists(BASE_VECTORS_FILE):
        print("Dataset not found. Run download_dataset.py first.")
        sys.exit(1)
        
    print("Loading vectors...")
    base_vectors = read_fvecs(BASE_VECTORS_FILE, max_count=NUM_BASE_VECTORS)
    query_vectors = read_fvecs(QUERY_VECTORS_FILE, max_count=NUM_QUERIES)
    
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
    
if __name__ == "__main__":
    main()

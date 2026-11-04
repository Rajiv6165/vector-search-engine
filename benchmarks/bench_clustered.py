import os
import sys
import time
import numpy as np
from typing import Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from vsearch.hnsw import HNSWIndex, HNSWConfig
from vsearch.brute_force import ExactKNN

def generate_gmm_dataset(
    num_vectors: int = 100_000,
    num_queries: int = 500,
    dim: int = 128,
    num_clusters: int = 50,
    cluster_std: float = 0.3,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    
    # Generate separated cluster centers
    centers = rng.normal(0.0, 5.0, size=(num_clusters, dim)).astype(np.float32)
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    centers = (centers / np.maximum(norms, 1e-6)) * 10.0
    
    cluster_assignments = rng.integers(0, num_clusters, size=num_vectors)
    base_vectors = np.zeros((num_vectors, dim), dtype=np.float32)
    for i in range(num_vectors):
        c_idx = cluster_assignments[i]
        base_vectors[i] = centers[c_idx] + rng.normal(0.0, cluster_std, size=dim).astype(np.float32)
        
    query_clusters = rng.integers(0, num_clusters, size=num_queries)
    query_vectors = np.zeros((num_queries, dim), dtype=np.float32)
    for i in range(num_queries):
        c_idx = query_clusters[i]
        query_vectors[i] = centers[c_idx] + rng.normal(0.0, cluster_std * 1.2, size=dim).astype(np.float32)
        
    return base_vectors, query_vectors

def compute_ground_truth(base_vectors: np.ndarray, query_vectors: np.ndarray, k: int = 10) -> np.ndarray:
    base_norms = np.sum(base_vectors * base_vectors, axis=1)
    gt = []
    for q in query_vectors:
        dists = base_norms - 2.0 * np.dot(base_vectors, q)
        top_k = np.argpartition(dists, k)[:k]
        top_k_sorted = top_k[np.argsort(dists[top_k])]
        gt.append(top_k_sorted)
    return np.array(gt, dtype=np.int32)

def evaluate_recall(approx_results, gt_results, k: int = 10) -> float:
    recalls = []
    for i in range(len(approx_results)):
        gt_set = set(gt_results[i][:k])
        approx_set = set([x[0] for x in approx_results[i][:k]])
        hits = len(gt_set.intersection(approx_set))
        recalls.append(hits / k)
    return float(np.mean(recalls))

def run_clustered_benchmark():
    print("=" * 80, flush=True)
    print("CLUSTERED DATA CHECK: Synthetic Gaussian Mixture (100K vectors, 50 clusters)", flush=True)
    print("=" * 80, flush=True)
    
    t0 = time.time()
    base_vecs, query_vecs = generate_gmm_dataset(num_vectors=100_000, num_queries=500, dim=128, num_clusters=50, cluster_std=0.3)
    print(f"Generated GMM dataset: base shape {base_vecs.shape}, query shape {query_vecs.shape} in {time.time() - t0:.2f}s", flush=True)
    
    print("Computing exact brute-force ground truth for 500 queries on 100K vectors...", flush=True)
    t0 = time.time()
    gt = compute_ground_truth(base_vecs, query_vecs, k=10)
    print(f"Ground truth computed in {time.time() - t0:.2f}s", flush=True)
    
    print("Building vsearch index (M=16, efConstruction=100, heuristic=True)...", flush=True)
    cfg = HNSWConfig(M=16, ef_construction=100, heuristic=True)
    idx = HNSWIndex(config=cfg, metric="l2")
    t0 = time.time()
    idx.batch_insert(base_vecs)
    build_time = time.time() - t0
    print(f"vsearch build completed in {build_time:.2f}s ({len(base_vecs) / build_time:.1f} vec/s)", flush=True)
    
    ef_values = [10, 20, 50, 100, 200]
    print("\nEvaluating Recall@10 vs efSearch:", flush=True)
    print(f"{'efSearch':>10} | {'Recall@10':>12} | {'QPS':>10} | {'Avg Latency (ms)':>18}", flush=True)
    print("-" * 58, flush=True)
    
    results = []
    for ef in ef_values:
        lats = []
        approx = []
        for q in query_vecs:
            t_start = time.perf_counter()
            res = idx.search(q, k=10, ef=ef)
            t_end = time.perf_counter()
            lats.append((t_end - t_start) * 1000.0)
            approx.append(res)
            
        qps = len(query_vecs) / (sum(lats) / 1000.0)
        rec = evaluate_recall(approx, gt, k=10)
        avg_lat = float(np.mean(lats))
        print(f"{ef:10d} | {rec * 100:11.2f}% | {qps:10.1f} | {avg_lat:18.3f}", flush=True)
        results.append((ef, rec, qps, avg_lat))
        
    return results

if __name__ == "__main__":
    run_clustered_benchmark()

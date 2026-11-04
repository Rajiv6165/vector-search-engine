import os
import sys
import time
import numpy as np
from typing import Dict, List, Set, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from vsearch.hnsw import HNSWIndex, HNSWConfig
from bench_vs_faiss import load_fvecs

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BASE_VECTORS_FILE = os.path.join(DATA_DIR, "sift", "sift_base.fvecs")
QUERY_VECTORS_FILE = os.path.join(DATA_DIR, "sift", "sift_query.fvecs")

def compute_exact_filtered_ground_truth(
    base_vectors: np.ndarray,
    query_vectors: np.ndarray,
    matching_indices: Set[int],
    k: int = 10
) -> List[List[int]]:
    match_list = sorted(list(matching_indices))
    if not match_list:
        return [[] for _ in range(len(query_vectors))]
    match_vecs = base_vectors[match_list]
    match_norms = np.sum(match_vecs * match_vecs, axis=1)
    gt = []
    k_actual = min(k, len(match_list))
    for q in query_vectors:
        dists = match_norms - 2.0 * np.dot(match_vecs, q)
        if k_actual < len(match_list):
            top_k_idx = np.argpartition(dists, k_actual - 1)[:k_actual]
            sorted_idx = top_k_idx[np.argsort(dists[top_k_idx])]
        else:
            sorted_idx = np.argsort(dists)
        gt.append([match_list[i] for i in sorted_idx])
    return gt

def compute_recall(approx_results: List[List[int]], gt_results: List[List[int]], k: int = 10) -> float:
    recalls = []
    for i in range(len(approx_results)):
        gt_set = set(gt_results[i][:k])
        if not gt_set:
            continue
        approx_set = set(approx_results[i][:k])
        hits = len(gt_set.intersection(approx_set))
        recalls.append(hits / len(gt_set))
    return float(np.mean(recalls)) if recalls else 0.0

def run_filtered_baseline():
    print("Loading 100K SIFT vectors and 500 queries...", flush=True)
    base_vectors = load_fvecs(BASE_VECTORS_FILE, 100_000)
    query_vectors = load_fvecs(QUERY_VECTORS_FILE, 500)
    
    print("Building vsearch index (M=16, efC=100, heuristic=True)...", flush=True)
    cfg = HNSWConfig(M=16, ef_construction=100, heuristic=True)
    index = HNSWIndex(config=cfg, metric="l2")
    t0 = time.time()
    index.batch_insert(base_vectors)
    print(f"Index built in {time.time() - t0:.2f}s", flush=True)
    
    # Define test conditions: Selectivity + Correlation
    # 1. Uncorrelated (uniform hash)
    # 2. Correlated with vector space (based on projection along 1st principal coordinate)
    proj = base_vectors[:, 0]
    sorted_order = np.argsort(proj)
    
    conditions = [
        ("Uncorrelated 50%", {i for i in range(100_000) if i % 2 == 0}, {"sel": "50%"}),
        ("Uncorrelated 10%", {i for i in range(100_000) if i % 10 == 0}, {"sel": "10%"}),
        ("Uncorrelated 1%",  {i for i in range(100_000) if i % 100 == 0}, {"sel": "1%"}),
        ("Uncorrelated 0.1%", {i for i in range(100_000) if i % 1000 == 0}, {"sel": "0.1%"}),
        ("Correlated 50%", set(sorted_order[:50_000]), {"cluster": "C50"}),
        ("Correlated 10%", set(sorted_order[:10_000]), {"cluster": "C10"}),
        ("Correlated 1%",  set(sorted_order[:1_000]),  {"cluster": "C1"}),
        ("Correlated 0.1%", set(sorted_order[:100]),   {"cluster": "C01"}),
    ]
    
    # Store metadata into the index
    print("Assigning metadata attributes...", flush=True)
    for name, match_set, meta_dict in conditions:
        for idx in match_set:
            if idx not in index.metadata_store:
                index.metadata_store[idx] = {}
            index.metadata_store[idx].update(meta_dict)
            
    print("\n" + "="*85, flush=True)
    print("PART B.1 BASELINE: Post-Filtering Performance Across Selectivities (efSearch=50)", flush=True)
    print("="*85, flush=True)
    print(f"{'Condition':<20} | {'Selectivity':<12} | {'Matching':<10} | {'Post-Filter Recall@10':<22} | {'QPS':<10}", flush=True)
    print("-" * 85, flush=True)
    
    results = []
    for name, match_set, meta_dict in conditions:
        # Ground truth
        gt = compute_exact_filtered_ground_truth(base_vectors, query_vectors, match_set, k=10)
        
        # Post-filter search: search with ef=50, then filter
        lats = []
        approx = []
        for q in query_vectors:
            t_start = time.perf_counter()
            res = index.search(q, k=10, ef=50, filter_dict=meta_dict)
            t_end = time.perf_counter()
            lats.append((t_end - t_start) * 1000.0)
            approx.append([node_id for node_id, _, _ in res])
            
        qps = len(query_vectors) / (sum(lats) / 1000.0)
        rec = compute_recall(approx, gt, k=10)
        sel_pct = f"{len(match_set) / 1000:.1f}%"
        print(f"{name:<20} | {sel_pct:<12} | {len(match_set):<10} | {rec * 100:20.2f}% | {qps:10.1f}", flush=True)
        results.append((name, len(match_set), rec, qps))
        
    return results

if __name__ == "__main__":
    run_filtered_baseline()

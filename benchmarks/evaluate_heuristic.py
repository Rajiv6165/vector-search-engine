import os
import sys
import time
import numpy as np
import faiss

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from vsearch.hnsw import HNSWIndex, HNSWConfig

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "sift")
BASE_FILE = os.path.join(DATA_DIR, "sift_base.fvecs")
QUERY_FILE = os.path.join(DATA_DIR, "sift_query.fvecs")

def load_fvecs(filename, n_max):
    return np.fromfile(filename, dtype=np.float32, count=n_max * 129).reshape(n_max, 129)[:, 1:].copy()

def compute_recall(I_approx, I_gt, k=10):
    hits = 0
    total = len(I_approx) * k
    for i in range(len(I_approx)):
        gt_set = set(I_gt[i][:k])
        hits += len(gt_set.intersection(I_approx[i][:k]))
    return hits / total

def get_avg_degree(index, layer=0):
    if layer >= len(index.graphs):
        return 0.0
    degrees = [len(nbrs) for nbrs in index.graphs[layer].values()]
    return float(np.mean(degrees)) if degrees else 0.0

def run_experiment(name, heuristic_enabled, base_vecs, query_vecs, I_gt, M=16, ef_c=100, ef_s=50):
    print(f"\n--- Testing: {name} (heuristic={heuristic_enabled}) ---")
    config = HNSWConfig(
        M=M,
        ef_construction=ef_c,
        ef_search=ef_s,
        heuristic=heuristic_enabled
    )
    idx = HNSWIndex(config=config, metric="l2")

    # Measure build time
    t0 = time.time()
    for i in range(len(base_vecs)):
        idx.insert(i, base_vecs[i])
    build_time = time.time() - t0
    print(f"Build time ({len(base_vecs)} vectors): {build_time:.2f} s")

    # Measure degree
    avg_deg_l0 = get_avg_degree(idx, 0)
    all_degrees = []
    for g in idx.graphs:
        for nbrs in g.values():
            all_degrees.append(len(nbrs))
    avg_deg_all = float(np.mean(all_degrees)) if all_degrees else 0.0
    print(f"Average node degree (Layer 0): {avg_deg_l0:.2f}, (All layers): {avg_deg_all:.2f}")

    # Measure search latency & recall
    latencies = []
    I_approx = []
    for q in query_vecs:
        t_start = time.perf_counter()
        res = idx.search(q, k=10, ef=ef_s)
        latencies.append((time.perf_counter() - t_start) * 1000)
        I_approx.append([node_id for node_id, _, _ in res])

    recall_10 = compute_recall(I_approx, I_gt, 10)
    qps = len(query_vecs) / (sum(latencies) / 1000)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    print(f"Recall@10: {recall_10:.4f} ({recall_10*100:.2f}%)")
    print(f"Search QPS: {qps:.1f} | p50: {p50:.3f}ms | p95: {p95:.3f}ms")

    return {
        "name": name,
        "heuristic": heuristic_enabled,
        "build_time": build_time,
        "avg_degree_l0": avg_deg_l0,
        "avg_degree_all": avg_deg_all,
        "recall_10": recall_10,
        "qps": qps,
        "p50": p50,
        "p95": p95
    }

def main():
    N_VECS = 20_000
    N_QUERIES = 1_000
    print(f"Loading {N_VECS} SIFT base vectors and {N_QUERIES} queries...")
    base_vecs = load_fvecs(BASE_FILE, N_VECS)
    query_vecs = load_fvecs(QUERY_FILE, N_QUERIES)

    print("Computing exact ground truth via FAISS IndexFlatL2...")
    exact_index = faiss.IndexFlatL2(128)
    exact_index.add(base_vecs)
    _, I_gt = exact_index.search(query_vecs, 10)

    # 1. Simple closest M
    res_simple = run_experiment("Closest M", False, base_vecs, query_vecs, I_gt)

    # 2. Diversity-aware heuristic (Algorithm 4)
    res_heuristic = run_experiment("Algorithm 4 Heuristic", True, base_vecs, query_vecs, I_gt)

    print("\n" + "="*70)
    print("HEURISTIC COMPARISON SUMMARY (20,000 SIFT Vectors, M=16, efC=100, efS=50)")
    print("="*70)
    print(f"{'Metric':<25} | {'Closest M':<18} | {'Algorithm 4 Heuristic':<22}")
    print("-"*70)
    print(f"{'Recall@10':<25} | {res_simple['recall_10']*100:<17.2f}% | {res_heuristic['recall_10']*100:<21.2f}%")
    print(f"{'Build Time (s)':<25} | {res_simple['build_time']:<18.2f} | {res_heuristic['build_time']:<22.2f}")
    print(f"{'Avg Degree (Layer 0)':<25} | {res_simple['avg_degree_l0']:<18.2f} | {res_heuristic['avg_degree_l0']:<22.2f}")
    print(f"{'Search QPS':<25} | {res_simple['qps']:<18.1f} | {res_heuristic['qps']:<22.1f}")
    print(f"{'p95 Latency (ms)':<25} | {res_simple['p95']:<18.3f} | {res_heuristic['p95']:<22.3f}")
    print("="*70)

if __name__ == "__main__":
    main()

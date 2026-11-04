import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from vsearch.hnsw import HNSWIndex, HNSWConfig, greedy_search_layer_native, search_layer_native
from bench_vs_faiss import load_fvecs

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BASE_VECTORS_FILE = os.path.join(DATA_DIR, "sift", "sift_base.fvecs")
QUERY_VECTORS_FILE = os.path.join(DATA_DIR, "sift", "sift_query.fvecs")
SNAP_DIR = os.path.join(DATA_DIR, "sift_index_snap")

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

def measure_post_filter(index: HNSWIndex, query_vectors: np.ndarray, k: int, ef: int, filter_dict: dict):
    latencies = []
    results = []
    for q in query_vectors:
        t0 = time.perf_counter()
        raw = index.search(q, k=ef, ef=ef, filter_dict=None)
        res = [item for item in raw if index._matches_filter(item[0], filter_dict)][:k]
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        results.append([node_id for node_id, _, _ in res])
    qps = len(query_vectors) / (sum(latencies) / 1000.0)
    p95_lat = float(np.percentile(latencies, 95))
    return results, qps, p95_lat

def measure_in_graph_custom(index: HNSWIndex, query_vectors: np.ndarray, k: int, ef: int, filter_dict: dict, visit_budget: int):
    latencies = []
    results = []
    entry = index._get_cached_filter(filter_dict)
    allowed_mask = entry["allowed_mask"]
    allowed_set = entry["allowed_set"]
    vec_matrix = index.nodes.get_matrix()
    idx_to_id = index.nodes.index_to_node_id

    for q in query_vectors:
        t0 = time.perf_counter()
        q_c = np.ascontiguousarray(q, dtype=np.float32)
        curr_obj = index.entry_point
        curr_dist = index.distance_fn(q_c, index.nodes[curr_obj])
        curr_idx = index.nodes.node_id_to_index[curr_obj]
        for lc in range(index.max_level, 0, -1):
            curr_idx, curr_dist = greedy_search_layer_native(
                q_c, curr_idx, curr_dist, vec_matrix,
                index.layer_edges[lc], index.layer_degrees[lc], index.metric_code
            )
        raw_res = search_layer_native(
            q_c, np.array([curr_idx], dtype=np.int32), ef, vec_matrix,
            index.layer_edges[0], index.layer_degrees[0], index.metric_code,
            return_visited=False, allowed=allowed_mask, visit_budget=visit_budget
        )
        W = [(idx_to_id[i], d) for i, d in raw_res if i in idx_to_id]
        W.sort(key=lambda x: x[1])
        res = []
        for n_id, dist in W:
            if n_id not in index.deleted_nodes and n_id in allowed_set:
                res.append(n_id)
                if len(res) == k:
                    break
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        results.append(res)
    qps = len(query_vectors) / (sum(latencies) / 1000.0)
    p95_lat = float(np.percentile(latencies, 95))
    return results, qps, p95_lat

def measure_brute_force(index: HNSWIndex, query_vectors: np.ndarray, k: int, filter_dict: dict):
    latencies = []
    results = []
    entry = index._get_cached_filter(filter_dict)
    allowed_set = entry["allowed_set"]
    matching_indices = entry["matching_indices"]
    active_matching_ids = entry["active_matching_ids"]

    for q in query_vectors:
        t0 = time.perf_counter()
        res = index._search_brute_force_matching(
            q, allowed_set, k=k,
            matching_indices=matching_indices,
            active_matching_ids=active_matching_ids
        )
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        results.append([node_id for node_id, _, _ in res])
    qps = len(query_vectors) / (sum(latencies) / 1000.0)
    p95_lat = float(np.percentile(latencies, 95))
    return results, qps, p95_lat

def measure_production_search(index: HNSWIndex, query_vectors: np.ndarray, k: int, ef: int, filter_dict: dict):
    latencies = []
    results = []
    for q in query_vectors:
        t0 = time.perf_counter()
        res = index.search(q, k=k, ef=ef, filter_dict=filter_dict)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        results.append([node_id for node_id, _, _ in res])
    qps = len(query_vectors) / (sum(latencies) / 1000.0)
    p95_lat = float(np.percentile(latencies, 95))
    return results, qps, p95_lat

def main():
    print("Loading 100K SIFT vectors and 200 queries...", flush=True)
    base_vectors = load_fvecs(BASE_VECTORS_FILE, 100_000)
    query_vectors = load_fvecs(QUERY_VECTORS_FILE, 200)

    if os.path.exists(os.path.join(SNAP_DIR, "graph.bin")):
        print(f"Loading index snapshot from {SNAP_DIR}...", flush=True)
        t0 = time.time()
        index = HNSWIndex.load_snapshot(SNAP_DIR)
        print(f"Snapshot loaded in {time.time() - t0:.2f}s", flush=True)
    else:
        print("Building vsearch index in RAM (M=16, efC=100, heuristic=True)...", flush=True)
        cfg = HNSWConfig(M=16, ef_construction=100, heuristic=True)
        index = HNSWIndex(config=cfg, metric="l2")
        t0 = time.time()
        index.batch_insert(base_vectors)
        print(f"Index built in {time.time() - t0:.2f}s. Saving snapshot...", flush=True)
        os.makedirs(SNAP_DIR, exist_ok=True)
        index.save_snapshot(SNAP_DIR)

    # 1. Sweep crossover threshold with cached bitmaps & indices
    print("\n" + "="*105, flush=True)
    print("PART B.4: Crossover Threshold Sweep (Cached Brute-Force Fallback vs Cached In-Graph Traversal)", flush=True)
    print("="*105, flush=True)
    print(f"{'Matches (M)':<12} | {'Brute-Force QPS':<18} | {'In-Graph (ef=50) QPS':<22} | {'BF p95 (ms)':<14} | {'IG p95 (ms)':<14} | {'Winner':<12}", flush=True)
    print("-" * 105, flush=True)

    sweep_sizes = [25, 50, 100, 250, 500, 1000, 1500, 2000, 3000, 5000, 10000]
    for sz in sweep_sizes:
        tag = f"sw_{sz}"
        for i in range(sz):
            if i not in index.metadata_store:
                index.metadata_store[i] = {}
            index.metadata_store[i][tag] = "1"
            index.metadata_index.insert(i, index.metadata_store[i])

        fdict = {tag: "1"}
        # Warm cache
        index._get_cached_filter(fdict)

        _, qps_bf, p95_bf = measure_brute_force(index, query_vectors, k=10, filter_dict=fdict)
        _, qps_ig, p95_ig = measure_in_graph_custom(index, query_vectors, k=10, ef=50, filter_dict=fdict, visit_budget=2000)

        winner = "Brute-Force" if qps_bf > qps_ig else "In-Graph"
        print(f"{sz:<12} | {qps_bf:15.1f} QPS | {qps_ig:19.1f} QPS | {p95_bf:10.3f} ms | {p95_ig:10.3f} ms | {winner:<12}", flush=True)

    # Setup Selectivity Conditions: 50%, 10%, 1%, 0.1% (Uncorrelated & Correlated)
    centroid = np.mean(base_vectors[:10_000], axis=0)
    dists_to_c = np.linalg.norm(base_vectors - centroid, axis=1)
    sorted_order = np.argsort(dists_to_c)

    conditions = [
        ("Uncorrelated 50%", 0.500, {i for i in range(100_000) if i % 2 == 0}, {"sel_50": "yes"}),
        ("Uncorrelated 10%", 0.100, {i for i in range(100_000) if i % 10 == 0}, {"sel_10": "yes"}),
        ("Uncorrelated 1%",  0.010, {i for i in range(100_000) if i % 100 == 0}, {"sel_1": "yes"}),
        ("Uncorrelated 0.1%",0.001, {i for i in range(100_000) if i % 1000 == 0}, {"sel_01": "yes"}),
        ("Correlated 50%",   0.500, set(sorted_order[:50_000]), {"clust_50": "yes"}),
        ("Correlated 10%",   0.100, set(sorted_order[:10_000]), {"clust_10": "yes"}),
        ("Correlated 1%",    0.010, set(sorted_order[:1_000]),  {"clust_1": "yes"}),
        ("Correlated 0.1%",  0.001, set(sorted_order[:100]),    {"clust_01": "yes"}),
    ]

    for name, sel, match_set, meta_dict in conditions:
        for idx in match_set:
            if idx not in index.metadata_store:
                index.metadata_store[idx] = {}
            index.metadata_store[idx].update(meta_dict)
            index.metadata_index.insert(idx, index.metadata_store[idx])

    # 2. Benchmark Across Selectivities with Adaptive Post-Filter & In-Graph Sweeps
    print("\n" + "="*120, flush=True)
    print("PART B.8: Filtered Search Benchmark across Selectivities (Equal-Recall Comparison)", flush=True)
    print("="*120, flush=True)

    curve_data = {}
    summary_table = []

    k = 10
    for name, sel, match_set, meta_dict in conditions:
        print(f"\n>>> Running Evaluation for {name} ({len(match_set)} matches, {sel*100:.1f}% selectivity)...", flush=True)
        gt = compute_exact_filtered_ground_truth(base_vectors, query_vectors, match_set, k=k)

        # Warm cache
        index._get_cached_filter(meta_dict)

        # A. Adaptive Post-Filter Sweep: ef = min(10000, max(k, round(c * k / sel))) with c in [1, 2, 4, 8, 16]
        c_multipliers = [1, 2, 4, 8, 16]
        pf_curves = []
        for c in c_multipliers:
            ef_val = min(10_000, max(k, int(round(c * k / sel))))
            res_pf, qps_pf, p95_pf = measure_post_filter(index, query_vectors, k=k, ef=ef_val, filter_dict=meta_dict)
            rec_pf = compute_recall(res_pf, gt, k=k)
            pf_curves.append((rec_pf, qps_pf, p95_pf, ef_val, c))

        # B. In-Graph Traversal Sweep: budgets [500, 1000, 2000, 5000, 10000] with ef=50
        budgets = [500, 1000, 2000, 5000, 10000]
        ig_curves = []
        for b in budgets:
            res_ig, qps_ig, p95_ig = measure_in_graph_custom(index, query_vectors, k=k, ef=50, filter_dict=meta_dict, visit_budget=b)
            rec_ig = compute_recall(res_ig, gt, k=k)
            ig_curves.append((rec_ig, qps_ig, p95_ig, b))

        # C. Brute Force Fallback (Exact)
        res_bf, qps_bf, p95_bf = measure_brute_force(index, query_vectors, k=k, filter_dict=meta_dict)
        rec_bf = compute_recall(res_bf, gt, k=k)

        # D. Production Engine (Hybrid with Fallback threshold=2000 + Escalation)
        res_prod, qps_prod, p95_prod = measure_production_search(index, query_vectors, k=k, ef=50, filter_dict=meta_dict)
        rec_prod = compute_recall(res_prod, gt, k=k)

        curve_data[name] = {
            "sel": sel,
            "pf": pf_curves,
            "ig": ig_curves,
            "bf": (rec_bf, qps_bf, p95_bf),
            "prod": (rec_prod, qps_prod, p95_prod)
        }

        # Pick best post-filter with highest recall, and among ties, highest QPS
        best_pf = max(pf_curves, key=lambda x: (round(x[0], 3), x[1]))
        # Pick in-graph at default budget=2000
        default_ig = ig_curves[2]  # b=2000

        print(f"  Adaptive Post-Filter (c={best_pf[4]}, ef={best_pf[3]}): Recall={best_pf[0]*100:.2f}%, QPS={best_pf[1]:.1f}, p95={best_pf[2]:.2f}ms")
        print(f"  In-Graph (budget={default_ig[3]}, ef=50):               Recall={default_ig[0]*100:.2f}%, QPS={default_ig[1]:.1f}, p95={default_ig[2]:.2f}ms")
        print(f"  Brute-Force Fallback (Exact):                        Recall={rec_bf*100:.2f}%, QPS={qps_bf:.1f}, p95={p95_bf:.2f}ms")
        print(f"  Production Engine (Hybrid + Escalation):             Recall={rec_prod*100:.2f}%, QPS={qps_prod:.1f}, p95={p95_prod:.2f}ms")

        summary_table.append({
            "name": name,
            "matches": len(match_set),
            "pf": best_pf,
            "ig": default_ig,
            "bf": (rec_bf, qps_bf, p95_bf),
            "prod": (rec_prod, qps_prod, p95_prod)
        })

    # Print Full Comparative Table
    print("\n" + "="*125, flush=True)
    print("SUMMARY COMPARISON TABLE: RECALL, QPS & LATENCY ACROSS STRATEGIES (k=10)")
    print("="*125, flush=True)
    print(f"{'Condition':<18} | {'Matches':<8} | {'Adaptive PF (Rec/QPS/p95)':<28} | {'In-Graph (Rec/QPS/p95)':<28} | {'Brute-Force (Rec/QPS)':<24} | {'Engine (Rec/QPS)'}", flush=True)
    print("-" * 125, flush=True)
    for row in summary_table:
        pf_str = f"{row['pf'][0]*100:5.1f}% / {row['pf'][1]:6.1f} / {row['pf'][2]:5.2f}ms"
        ig_str = f"{row['ig'][0]*100:5.1f}% / {row['ig'][1]:6.1f} / {row['ig'][2]:5.2f}ms"
        bf_str = f"{row['bf'][0]*100:5.1f}% / {row['bf'][1]:6.1f} QPS"
        prod_str = f"{row['prod'][0]*100:5.1f}% / {row['prod'][1]:6.1f} QPS"
        print(f"{row['name']:<18} | {row['matches']:<8} | {pf_str:<28} | {ig_str:<28} | {bf_str:<24} | {prod_str}", flush=True)

    # Plot Recall vs QPS for all conditions
    plot_curves(curve_data)

def plot_curves(curve_data: dict):
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle("Filtered Search: Recall vs QPS Trade-offs Across Strategies (100K SIFT)", fontsize=16, fontweight='bold')

    order = [
        ("Uncorrelated 50%", 0, 0), ("Uncorrelated 10%", 0, 1), ("Uncorrelated 1%", 0, 2), ("Uncorrelated 0.1%", 0, 3),
        ("Correlated 50%", 1, 0),   ("Correlated 10%", 1, 1),   ("Correlated 1%", 1, 2),   ("Correlated 0.1%", 1, 3)
    ]

    for title, row, col in order:
        ax = axes[row, col]
        data = curve_data[title]

        # Post filter curve
        pf_rec = [p[0] * 100 for p in data["pf"]]
        pf_qps = [p[1] for p in data["pf"]]
        ax.plot(pf_rec, pf_qps, marker='o', color='#d62728', lw=2, label='Adaptive Post-Filter')

        # In-graph curve
        ig_rec = [p[0] * 100 for p in data["ig"]]
        ig_qps = [p[1] for p in data["ig"]]
        ax.plot(ig_rec, ig_qps, marker='s', color='#1f77b4', lw=2, label='In-Graph Traversal')

        # Brute force point
        bf_rec = data["bf"][0] * 100
        bf_qps = data["bf"][1]
        ax.scatter([bf_rec], [bf_qps], marker='*', color='#ff7f0e', s=160, zorder=5, label='Brute-Force Fallback')

        # Engine point
        prod_rec = data["prod"][0] * 100
        prod_qps = data["prod"][1]
        ax.scatter([prod_rec], [prod_qps], marker='D', color='#2ca02c', s=100, zorder=6, label='Production Engine')

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Recall@10 (%)", fontsize=10)
        ax.set_ylabel("Throughput (QPS)", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_yscale('log')
        if row == 0 and col == 0:
            ax.legend(loc='lower left', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(os.path.dirname(__file__), "filtered_recall_vs_qps.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"\nMulti-panel Recall vs QPS plot saved to {out_path}", flush=True)

if __name__ == "__main__":
    main()

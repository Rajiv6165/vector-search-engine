import os
import sys
import time
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from vsearch.hnsw import HNSWIndex, HNSWConfig, greedy_search_layer_native, search_layer_native
from bench_vs_faiss import load_fvecs

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BASE_VECTORS_FILE = os.path.join(DATA_DIR, "sift", "sift_base.fvecs")
QUERY_VECTORS_FILE = os.path.join(DATA_DIR, "sift", "sift_query.fvecs")

def profile_filtered_search():
    print("Loading 100K SIFT vectors and 100 queries...", flush=True)
    base_vectors = load_fvecs(BASE_VECTORS_FILE, 100_000)
    query_vectors = load_fvecs(QUERY_VECTORS_FILE, 100)

    SNAP_DIR = os.path.join(DATA_DIR, "sift_index_snap")
    if os.path.exists(os.path.join(SNAP_DIR, "graph.bin")):
        print(f"Loading existing snapshot from {SNAP_DIR}...", flush=True)
        t0 = time.time()
        index = HNSWIndex.load_snapshot(SNAP_DIR)
        print(f"Index loaded in {time.time() - t0:.2f}s", flush=True)
    else:
        print("Building index in RAM and saving snapshot...", flush=True)
        cfg = HNSWConfig(M=16, ef_construction=100, heuristic=True)
        index = HNSWIndex(config=cfg, metric="l2")
        t0 = time.time()
        index.batch_insert(base_vectors)
        print(f"Index built in {time.time() - t0:.2f}s. Saving snapshot to {SNAP_DIR}...", flush=True)
        os.makedirs(SNAP_DIR, exist_ok=True)
        index.save_snapshot(SNAP_DIR)
        print("Snapshot saved.", flush=True)

    # Setup 10K and 50K metadata
    for i in range(50_000):
        meta = {"sel_50k": "yes"}
        if i < 10_000:
            meta["sel_10k"] = "yes"
        index.metadata_store[i] = meta
        index.metadata_index.insert(i, meta)

    test_cases = [
        ("10K Matches (10% Selectivity)", {"sel_10k": "yes"}),
        ("50K Matches (50% Selectivity)", {"sel_50k": "yes"})
    ]

    print("\n" + "="*95, flush=True)
    print("PROFILING FILTERED SEARCH LATENCY BREAKDOWN (Averaged over 100 queries, ef=50, k=10)")
    print("="*95, flush=True)

    for label, fdict in test_cases:
        t_meta_lookup = []
        t_allowed_set = []
        t_bitmap_alloc = []
        t_bitmap_populate = []
        t_greedy = []
        t_search_layer = []
        t_format = []
        t_total = []

        k = 10
        ef = 50

        for q in query_vectors:
            q_c = np.ascontiguousarray(q, dtype=np.float32)
            t_start = time.perf_counter()

            # 1. Metadata lookup
            t0 = time.perf_counter()
            matching_ids = index.metadata_index.get_matching_ids(fdict)
            t1 = time.perf_counter()
            t_meta_lookup.append((t1 - t0) * 1e6)

            # 2. Allowed set filtering
            t0 = time.perf_counter()
            allowed_set = {nid for nid in matching_ids if nid not in index.deleted_nodes}
            t1 = time.perf_counter()
            t_allowed_set.append((t1 - t0) * 1e6)

            # 3. Bitmap allocation
            t0 = time.perf_counter()
            num_vectors = index.nodes.get_matrix().shape[0]
            allowed_mask = np.zeros(num_vectors, dtype=np.uint8)
            t1 = time.perf_counter()
            t_bitmap_alloc.append((t1 - t0) * 1e6)

            # 4. Bitmap population
            t0 = time.perf_counter()
            id_to_idx = index.nodes.node_id_to_index
            for nid in allowed_set:
                idx = id_to_idx.get(nid)
                if idx is not None and idx < num_vectors:
                    allowed_mask[idx] = 1
            t1 = time.perf_counter()
            t_bitmap_populate.append((t1 - t0) * 1e6)

            # 5. Greedy upper layer search
            t0 = time.perf_counter()
            curr_obj = index.entry_point
            curr_dist = index.distance_fn(q_c, index.nodes[curr_obj])
            vec_matrix = index.nodes.get_matrix()
            curr_idx = index.nodes.node_id_to_index[curr_obj]
            for lc in range(index.max_level, 0, -1):
                curr_idx, curr_dist = greedy_search_layer_native(
                    q_c, curr_idx, curr_dist, vec_matrix,
                    index.layer_edges[lc], index.layer_degrees[lc], index.metric_code
                )
            curr_obj = index.nodes.index_to_node_id[curr_idx]
            t1 = time.perf_counter()
            t_greedy.append((t1 - t0) * 1e6)

            # 6. Native search layer (graph traversal + distance math)
            t0 = time.perf_counter()
            ep_indices = np.array([curr_idx], dtype=np.int32)
            raw_res = search_layer_native(
                q_c, ep_indices, ef, vec_matrix,
                index.layer_edges[0], index.layer_degrees[0], index.metric_code,
                return_visited=False, allowed=allowed_mask, visit_budget=max(2000, ef * 20)
            )
            t1 = time.perf_counter()
            t_search_layer.append((t1 - t0) * 1e6)

            # 7. Format results
            t0 = time.perf_counter()
            idx_to_id = index.nodes.index_to_node_id
            W = [(idx_to_id[i], d) for i, d in raw_res if i in idx_to_id]
            W.sort(key=lambda x: x[1])
            results = []
            for n_id, dist in W:
                if n_id not in index.deleted_nodes and n_id in allowed_set:
                    results.append((n_id, dist, index.metadata_store.get(n_id, {})))
                    if len(results) == k:
                        break
            t1 = time.perf_counter()
            t_format.append((t1 - t0) * 1e6)

            t_end = time.perf_counter()
            t_total.append((t_end - t_start) * 1e6)

        avg_total = np.mean(t_total)
        print(f"\n--- {label} ---")
        print(f"Total Query Latency: {avg_total / 1000.0:.3f} ms ({1e6 / avg_total:.1f} QPS)")
        print(f"{'Operation':<35} | {'Avg Time (us)':<15} | {'% of Total':<10}")
        print("-" * 65)

        sub_steps = [
            ("1. Metadata index lookup", np.mean(t_meta_lookup)),
            ("2. Allowed set creation", np.mean(t_allowed_set)),
            ("3. Bitmap allocation (np.zeros)", np.mean(t_bitmap_alloc)),
            ("4. Bitmap population (loop)", np.mean(t_bitmap_populate)),
            ("5. Upper layers greedy descent", np.mean(t_greedy)),
            ("6. In-graph traversal + SIMD dist", np.mean(t_search_layer)),
            ("7. Result formatting & sorting", np.mean(t_format)),
        ]

        conversion_overhead = np.mean(t_meta_lookup) + np.mean(t_allowed_set) + np.mean(t_bitmap_alloc) + np.mean(t_bitmap_populate)
        for name, us in sub_steps:
            pct = (us / avg_total) * 100.0
            print(f"{name:<35} | {us:15.2f} us | {pct:8.2f}%")
        print("-" * 65)
        print(f"{'TOTAL CONVERSION OVERHEAD (1-4)':<35} | {conversion_overhead:15.2f} us | {(conversion_overhead / avg_total)*100:8.2f}%")
        print(f"{'ACTUAL GRAPH TRAVERSAL (5-6)':<35} | {(np.mean(t_greedy) + np.mean(t_search_layer)):15.2f} us | {((np.mean(t_greedy) + np.mean(t_search_layer)) / avg_total)*100:8.2f}%")

if __name__ == "__main__":
    profile_filtered_search()


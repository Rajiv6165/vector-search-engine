"""
scripts/seed_demo.py
Generate and insert ~20,000 clustered vectors with a 'category' metadata field in batches.
Reports actual RSS after seeding.
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
from typing import Dict, Any, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("vsearch.seed")

def get_process_rss_mb() -> float:
    """Return the current process RSS in Megabytes."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        pass
    
    # Linux procfs fallback
    try:
        if os.path.exists("/proc/self/statm"):
            with open("/proc/self/statm") as f:
                pages = int(f.read().split()[1])
                return (pages * os.sysconf("SC_PAGE_SIZE")) / (1024.0 * 1024.0)
    except Exception:
        pass

    # Windows fallback
    try:
        import ctypes
        from ctypes import wintypes
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        fn = ctypes.windll.psapi.GetProcessMemoryInfo
        fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
        fn.restype = wintypes.BOOL
        handle = wintypes.HANDLE(ctypes.windll.kernel32.GetCurrentProcess())
        if fn(handle, ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize / (1024.0 * 1024.0)
    except Exception:
        pass

    return 0.0

CATEGORIES = ["electronics", "clothing", "home", "books", "sports"]

def generate_demo_dataset(total_vectors: int = 20000, dim: int = 64, seed: int = 42) -> Tuple[np.ndarray, List[Dict[str, Any]], List[int]]:
    """Generate synthetic Gaussian-mixture clustered vectors with metadata."""
    rng = np.random.default_rng(seed)
    num_categories = len(CATEGORIES)
    vecs_per_cat = total_vectors // num_categories
    
    # Generate well-separated cluster centers
    centers = rng.normal(size=(num_categories, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    centers *= 3.0  # Separate centers

    all_vectors = []
    all_metadatas = []
    all_ids = []
    
    node_id = 1
    for cat_idx, category in enumerate(CATEGORIES):
        count = vecs_per_cat if cat_idx < num_categories - 1 else total_vectors - len(all_ids)
        # Cluster noise
        noise = rng.normal(loc=0.0, scale=0.4, size=(count, dim)).astype(np.float32)
        vectors = centers[cat_idx] + noise
        
        for i in range(count):
            all_vectors.append(vectors[i])
            all_metadatas.append({
                "category": category,
                "cluster_id": cat_idx,
                "price": float(round(float(rng.uniform(10.0, 500.0)), 2)),
                "rating": float(round(float(rng.uniform(1.0, 5.0)), 1))
            })
            all_ids.append(node_id)
            node_id += 1

    return np.array(all_vectors, dtype=np.float32), all_metadatas, all_ids

def seed_index(index, total_vectors: int = 20000, dim: int = 64, batch_size: int = 1000, progress_callback=None) -> float:
    """Seed an in-memory or persisted HNSWIndex directly in batches and return post-seeding RSS in MB."""
    rss_start = get_process_rss_mb()
    logger.info(f"Generating demo dataset ({total_vectors} vectors, dim={dim}, {len(CATEGORIES)} categories)...")
    vectors, metadatas, node_ids = generate_demo_dataset(total_vectors, dim=dim)
    
    logger.info(f"Seeding HNSW index in batches of {batch_size}...")
    start_time = time.perf_counter()
    
    for i in range(0, total_vectors, batch_size):
        b_vecs = vectors[i:i + batch_size]
        b_meta = metadatas[i:i + batch_size]
        b_ids = node_ids[i:i + batch_size]
        index.batch_insert(b_vecs, node_ids=b_ids, metadatas=b_meta)
        current_inserted = min(i + batch_size, total_vectors)
        if progress_callback:
            progress_callback(current_inserted / float(total_vectors))
        if current_inserted % 5000 == 0 or current_inserted >= total_vectors:
            logger.info(f"Inserted {current_inserted} / {total_vectors} vectors...")
            
    elapsed = time.perf_counter() - start_time
    rss_end = get_process_rss_mb()
    logger.info(
        f"Seeding complete: {total_vectors} vectors indexed in {elapsed:.2f}s "
        f"({total_vectors / elapsed:.1f} vecs/sec). "
        f"Process RSS: {rss_end:.2f} MB (Delta: +{rss_end - rss_start:.2f} MB)."
    )
    return rss_end

def seed_via_api(url: str, api_key: str = None, total_vectors: int = 20000, dim: int = 64, batch_size: int = 1000):
    """Seed a running API server via HTTP POST /vectors/batch."""
    import urllib.request
    import json

    logger.info(f"Generating {total_vectors} vectors for remote seeding into {url}...")
    vectors, metadatas, node_ids = generate_demo_dataset(total_vectors, dim=dim)
    
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
        
    start_time = time.perf_counter()
    inserted = 0
    for i in range(0, total_vectors, batch_size):
        b_vecs = vectors[i:i + batch_size].tolist()
        b_meta = metadatas[i:i + batch_size]
        b_ids = node_ids[i:i + batch_size]
        
        payload = json.dumps({
            "vectors": b_vecs,
            "node_ids": b_ids,
            "metadatas": b_meta
        }).encode("utf-8")
        
        req = urllib.request.Request(f"{url.rstrip('/')}/vectors/batch", data=payload, headers=headers)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            inserted += data.get("inserted_count", 0)
            
        if (i + batch_size) % 5000 == 0 or (i + batch_size) >= total_vectors:
            logger.info(f"Remote indexed: {inserted} / {total_vectors} vectors...")

    elapsed = time.perf_counter() - start_time
    logger.info(f"Remote seeding finished: {inserted} vectors inserted in {elapsed:.2f}s ({inserted / elapsed:.1f} vecs/sec).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed vsearch demo dataset")
    parser.add_argument("--url", type=str, default=None, help="Remote server URL (e.g. http://localhost:8000)")
    parser.add_argument("--api-key", type=str, default=None, help="Write API key if auth is required")
    parser.add_argument("--count", type=int, default=20000, help="Total vectors to insert (default: 20000)")
    parser.add_argument("--dim", type=int, default=64, help="Vector dimension (default: 64)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size (default: 1000)")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save snapshot after seeding")
    args = parser.parse_args()

    if args.url:
        seed_via_api(args.url, api_key=args.api_key, total_vectors=args.count, dim=args.dim, batch_size=args.batch_size)
    else:
        # Local standalone index seeding
        from vsearch.hnsw import HNSWIndex, HNSWConfig
        index = HNSWIndex(config=HNSWConfig(M=16, ef_construction=100, ef_search=50))
        seed_index(index, total_vectors=args.count, dim=args.dim, batch_size=args.batch_size)
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            index.save_snapshot(path=args.save_dir)
            logger.info(f"Snapshot successfully saved to {args.save_dir}")

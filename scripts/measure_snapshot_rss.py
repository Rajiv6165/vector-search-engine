"""
scripts/measure_snapshot_rss.py
Measures peak process RSS during HNSWIndex snapshot serialization at VSEARCH_MAX_VECTORS.
Evaluates memory headroom against 512 MB and 1024 MB VM targets.
"""

import os
import sys
import time
import tempfile
import threading
import numpy as np
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("vsearch.rss")

from scripts.seed_demo import get_process_rss_mb, generate_demo_dataset
from vsearch.hnsw import HNSWIndex, HNSWConfig

class RSSSampler:
    def __init__(self, interval: float = 0.005):
        self.interval = interval
        self.peak_rss = 0.0
        self.base_rss = 0.0
        self._running = False
        self._thread = None

    def start(self):
        self.base_rss = get_process_rss_mb()
        self.peak_rss = self.base_rss
        self._running = True
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self):
        while self._running:
            current = get_process_rss_mb()
            if current > self.peak_rss:
                self.peak_rss = current
            time.sleep(self.interval)

    def stop(self) -> float:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        final_sample = get_process_rss_mb()
        if final_sample > self.peak_rss:
            self.peak_rss = final_sample
        return self.peak_rss

def measure_snapshot_rss(num_vectors: int = 20000, dim: int = 64):
    temp_dir = tempfile.mkdtemp(prefix="vsearch_snapshot_rss_")
    logger.info(f"Generating {num_vectors} vectors (dim={dim}) for snapshot RSS profiling...")
    vectors, metadatas, node_ids = generate_demo_dataset(total_vectors=num_vectors, dim=dim)

    logger.info("Building HNSW index...")
    config = HNSWConfig(M=16, ef_construction=100, ef_search=50)
    index = HNSWIndex(config=config, persist_dir=temp_dir)
    
    # Batch insert
    batch_size = 1000
    for i in range(0, num_vectors, batch_size):
        index.batch_insert(
            vectors[i:i + batch_size],
            node_ids=node_ids[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size]
        )

    rss_before_snapshot = get_process_rss_mb()
    logger.info(f"Index built ({num_vectors} vectors). Resident memory before snapshot: {rss_before_snapshot:.2f} MB")

    # Start peak sampler
    sampler = RSSSampler(interval=0.002)
    sampler.start()
    t0 = time.perf_counter()
    index.save_snapshot()
    duration = time.perf_counter() - t0
    peak_rss = sampler.stop()

    delta_mb = peak_rss - rss_before_snapshot
    headroom_512 = 512.0 - peak_rss
    headroom_1024 = 1024.0 - peak_rss

    print("\n" + "=" * 70)
    print(f"SNAPSHOT PEAK MEMORY REPORT (N={num_vectors}, dim={dim})")
    print("=" * 70)
    print(f"Base RSS before snapshot:   {rss_before_snapshot:.2f} MB")
    print(f"Peak RSS during snapshot:   {peak_rss:.2f} MB")
    print(f"Peak Memory Spike:          +{delta_mb:.2f} MB")
    print(f"Snapshot serialization time:{duration:.3f} s")
    print(f"Remaining Headroom in 512MB: {headroom_512:.2f} MB ({headroom_512 / 512.0 * 100:.1f}% free)")
    print(f"Remaining Headroom in 1024MB:{headroom_1024:.2f} MB ({headroom_1024 / 1024.0 * 1024.0 * 100 / 1024.0:.1f}% free)")
    print("=" * 70)

    # Cleanup
    if hasattr(index.nodes, "close"):
        index.nodes.close()
    if index.wal:
        index.wal.close()
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

    return peak_rss, headroom_512

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    measure_snapshot_rss(n)

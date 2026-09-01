import os
import time
import shutil
import tempfile
import numpy as np
from src.vsearch.hnsw import HNSWIndex, HNSWConfig

def benchmark_persistence():
    np.random.seed(42)
    N = 10_000
    dim = 64
    
    print(f"Generating {N} random vectors of dimension {dim}...")
    vectors = np.random.rand(N, dim).astype(np.float32)
    
    config = HNSWConfig(M=16, ef_construction=100)
    
    # 1. Baseline: In-memory Insert
    print("\n--- Baseline: In-Memory ---")
    mem_index = HNSWIndex(config=config)
    start_time = time.time()
    for i in range(N):
        mem_index.insert(i, vectors[i])
    mem_insert_time = time.time() - start_time
    print(f"In-memory insert time for {N} vectors: {mem_insert_time:.2f} seconds")
    
    # 2. Persistence: WAL Insert
    print("\n--- Phase 3: Persistent (WAL + Memmap) ---")
    temp_dir = tempfile.mkdtemp()
    try:
        pers_index = HNSWIndex(config=config, persist_dir=temp_dir)
        start_time = time.time()
        for i in range(N):
            pers_index.insert(i, vectors[i])
        pers_insert_time = time.time() - start_time
        print(f"Persistent insert time for {N} vectors: {pers_insert_time:.2f} seconds")
        overhead = (pers_insert_time - mem_insert_time) / mem_insert_time * 100
        print(f"WAL Overhead: +{overhead:.1f}%")
        
        # 3. Snapshot Save
        print("\n--- Snapshotting ---")
        start_time = time.time()
        pers_index.save_snapshot()
        save_time = time.time() - start_time
        print(f"Save snapshot time for {N} vectors: {save_time:.2f} seconds")
        
        pers_index.nodes.close()
        pers_index.wal.close()
        
        # 4. Snapshot Load
        start_time = time.time()
        loaded_index = HNSWIndex.load_snapshot(path=temp_dir, persist_dir=temp_dir)
        load_time = time.time() - start_time
        print(f"Load snapshot time for {N} vectors: {load_time:.2f} seconds")
        
        loaded_index.nodes.close()
        loaded_index.wal.close()
        
    finally:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    benchmark_persistence()

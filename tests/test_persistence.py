import os
import shutil
import tempfile
import numpy as np
import pytest
from src.vsearch.hnsw import HNSWIndex, HNSWConfig

def test_crash_recovery_wal():
    # 1. Create temporary directory for persist_dir
    temp_dir = tempfile.mkdtemp()
    try:
        config = HNSWConfig(M=16, ef_construction=100)
        index = HNSWIndex(config=config, persist_dir=temp_dir)
        
        dim = 16
        np.random.seed(42)
        
        # Insert first 50 vectors
        base_vectors = {}
        for i in range(50):
            vec = np.random.rand(dim).astype(np.float32)
            base_vectors[i] = vec
            index.insert(i, vec)
            
        # Save snapshot
        index.save_snapshot()
        
        # Insert 50 more vectors (these will only be in WAL, not in snapshot)
        for i in range(50, 100):
            vec = np.random.rand(dim).astype(np.float32)
            base_vectors[i] = vec
            index.insert(i, vec)
            
        # Delete some vectors to test WAL delete
        index.delete(10)
        index.delete(60)
        del base_vectors[10]
        del base_vectors[60]
        
        # Close to flush to disk (simulating crash)
        index.nodes.close()
        index.wal.close()
        
        # 2. Reload index from snapshot + WAL
        recovered_index = HNSWIndex.load_snapshot(path=temp_dir, persist_dir=temp_dir)
        
        # 3. Verify consistency
        assert len(recovered_index.nodes.node_id_to_index) == len(base_vectors)
        for node_id, original_vec in base_vectors.items():
            assert node_id in recovered_index.nodes
            recovered_vec = recovered_index.nodes[node_id]
            np.testing.assert_array_almost_equal(original_vec, recovered_vec)
            
        # Verify graph integrity by running a search
        query = np.random.rand(dim).astype(np.float32)
        results = recovered_index.search(query, k=5)
        assert len(results) == 5
        
        recovered_index.nodes.close()
        recovered_index.wal.close()
    finally:
        shutil.rmtree(temp_dir)

def test_snapshot_roundtrip_recall():
    temp_dir = tempfile.mkdtemp()
    try:
        config = HNSWConfig(M=8, ef_construction=50)
        index = HNSWIndex(config=config, persist_dir=temp_dir)
        
        dim = 16
        np.random.seed(42)
        
        # Insert 200 vectors
        vectors = []
        for i in range(200):
            vec = np.random.rand(dim).astype(np.float32)
            vectors.append(vec)
            index.insert(i, vec)
            
        queries = [np.random.rand(dim).astype(np.float32) for _ in range(10)]
        
        # Get original search results
        original_results = []
        for q in queries:
            res = index.search(q, k=10)
            original_results.append([node_id for node_id, _ in res])
            
        # Save and close
        index.save_snapshot()
        index.nodes.close()
        index.wal.close()
        
        # Reload
        reloaded = HNSWIndex.load_snapshot(path=temp_dir, persist_dir=temp_dir)
        
        # Get reloaded search results
        reloaded_results = []
        for q in queries:
            res = reloaded.search(q, k=10)
            reloaded_results.append([node_id for node_id, _ in res])
            
        # Compare
        for orig, reload in zip(original_results, reloaded_results):
            assert orig == reload, f"Search results differ: {orig} != {reload}"
            
        reloaded.nodes.close()
        reloaded.wal.close()
    finally:
        shutil.rmtree(temp_dir)

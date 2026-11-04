import os
import shutil
import tempfile
import numpy as np
import pytest
from vsearch.hnsw import HNSWIndex, HNSWConfig

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
        assert len(recovered_index.nodes.node_id_to_index) - len(recovered_index.deleted_nodes) == len(base_vectors)
        for node_id, original_vec in base_vectors.items():
            assert node_id in recovered_index.nodes and node_id not in recovered_index.deleted_nodes
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
            original_results.append([node_id for node_id, _, _ in res])
            
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
            reloaded_results.append([node_id for node_id, _, _ in res])
            
        # Compare
        for orig, reload in zip(original_results, reloaded_results):
            assert orig == reload, f"Search results differ: {orig} != {reload}"
            
        reloaded.nodes.close()
        reloaded.wal.close()
    finally:
        shutil.rmtree(temp_dir)

def test_cached_filter_bitmap_wal_replay_and_incremental():
    temp_dir = tempfile.mkdtemp()
    try:
        config = HNSWConfig(M=8, ef_construction=50)
        index = HNSWIndex(config=config, persist_dir=temp_dir)
        dim = 8
        np.random.seed(42)

        # 1. Insert initial vectors with metadata
        for i in range(50):
            vec = np.random.rand(dim).astype(np.float32)
            color = "blue" if i % 2 == 0 else "red"
            index.insert(i, vec, metadata={"color": color, "batch": 1})

        # Warm filter cache for color=blue
        fdict = {"color": "blue"}
        q = np.random.rand(dim).astype(np.float32)
        res = index.search(q, k=5, filter_dict=fdict)
        assert len(res) == 5
        for nid, _, meta in res:
            assert meta.get("color") == "blue"

        # Verify cached structures
        cache_key = tuple(sorted(fdict.items()))
        assert cache_key in index._filter_cache
        entry = index._filter_cache[cache_key]
        assert len(entry["allowed_set"]) == 25
        for nid in entry["allowed_set"]:
            idx = index.nodes.node_id_to_index[nid]
            assert entry["allowed_mask"][idx] == 1

        # 2. Incrementally insert a matching vector (node_id=100, blue)
        vec_100 = np.random.rand(dim).astype(np.float32)
        index.insert(100, vec_100, metadata={"color": "blue", "batch": 2})
        idx_100 = index.nodes.node_id_to_index[100]
        assert 100 in entry["allowed_set"]
        assert entry["allowed_mask"][idx_100] == 1

        # 3. Incrementally insert a non-matching vector (node_id=101, red)
        vec_101 = np.random.rand(dim).astype(np.float32)
        index.insert(101, vec_101, metadata={"color": "red", "batch": 2})
        idx_101 = index.nodes.node_id_to_index[101]
        assert 101 not in entry["allowed_set"]
        assert entry["allowed_mask"][idx_101] == 0

        # 4. Incrementally delete a matching vector (node_id=0, blue)
        idx_0 = index.nodes.node_id_to_index[0]
        index.delete(0)
        assert 0 not in entry["allowed_set"]
        assert entry["allowed_mask"][idx_0] == 0

        # Save snapshot
        index.save_snapshot()

        # 5. Insert and delete more via WAL (post-snapshot)
        vec_102 = np.random.rand(dim).astype(np.float32)
        index.insert(102, vec_102, metadata={"color": "blue", "batch": 3})
        index.delete(2) # another blue node

        # Flush and close to simulate crash
        index.nodes.close()
        index.wal.close()

        # 6. Replay from snapshot + WAL
        reloaded = HNSWIndex.load_snapshot(path=temp_dir, persist_dir=temp_dir)

        # Search reloaded index with filter
        res_reloaded = reloaded.search(q, k=10, filter_dict=fdict)
        returned_ids = {nid for nid, _, _ in res_reloaded}
        assert 0 not in returned_ids, "Deleted node 0 returned after WAL replay"
        assert 2 not in returned_ids, "Deleted node 2 returned after WAL replay"
        assert 100 in reloaded.metadata_index.get_matching_ids(fdict)
        assert 102 in reloaded.metadata_index.get_matching_ids(fdict)

        # Verify reloaded cache entry
        cache_entry = reloaded._get_cached_filter(fdict)
        assert 0 not in cache_entry["allowed_set"]
        assert 2 not in cache_entry["allowed_set"]
        assert 100 in cache_entry["allowed_set"]
        assert 102 in cache_entry["allowed_set"]
        idx_102 = reloaded.nodes.node_id_to_index[102]
        assert cache_entry["allowed_mask"][idx_102] == 1

        reloaded.nodes.close()
        reloaded.wal.close()
    finally:
        shutil.rmtree(temp_dir)

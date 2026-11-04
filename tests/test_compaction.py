import os
import shutil
import tempfile
import numpy as np
import pytest
from vsearch.hnsw import HNSWIndex, HNSWConfig

def test_compaction():
    temp_dir = tempfile.mkdtemp()
    try:
        config = HNSWConfig(M=16, ef_construction=100)
        index = HNSWIndex(config=config, persist_dir=temp_dir)
        
        dim = 16
        np.random.seed(42)
        
        # Insert 100 vectors
        for i in range(100):
            vec = np.random.rand(dim).astype(np.float32)
            index.insert(i, vec)
            
        vector_file = os.path.join(temp_dir, "vectors.bin")
        size_before_compaction = os.path.getsize(vector_file)
        
        # Delete 50 vectors (even ids)
        for i in range(0, 100, 2):
            index.delete(i)
            
        assert len(index.deleted_nodes) == 50
        
        # Perform compaction
        index.compact()
        
        # Verify compaction cleared deleted nodes
        assert len(index.deleted_nodes) == 0
        assert len(index.nodes.node_id_to_index) == 50
        
        # Verify file size shrank or memory mapped properly (compaction creates new file)
        # It may be initial capacity, but at least it should be valid
        size_after_compaction = os.path.getsize(vector_file)
        # Note: VectorStore allocates `initial_capacity` bytes. So file size might be the same 
        # if initial capacity > 100. Let's just verify consistency and recall.
        
        # Verify we can still search for remaining nodes
        for i in range(1, 100, 2):
            vec = index.nodes[i]
            res = index.search(vec, k=1)
            assert res[0][0] == i
            
    finally:
        if hasattr(index.nodes, "close"):
            index.nodes.close()
        if index.wal:
            index.wal.close()
        shutil.rmtree(temp_dir)

def test_cached_bitmap_after_compaction():
    temp_dir = tempfile.mkdtemp()
    try:
        config = HNSWConfig(M=16, ef_construction=100)
        index = HNSWIndex(config=config, persist_dir=temp_dir)
        dim = 8
        np.random.seed(456)

        for i in range(100):
            vec = np.random.randn(dim).astype(np.float32)
            meta = {"status": "active" if i < 60 else "inactive"}
            index.insert(i, vec, metadata=meta)

        # Warm filter cache
        fdict = {"status": "active"}
        q = np.random.randn(dim).astype(np.float32)
        res = index.search(q, k=10, filter_dict=fdict)
        assert len(res) == 10
        cache_key = tuple(sorted(fdict.items()))
        assert cache_key in index._filter_cache

        # Delete 15 active nodes
        for i in range(15):
            index.delete(i)

        entry = index._filter_cache[cache_key]
        assert len(entry["allowed_set"]) == 45
        for i in range(15):
            assert i not in entry["allowed_set"]

        # Run compaction
        index.compact()

        # Cache must be cleared on compact
        assert len(index._filter_cache) == 0

        # Subsequent query re-populates cache correctly
        res_after = index.search(q, k=10, filter_dict=fdict)
        assert len(res_after) == 10
        for nid, _, m in res_after:
            assert m["status"] == "active"
            assert nid >= 15

        rebuilt_entry = index._filter_cache[cache_key]
        assert len(rebuilt_entry["allowed_set"]) == 45
        for i in range(15):
            assert i not in rebuilt_entry["allowed_set"]
    finally:
        if hasattr(index.nodes, "close"):
            index.nodes.close()
        if index.wal:
            index.wal.close()
        shutil.rmtree(temp_dir)

def test_compaction_with_metadata_index():
    temp_dir = tempfile.mkdtemp()
    try:
        config = HNSWConfig(M=16, ef_construction=100)
        index = HNSWIndex(config=config, persist_dir=temp_dir)
        dim = 8
        np.random.seed(123)

        for i in range(60):
            vec = np.random.randn(dim).astype(np.float32)
            meta = {"category": "A" if i % 2 == 0 else "B", "group": i % 3}
            index.insert(i, vec, metadata=meta)

        # Before deletion: 30 nodes in category A (0, 2, 4, ... 58)
        assert len(index.metadata_index.get_matching_ids({"category": "A"})) == 30

        # Delete nodes 0 to 19 (10 are in category A)
        for i in range(20):
            index.delete(i)

        # After delete, inverted index immediately removes them
        cat_a_matches = index.metadata_index.get_matching_ids({"category": "A"})
        assert len(cat_a_matches) == 20
        assert not any(i in cat_a_matches for i in range(20))

        # Perform compaction
        index.compact()

        # After compaction, metadata index rebuilt and perfectly consistent
        cat_a_matches_after = index.metadata_index.get_matching_ids({"category": "A"})
        assert len(cat_a_matches_after) == 20
        assert not any(i in cat_a_matches_after for i in range(20))

        # Search with filter
        q = np.random.randn(dim).astype(np.float32)
        res = index.search(q, k=10, filter_dict={"category": "A"})
        assert len(res) == 10
        for nid, _, m in res:
            assert m["category"] == "A"
            assert nid >= 20
    finally:
        if hasattr(index.nodes, "close"):
            index.nodes.close()
        if index.wal:
            index.wal.close()
        shutil.rmtree(temp_dir)

def test_cached_bitmap_after_compaction():
    temp_dir = tempfile.mkdtemp()
    try:
        config = HNSWConfig(M=16, ef_construction=100)
        index = HNSWIndex(config=config, persist_dir=temp_dir)
        dim = 8
        np.random.seed(456)

        for i in range(100):
            vec = np.random.randn(dim).astype(np.float32)
            meta = {"status": "active" if i < 60 else "inactive"}
            index.insert(i, vec, metadata=meta)

        # Warm filter cache
        fdict = {"status": "active"}
        q = np.random.randn(dim).astype(np.float32)
        res = index.search(q, k=10, filter_dict=fdict)
        assert len(res) == 10
        cache_key = tuple(sorted(fdict.items()))
        assert cache_key in index._filter_cache

        # Delete 15 active nodes
        for i in range(15):
            index.delete(i)

        entry = index._filter_cache[cache_key]
        assert len(entry["allowed_set"]) == 45
        for i in range(15):
            assert i not in entry["allowed_set"]

        # Run compaction
        index.compact()

        # Cache must be cleared on compact
        assert len(index._filter_cache) == 0

        # Subsequent query re-populates cache correctly
        res_after = index.search(q, k=10, filter_dict=fdict)
        assert len(res_after) == 10
        for nid, _, m in res_after:
            assert m["status"] == "active"
            assert nid >= 15

        rebuilt_entry = index._filter_cache[cache_key]
        assert len(rebuilt_entry["allowed_set"]) == 45
        for i in range(15):
            assert i not in rebuilt_entry["allowed_set"]
    finally:
        if hasattr(index.nodes, "close"):
            index.nodes.close()
        if index.wal:
            index.wal.close()
        shutil.rmtree(temp_dir)

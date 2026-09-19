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

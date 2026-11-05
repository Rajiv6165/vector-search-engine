import pytest
import time
import numpy as np
from vsearch.hnsw import HNSWIndex, HNSWConfig, FilterCache, FilterCacheEntry

def test_filter_cache_lru_eviction():
    cache = FilterCache(max_entries=3, max_bytes=10 * 1024 * 1024)
    for i in range(3):
        entry = FilterCacheEntry(
            field="f", value=f"v_{i}",
            allowed_mask=np.zeros(10, dtype=np.uint8),
            allowed_set={i},
            matching_indices=np.array([i], dtype=np.int32),
            active_matching_ids=[i]
        )
        cache.put(("f", f"v_{i}"), entry)

    assert len(cache) == 3
    assert ("f", "v_0") in cache
    assert ("f", "v_1") in cache
    assert ("f", "v_2") in cache

    # Access v_0 to make it recently used
    _ = cache.get(("f", "v_0"))

    # Add v_3, which should evict v_1 (the least recently used)
    entry3 = FilterCacheEntry(
        field="f", value="v_3",
        allowed_mask=np.zeros(10, dtype=np.uint8),
        allowed_set={3},
        matching_indices=np.array([3], dtype=np.int32),
        active_matching_ids=[3]
    )
    cache.put(("f", "v_3"), entry3)

    assert len(cache) == 3
    assert ("f", "v_0") in cache
    assert ("f", "v_1") not in cache, "v_1 should have been evicted as LRU"
    assert ("f", "v_2") in cache
    assert ("f", "v_3") in cache


def test_filter_cache_byte_budget_eviction():
    # Set a tiny byte budget of 1000 bytes
    cache = FilterCache(max_entries=100, max_bytes=1000)
    
    # Each entry with 600 bytes
    mask = np.zeros(500, dtype=np.uint8)
    e1 = FilterCacheEntry("f", "v1", mask, {1}, np.array([1], dtype=np.int32), [1])
    cache.put(("f", "v1"), e1)
    assert len(cache) == 1

    e2 = FilterCacheEntry("f", "v2", mask, {2}, np.array([2], dtype=np.int32), [2])
    cache.put(("f", "v2"), e2)

    # Total bytes would exceed 1000, so e1 must be evicted
    assert len(cache) == 1
    assert ("f", "v1") not in cache
    assert ("f", "v2") in cache


def test_multifield_filter_and_combination():
    cfg = HNSWConfig(M=8, ef_construction=50)
    index = HNSWIndex(config=cfg)
    dim = 4
    np.random.seed(42)

    data = [
        (0, {"dept": "eng", "role": "lead"}),
        (1, {"dept": "eng", "role": "intern"}),
        (2, {"dept": "sales", "role": "lead"}),
        (3, {"dept": "sales", "role": "rep"}),
        (4, {"dept": "eng", "role": "lead"}),
    ]

    for nid, meta in data:
        vec = np.random.randn(dim).astype(np.float32)
        index.insert(nid, vec, metadata=meta)

    # Search with multi-field filter
    q = np.random.randn(dim).astype(np.float32)
    res = index.search(q, k=10, filter_dict={"dept": "eng", "role": "lead"})
    returned_ids = {r[0] for r in res}
    assert returned_ids == {0, 4}

    # Verify per-(field, value) entries exist in cache
    assert ("dept", "eng") in index._filter_cache
    assert ("role", "lead") in index._filter_cache


def test_high_cardinality_insert_latency():
    cfg = HNSWConfig(M=8, ef_construction=50)
    index = HNSWIndex(config=cfg)
    dim = 8
    np.random.seed(42)

    # Insert 1000 items with unique user IDs
    vectors = np.random.randn(1000, dim).astype(np.float32)
    metas = [{"user_id": f"user_{i}", "group": "test"} for i in range(1000)]
    index.batch_insert(vectors, metadatas=metas)

    # Warm cache with 500 distinct filters
    q = np.random.randn(dim).astype(np.float32)
    for i in range(500):
        index.search(q, k=1, filter_dict={"user_id": f"user_{i}"})

    assert len(index._filter_cache) == 500

    # Measure insert latency with 500 cached filters
    t_inserts = []
    for i in range(1000, 1050):
        new_vec = np.random.randn(dim).astype(np.float32)
        t0 = time.perf_counter()
        index.insert(i, new_vec, metadata={"user_id": f"user_{i}", "group": "test"})
        t1 = time.perf_counter()
        t_inserts.append((t1 - t0) * 1e6)

    avg_insert_us = np.mean(t_inserts)
    # Insert should remain fast and not loop over the 500 cached filters
    assert avg_insert_us < 2000.0, f"Insert latency too high: {avg_insert_us:.2f} us"


def test_escalation_instrumentation():
    cfg = HNSWConfig(M=8, ef_construction=50)
    index = HNSWIndex(config=cfg)
    dim = 4
    np.random.seed(42)

    for i in range(100):
        vec = np.random.randn(dim).astype(np.float32)
        index.insert(i, vec, metadata={"tier": "gold" if i < 10 else "bronze"})

    index.reset_escalation_count()
    assert index.get_escalation_count() == 0

    # Filter with 10 matching vectors
    q = np.random.randn(dim).astype(np.float32)
    res = index.search(q, k=5, filter_dict={"tier": "gold"})
    assert len(res) == 5
    for _, _, m in res:
        assert m["tier"] == "gold"


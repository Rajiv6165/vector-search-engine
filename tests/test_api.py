import os
import sys
import shutil
import tempfile
import pytest

# Disable automatic 20k vector seeding during fast unit tests
os.environ["VSEARCH_AUTO_SEED"] = "0"
os.environ["VSEARCH_ALLOW_INSECURE"] = "1"

from fastapi.testclient import TestClient
import vsearch.api.app as api_app

def test_api():
    temp_dir = tempfile.mkdtemp()
    os.environ["VSEARCH_DATA_DIR"] = temp_dir
    os.environ["VSEARCH_PERSIST_DIR"] = temp_dir
    os.environ["VSEARCH_ALLOW_INSECURE"] = "1"
    os.environ.pop("VSEARCH_API_KEY", None)
    os.environ.pop("VSEARCH_READ_KEY", None)
    
    api_app.PERSIST_DIR = temp_dir
    api_app.VSEARCH_API_KEY = None
    api_app.VSEARCH_READ_KEY = None
    api_app.VSEARCH_ALLOW_INSECURE = True
    api_app.MAX_VECTORS = None
    api_app.MAX_K = 1000
    api_app.MAX_EF = 500
    api_app.MAX_BATCH = 1000
    
    app = api_app.app
    
    try:
        with TestClient(app) as client:
            # 1. Healthcheck endpoint (Unauthenticated)
            resp = client.get("/health")
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}
            
            # 2. Stats endpoint (native flag, version, ready)
            resp = client.get("/stats")
            assert resp.status_code == 200
            stats_data = resp.json()
            assert stats_data["size"] == 0
            assert "native" in stats_data
            assert isinstance(stats_data["native"], bool)
            assert "version" in stats_data
            assert stats_data["version"] == "0.1.0"
            assert stats_data["ready"] is True
            
            # 3. Vector Insert and Search
            resp = client.post("/vectors", json={
                "node_id": 1,
                "vector": [0.1, 0.2, 0.3, 0.4],
                "metadata": {"category": "electronics", "price": 100}
            })
            assert resp.status_code == 200
            
            resp = client.post("/vectors", json={
                "node_id": 2,
                "vector": [0.1, 0.2, 0.3, 0.9],
                "metadata": {"category": "clothing"}
            })
            assert resp.status_code == 200
            
            resp = client.post("/search", json={
                "vector": [0.1, 0.2, 0.3, 0.5],
                "k": 2
            })
            assert resp.status_code == 200
            results = resp.json()["results"]
            assert len(results) == 2
            assert results[0]["node_id"] == 1
            
            # 4. Filtered Search
            resp = client.post("/search", json={
                "vector": [0.1, 0.2, 0.3, 0.5],
                "k": 2,
                "filter": {"category": "clothing"}
            })
            assert resp.status_code == 200
            results = resp.json()["results"]
            assert len(results) == 1
            assert results[0]["node_id"] == 2
            
            # 5. Delete
            resp = client.delete("/vectors/1")
            assert resp.status_code == 200
            resp = client.post("/search", json={"vector": [0.1, 0.2, 0.3, 0.5], "k": 2})
            assert len(resp.json()["results"]) == 1
            
            # 6. Dimension and Vector Validation
            resp = client.post("/vectors", json={"node_id": 3, "vector": [0.1, 0.2]})
            assert resp.status_code == 400
            assert "dimension mismatch" in resp.json()["detail"].lower()

            resp = client.post("/vectors", json={"node_id": 3, "vector": []})
            assert resp.status_code == 400
            assert "empty" in resp.json()["detail"].lower()

            resp = client.post("/vectors", json={"node_id": 3, "vector": [0.1, "invalid", 0.3, 0.4]})
            assert resp.status_code == 422

            # Filter validation
            resp = client.post("/search", json={
                "vector": [0.1, 0.2, 0.3, 0.5],
                "k": 2,
                "filter": {"category": ["electronics", "appliances"]}
            })
            assert resp.status_code == 422

            # Batch operations
            resp = client.post("/vectors/batch", json={
                "vectors": [[0.5, 0.5, 0.5, 0.5], [0.6, 0.6, 0.6, 0.6]],
                "node_ids": [10, 11],
                "metadatas": [{"category": "appliances"}, {"category": "clothing"}]
            })
            assert resp.status_code == 200
            assert resp.json()["inserted_count"] == 2

            resp = client.post("/search/batch", json={
                "vectors": [[0.5, 0.5, 0.5, 0.5], [0.1, 0.2, 0.3, 0.9]],
                "k": 1
            })
            assert resp.status_code == 200
            batch_res = resp.json()["results"]
            assert len(batch_res) == 2
            assert batch_res[0][0]["node_id"] == 10
            assert batch_res[1][0]["node_id"] == 2

        # Test reload lifecycle
        with TestClient(app) as client2:
            resp = client2.get("/stats")
            assert resp.status_code == 200
            data = resp.json()
            assert data["size"] == 3
            assert data["dim"] == 4
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_api_auth():
    """Verify write key and read key permissions with secrets.compare_digest."""
    temp_dir = tempfile.mkdtemp()
    api_app.PERSIST_DIR = temp_dir
    api_app.VSEARCH_API_KEY = "secret-write-key"
    api_app.VSEARCH_READ_KEY = "private-read-key"
    api_app.VSEARCH_PUBLIC_KEY = "public-demo-key"
    os.environ["VSEARCH_API_KEY"] = "secret-write-key"
    os.environ["VSEARCH_READ_KEY"] = "private-read-key"
    os.environ["VSEARCH_PUBLIC_KEY"] = "public-demo-key"
    os.environ.pop("VSEARCH_ALLOW_INSECURE", None)
    
    app = api_app.app
    try:
        with TestClient(app) as client:
            # 1. Health is always public
            assert client.get("/health").status_code == 200
            
            # 2. Stats requires read or write key; rejects public demo key
            assert client.get("/stats").status_code == 401
            assert client.get("/stats", headers={"X-API-Key": "wrong"}).status_code == 401
            assert client.get("/stats", headers={"X-API-Key": "public-demo-key"}).status_code == 401
            assert client.get("/stats", headers={"X-API-Key": "private-read-key"}).status_code == 200
            assert client.get("/stats", headers={"X-API-Key": "secret-write-key"}).status_code == 200
            
            # 3. Write endpoints strictly reject read key and public key
            for key in ["private-read-key", "public-demo-key"]:
                resp = client.post(
                    "/vectors",
                    json={"node_id": 1, "vector": [1.0, 2.0, 3.0, 4.0]},
                    headers={"X-API-Key": key}
                )
                assert resp.status_code == 401
                assert "write API Key" in resp.json()["detail"]

                resp = client.post(
                    "/vectors/batch",
                    json={"vectors": [[1.0, 2.0, 3.0, 4.0]], "node_ids": [1]},
                    headers={"X-API-Key": key}
                )
                assert resp.status_code == 401

                resp = client.delete("/vectors/1", headers={"X-API-Key": key})
                assert resp.status_code == 401

            # 4. Write key succeeds on write endpoints
            resp = client.post(
                "/vectors",
                json={"node_id": 1, "vector": [1.0, 2.0, 3.0, 4.0], "metadata": {"category": "electronics"}},
                headers={"X-API-Key": "secret-write-key"}
            )
            assert resp.status_code == 200
            
            # 5. Search accepts read key, but strictly rejects public demo key
            assert client.post(
                "/search",
                json={"vector": [1.0, 2.0, 3.0, 4.0], "k": 1},
                headers={"X-API-Key": "public-demo-key"}
            ).status_code == 401

            assert client.post(
                "/search/batch",
                json={"vectors": [[1.0, 2.0, 3.0, 4.0]], "k": 1},
                headers={"X-API-Key": "public-demo-key"}
            ).status_code == 401

            resp = client.post(
                "/search",
                json={"vector": [1.0, 2.0, 3.0, 4.0], "k": 1},
                headers={"X-API-Key": "private-read-key"}
            )
            assert resp.status_code == 200
            assert len(resp.json()["results"]) == 1

            # 6. Demo routes accept public demo key
            resp = client.get("/similar/1", headers={"X-API-Key": "public-demo-key"})
            assert resp.status_code == 200

            resp = client.get("/demo/random", headers={"X-API-Key": "public-demo-key"})
            assert resp.status_code == 200

            # 7. Delete with write key
            resp = client.delete("/vectors/1", headers={"X-API-Key": "secret-write-key"})
            assert resp.status_code == 200
    finally:
        api_app.VSEARCH_API_KEY = None
        api_app.VSEARCH_READ_KEY = None
        api_app.VSEARCH_PUBLIC_KEY = None
        os.environ["VSEARCH_ALLOW_INSECURE"] = "1"
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_auth_fails_closed():
    """Verify server refuses to start without VSEARCH_API_KEY unless VSEARCH_ALLOW_INSECURE=1."""
    temp_dir = tempfile.mkdtemp()
    api_app.PERSIST_DIR = temp_dir
    os.environ.pop("VSEARCH_API_KEY", None)
    os.environ["VSEARCH_ALLOW_INSECURE"] = "0"
    api_app.VSEARCH_API_KEY = None
    api_app.VSEARCH_ALLOW_INSECURE = False

    app = api_app.app
    try:
        with pytest.raises(RuntimeError) as exc_info:
            with TestClient(app):
                pass
        assert "VSEARCH_API_KEY is not set" in str(exc_info.value)
    finally:
        os.environ["VSEARCH_ALLOW_INSECURE"] = "1"
        api_app.VSEARCH_ALLOW_INSECURE = True
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_api_limits():
    """Verify MAX_VECTORS, MAX_K, MAX_EF, MAX_BATCH, IP rate limiting, and request body size caps."""
    temp_dir = tempfile.mkdtemp()
    api_app.PERSIST_DIR = temp_dir
    api_app.VSEARCH_API_KEY = None
    api_app.VSEARCH_READ_KEY = None
    api_app.VSEARCH_ALLOW_INSECURE = True
    os.environ["VSEARCH_ALLOW_INSECURE"] = "1"
    
    api_app.MAX_VECTORS = 3
    api_app.MAX_K = 5
    api_app.MAX_EF = 50
    api_app.MAX_BATCH = 2
    api_app.MAX_REQUEST_BYTES = 1000
    api_app.rate_limiter.max_requests = 100
    
    app = api_app.app
    try:
        with TestClient(app) as client:
            # Insert up to limit (3 vectors)
            for i in range(1, 4):
                resp = client.post("/vectors", json={"node_id": i, "vector": [0.1, 0.2, 0.3, 0.4]})
                assert resp.status_code == 200
                
            # 4th insert should be rejected by MAX_VECTORS
            resp = client.post("/vectors", json={"node_id": 4, "vector": [0.1, 0.2, 0.3, 0.4]})
            assert resp.status_code == 400
            assert "MAX_VECTORS" in resp.json()["detail"]
            
            # Delete one vector to free capacity (3 -> 2)
            resp = client.delete("/vectors/1")
            assert resp.status_code == 200

            # Now inserting node 4 succeeds (2 -> 3)
            resp = client.post("/vectors", json={"node_id": 4, "vector": [0.1, 0.2, 0.3, 0.4]})
            assert resp.status_code == 200

            # Inserting node 5 is rejected by MAX_VECTORS
            resp = client.post("/vectors", json={"node_id": 5, "vector": [0.1, 0.2, 0.3, 0.4]})
            assert resp.status_code == 400
            assert "MAX_VECTORS" in resp.json()["detail"]
            
            # Batch insert exceeding MAX_BATCH
            resp = client.post("/vectors/batch", json={
                "vectors": [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5], [0.3, 0.4, 0.5, 0.6]],
                "node_ids": [10, 11, 12]
            })
            assert resp.status_code == 400
            assert "MAX_BATCH" in resp.json()["detail"]
            
            # Search exceeding MAX_K
            resp = client.post("/search", json={"vector": [0.1, 0.2, 0.3, 0.4], "k": 10})
            assert resp.status_code == 400
            assert "MAX_K" in resp.json()["detail"]

            # Search exceeding MAX_EF
            resp = client.post("/search", json={"vector": [0.1, 0.2, 0.3, 0.4], "k": 2, "ef": 100})
            assert resp.status_code == 400
            assert "MAX_EF" in resp.json()["detail"]

            # Batch search exceeding MAX_BATCH
            resp = client.post("/search/batch", json={
                "vectors": [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5], [0.3, 0.4, 0.5, 0.6]],
                "k": 2
            })
            assert resp.status_code == 400
            assert "MAX_BATCH" in resp.json()["detail"]
            
            # Request body size cap
            resp = client.post(
                "/search",
                content=b"x" * 2000,
                headers={"Content-Length": "2000", "Content-Type": "application/json"}
            )
            assert resp.status_code == 413
            assert "maximum allowed size" in resp.json()["detail"]

            # Per-IP rate limiting test
            api_app.rate_limiter.max_requests = 2
            api_app.rate_limiter.window_seconds = 10.0
            api_app.rate_limiter._requests.clear()
            assert client.post("/search", json={"vector": [0.1, 0.2, 0.3, 0.4], "k": 1}).status_code == 200
            assert client.post("/search", json={"vector": [0.1, 0.2, 0.3, 0.4], "k": 1}).status_code == 200
            rate_resp = client.post("/search", json={"vector": [0.1, 0.2, 0.3, 0.4], "k": 1})
            assert rate_resp.status_code == 429
            assert "Rate limit exceeded" in rate_resp.json()["detail"]

            # Requests from a different IP via X-Forwarded-For should succeed
            diff_ip_resp = client.post("/search", json={"vector": [0.1, 0.2, 0.3, 0.4], "k": 1}, headers={"X-Forwarded-For": "198.51.100.42"})
            assert diff_ip_resp.status_code == 200
    finally:
        api_app.MAX_VECTORS = None
        api_app.MAX_K = 1000
        api_app.MAX_EF = 500
        api_app.MAX_BATCH = 1000
        api_app.MAX_REQUEST_BYTES = 10 * 1024 * 1024
        api_app.rate_limiter.max_requests = 100
        api_app.rate_limiter.window_seconds = 1.0
        api_app.rate_limiter._requests.clear()
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_exclusive_data_dir_lock():
    """Verify exclusive lock file in VSEARCH_DATA_DIR prevents second process from attaching."""
    temp_dir = tempfile.mkdtemp()
    api_app.PERSIST_DIR = temp_dir
    os.environ["VSEARCH_ALLOW_INSECURE"] = "1"
    api_app.VSEARCH_ALLOW_INSECURE = True
    
    app = api_app.app
    try:
        with TestClient(app):
            # First instance is running and holds the lock.
            # Attempt to acquire lock from a second lock instance on the same directory
            second_lock = api_app.DataDirLock(temp_dir)
            with pytest.raises(RuntimeError) as exc_info:
                second_lock.acquire()
            assert "Another vsearch process is currently running" in str(exc_info.value)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_single_process_safety():
    """Verify engine refuses to start if WEB_CONCURRENCY > 1."""
    os.environ["WEB_CONCURRENCY"] = "2"
    try:
        with pytest.raises(RuntimeError) as exc_info:
            api_app.check_single_process_safety()
        assert "WEB_CONCURRENCY=2" in str(exc_info.value)
    finally:
        os.environ.pop("WEB_CONCURRENCY", None)

def test_seed_demo():
    """Verify seed_demo generates clustered vectors with categories and measures RSS."""
    from scripts.seed_demo import generate_demo_dataset, seed_index
    from vsearch.hnsw import HNSWIndex, HNSWConfig

    # 1. Dataset generation validation
    vectors, metadatas, node_ids = generate_demo_dataset(total_vectors=250, dim=32)
    assert len(vectors) == 250
    assert len(metadatas) == 250
    assert len(node_ids) == 250
    assert vectors.shape == (250, 32)
    categories = set(m["category"] for m in metadatas)
    assert len(categories) == 5
    assert "electronics" in categories

    # 2. Index batch insertion
    index = HNSWIndex(config=HNSWConfig(M=8, ef_construction=50, ef_search=20))
    final_rss = seed_index(index, total_vectors=250, dim=32, batch_size=50)
    assert len(index.nodes) == 250
    assert final_rss >= 0.0

    # 3. Filtered query verification
    query = vectors[0]
    results = index.search(query, k=5, filter_dict={"category": "electronics"})
    assert len(results) > 0
    assert all(m["category"] == "electronics" for _, _, m in results)

def test_demo_endpoints():
    """Verify Phase 12 endpoints: GET /, GET /similar/{node_id}, GET /demo/random, auth, k cap, unknown id."""
    from vsearch.hnsw import HNSWIndex, HNSWConfig
    from fastapi import HTTPException
    import numpy as np

    temp_dir = tempfile.mkdtemp()
    try:
        api_app.PERSIST_DIR = temp_dir
        config = HNSWConfig(M=16, ef_construction=100, ef_search=50)
        test_index = HNSWIndex(config=config, persist_dir=temp_dir)

        # Insert 10 vectors across 2 categories
        rng = np.random.default_rng(42)
        for i in range(1, 11):
            cat = "electronics" if i <= 5 else "clothing"
            vec = rng.normal(size=16).astype(np.float32)
            test_index.insert(i, vec, metadata={"category": cat, "item_id": i})

        api_app.index = test_index
        api_app.is_ready = True
        api_app.MAX_K = 10
        api_app.VSEARCH_API_KEY = "write-key-123"
        api_app.VSEARCH_READ_KEY = "read-key-123"
        api_app.VSEARCH_PUBLIC_KEY = "public-key-123"
        api_app.VSEARCH_ALLOW_INSECURE = False

        # 1. GET / (HTML response)
        html_resp = api_app.get_demo_page()
        assert html_resp.status_code == 200
        html_content = html_resp.body.decode("utf-8")
        assert "vsearch Demo" in html_content
        assert "Pick random item" in html_content
        assert "/docs" in html_content
        assert "20,000 synthetic 64-d vectors" in html_content
        assert "public-key-123" in html_content

        # 2. Auth: verify_demo_key vs verify_write_key
        assert api_app.verify_demo_key("public-key-123") == "public-key-123"
        assert api_app.verify_demo_key("read-key-123") == "read-key-123"
        assert api_app.verify_demo_key("write-key-123") == "write-key-123"
        with pytest.raises(HTTPException) as exc:
            api_app.verify_demo_key("wrong-key")
        assert exc.value.status_code == 401

        # Public key must NOT have read key access to /search, /search/batch, /stats
        with pytest.raises(HTTPException) as exc:
            api_app.verify_read_key("public-key-123")
        assert exc.value.status_code == 401

        # Public key must NOT have write access
        with pytest.raises(HTTPException) as exc:
            api_app.verify_write_key("public-key-123")
        assert exc.value.status_code == 401

        # 3. GET /similar/{node_id} success
        sim_res = api_app.get_similar_vectors(node_id=1, k=5, category=None, request=None, api_key="public-key-123")
        assert sim_res.node_id == 1
        assert len(sim_res.results) == 5
        assert sim_res.latency_ms >= 0.0
        assert sim_res.search_path in ("graph", "exact_fallback")
        assert isinstance(sim_res.used_exact_fallback, bool)

        # 4. GET /similar/{node_id} with category filter
        sim_cat = api_app.get_similar_vectors(node_id=1, k=3, category="electronics", request=None, api_key="public-key-123")
        assert len(sim_cat.results) == 3
        assert all(r.metadata["category"] == "electronics" for r in sim_cat.results)

        # 5. GET /similar/{node_id} with unknown id -> 404
        with pytest.raises(HTTPException) as exc:
            api_app.get_similar_vectors(node_id=999999, k=5, category=None, request=None, api_key="public-key-123")
        assert exc.value.status_code == 404
        assert "Node 999999 not found" in exc.value.detail

        # 6. GET /similar/{node_id} k cap exceeded -> 400
        with pytest.raises(HTTPException) as exc:
            api_app.get_similar_vectors(node_id=1, k=15, category=None, request=None, api_key="public-key-123")
        assert exc.value.status_code == 400
        assert "exceeds maximum allowed" in exc.value.detail

        # 7. GET /demo/random
        rand_node = api_app.get_random_node(category=None, request=None, api_key="public-key-123")
        assert 1 <= rand_node.node_id <= 10
        assert "category" in rand_node.metadata

        # GET /demo/random with category filter
        rand_elec = api_app.get_random_node(category="electronics", request=None, api_key="public-key-123")
        assert rand_elec.node_id in (1, 2, 3, 4, 5)
        assert rand_elec.metadata["category"] == "electronics"

        # GET /demo/random with non-existent category -> 404
        with pytest.raises(HTTPException) as exc:
            api_app.get_random_node(category="nonexistent_cat", request=None, api_key="public-key-123")
        assert exc.value.status_code == 404
    finally:
        api_app.index = None
        api_app.VSEARCH_API_KEY = None
        api_app.VSEARCH_READ_KEY = None
        api_app.VSEARCH_PUBLIC_KEY = None
        api_app.VSEARCH_ALLOW_INSECURE = True
        shutil.rmtree(temp_dir, ignore_errors=True)


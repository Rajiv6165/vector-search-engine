import os
import shutil
import tempfile
from fastapi.testclient import TestClient

def run_api_tests():
    temp_dir = tempfile.mkdtemp()
    os.environ["VSEARCH_PERSIST_DIR"] = temp_dir
    import src.vsearch.api.app as api_app
    api_app.PERSIST_DIR = temp_dir
    
    app = api_app.app
    
    try:
        with TestClient(app) as client:
            print("Running test_api_stats_empty...")
            resp = client.get("/stats")
            assert resp.status_code == 200
            assert resp.json()["size"] == 0
            
            print("Running test_api_insert_and_search...")
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
            
            print("Running test_api_filtered_search...")
            resp = client.post("/search", json={
                "vector": [0.1, 0.2, 0.3, 0.5],
                "k": 2,
                "filter": {"category": "clothing"}
            })
            assert resp.status_code == 200
            results = resp.json()["results"]
            assert len(results) == 1
            assert results[0]["node_id"] == 2
            
            print("Running test_api_delete...")
            resp = client.delete("/vectors/1")
            assert resp.status_code == 200
            resp = client.post("/search", json={"vector": [0.1, 0.2, 0.3, 0.5], "k": 2})
            assert len(resp.json()["results"]) == 1
            
            print("Running test_api_dimension_mismatch...")
            resp = client.post("/vectors", json={"node_id": 3, "vector": [0.1, 0.2]})
            assert resp.status_code == 400
            
        # Test shutdown and reload
        print("Running test_lifecycle_reload...")
        with TestClient(app) as client2:
            resp = client2.get("/stats")
            assert resp.status_code == 200
            data = resp.json()
            assert data["size"] == 1
            assert data["dim"] == 4
            
        print("All API tests passed successfully!")
    finally:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    run_api_tests()

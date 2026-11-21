"""
scripts/test_docker_persistence.py
Validates Docker container lifecycle, volume persistence, and native acceleration:
1. Builds Docker image 'vsearch:test'
2. Runs container with a mounted volume
3. Verifies GET /stats returns native: true
4. Inserts a test vector via POST /vectors
5. Restarts the container (simulating crash / redeploy)
6. Verifies that the inserted vector survived and is retrievable
"""

import subprocess
import time
import urllib.request
import json
import sys
import tempfile
import os

def run_cmd(cmd, check=True):
    print(f">> Executing: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    res = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True)
    if check and res.returncode != 0:
        print(f"STDOUT: {res.stdout}")
        print(f"STDERR: {res.stderr}")
        raise RuntimeError(f"Command failed with return code {res.returncode}")
    return res

def wait_for_healthy(url: str, timeout: int = 30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode())
                    if data.get("status") == "ok":
                        return True
        except Exception as e:
            # print for debugging
            # print(f"Health probe error: {e}")
            pass
        time.sleep(1)
    raise TimeoutError(f"Service at {url} did not become healthy within {timeout}s")

def main():
    print("=" * 70)
    print("DOCKER VOLUME PERSISTENCE & NATIVE ENGINE VERIFICATION")
    print("=" * 70)

    # 1. Build image
    print("\n[Step 1] Building Docker image 'vsearch:test'...")
    run_cmd(["docker", "build", "-t", "vsearch:test", "."])
    print("  [PASS] Image 'vsearch:test' built successfully.")

    volume_name = f"vsearch_vol_{int(time.time())}"
    run_cmd(["docker", "volume", "create", volume_name])
    container_name = f"vsearch_persistence_test_{int(time.time())}"
    port = 8009
    base_url = f"http://localhost:{port}"

    try:
        # 2. Run container with mounted volume
        print(f"\n[Step 2] Starting container '{container_name}' with Docker volume '{volume_name}'...")
        run_cmd([
            "docker", "run", "-d",
            "--name", container_name,
            "-p", f"{port}:8000",
            "-v", f"{volume_name}:/app/vsearch_data",
            "-e", "VSEARCH_ALLOW_INSECURE=1",
            "-e", "VSEARCH_AUTO_SEED=0",
            "-e", "VSEARCH_SNAPSHOT_INTERVAL=300",
            "vsearch:test"
        ])

        print("Waiting for container service to be healthy...")
        wait_for_healthy(base_url, timeout=30)
        print("  [PASS] Container is healthy.")

        # 3. Check /stats inside container
        print("\n[Step 3] Checking /stats inside container...")
        with urllib.request.urlopen(f"{base_url}/stats") as resp:
            stats = json.loads(resp.read().decode())
            print(f"  Container /stats: {json.dumps(stats, indent=2)}")
            assert stats.get("native") is True, f"Expected native: true inside container, got {stats.get('native')}"
            initial_size = stats.get("size", 0)
            dim = stats.get("dim", 64)
            print(f"  [PASS] Confirmed 'native: true', version='{stats.get('version')}', size={initial_size}, dim={dim}")

        # 4. Insert test vector
        print("\n[Step 4] Inserting test vector node_id=999999...")
        test_vec = [0.1] * dim
        insert_payload = json.dumps({
            "node_id": 999999,
            "vector": test_vec,
            "metadata": {"test_marker": "survived_restart", "category": "electronics"}
        }).encode("utf-8")
        
        req = urllib.request.Request(
            f"{base_url}/vectors",
            data=insert_payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            res = json.loads(resp.read().decode())
            print(f"  Insert response: {res}")
            assert res.get("status") == "inserted"

        # Verify search returns it before restart
        search_req = urllib.request.Request(
            f"{base_url}/search",
            data=json.dumps({"vector": test_vec, "k": 1, "filter": {"category": "electronics"}}).encode(),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(search_req) as resp:
            s_res = json.loads(resp.read().decode())
            assert len(s_res["results"]) >= 1
            print(f"  Top result: {s_res['results'][0]}")
            print("  [PASS] Search returned matching results before restart.")

        # Allow snapshot / WAL flush
        time.sleep(2)

        # 5. Restart container
        print(f"\n[Step 5] Restarting container '{container_name}'...")
        run_cmd(["docker", "restart", "-t", "2", container_name])
        print("Waiting for container to recover after restart...")
        try:
            wait_for_healthy(base_url, timeout=45)
            print("  [PASS] Container successfully restarted.")
        except Exception as e:
            print(f"  [ERROR] Container failed to become healthy. Dumping logs:")
            subprocess.run(["docker", "logs", "--tail", "50", container_name])
            raise e

        # 6. Verify data survived
        print("\n[Step 6] Verifying vector data survived restart...")
        with urllib.request.urlopen(f"{base_url}/stats") as resp:
            stats_after = json.loads(resp.read().decode())
            print(f"  Post-restart /stats: size={stats_after.get('size')}, native={stats_after.get('native')}")
            assert stats_after.get("size") == initial_size + 1, f"Expected size {initial_size + 1} after restart, got {stats_after.get('size')}"
            assert stats_after.get("native") is True

        # Search for the vector
        with urllib.request.urlopen(search_req) as resp:
            post_restart_results = json.loads(resp.read().decode())
            assert len(post_restart_results["results"]) >= 1
            found_test_node = any(r["node_id"] == 999999 for r in post_restart_results["results"])
            print(f"  Post-restart top result: {post_restart_results['results'][0]}")
            assert found_test_node, "Inserted node 999999 not found in search results after restart"
            print("  [SUCCESS] Data successfully survived container restart across mounted volume!")

    finally:
        print("\nCleaning up test container and volume...")
        run_cmd(["docker", "rm", "-f", container_name], check=False)
        run_cmd(["docker", "volume", "rm", "-f", volume_name], check=False)

    print("\n" + "=" * 70)
    print(">>> DOCKER PERSISTENCE & NATIVE ENGINE VERIFICATION PASSED! <<<")
    print("=" * 70)

if __name__ == "__main__":
    main()

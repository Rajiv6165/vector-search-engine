"""
scripts/measure_throttled_cold_start.py
Measures cold start latency from container start to first successful /search in a CPU-throttled container (docker run --cpus=0.25).
Then runs scripts/smoke_test.py against the live container with both API keys set.
"""

import subprocess
import time
import urllib.request
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("=" * 70)
    print("CPU-THROTTLED COLD START & SMOKE TEST BENCHMARK (--cpus=0.25)")
    print("=" * 70)

    container_name = "vsearch_throttled_cold_start"
    port = 8085
    base_url = f"http://localhost:{port}"
    write_key = "admin-secret-key-12345"
    read_key = "read-secret-key-67890"

    # Ensure any previous container is cleaned up
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

    print(f"\n[1] Starting CPU-throttled container ({container_name}, --cpus=0.25)...")
    cmd = [
        "docker", "run", "-d",
        "--name", container_name,
        "--cpus=0.25",
        "-p", f"{port}:8000",
        "-e", f"VSEARCH_API_KEY={write_key}",
        "-e", f"VSEARCH_READ_KEY={read_key}",
        "vsearch:test"
    ]
    
    t_start = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"docker run failed: {res.stderr}")
    container_id = res.stdout.strip()[:12]
    print(f"    Container launched: {container_id}")

    # Poll for first successful /search response
    print("\n[2] Polling for first successful /search response...")
    t_first_health = None
    t_first_search = None
    first_search_response = None
    query_payload = json.dumps({"vector": [0.1] * 64, "k": 5}).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": read_key
    }

    timeout = 30.0
    while time.perf_counter() - t_start < timeout:
        # Check health
        if t_first_health is None:
            try:
                with urllib.request.urlopen(f"{base_url}/health", timeout=1) as resp:
                    if resp.status == 200:
                        t_first_health = time.perf_counter() - t_start
                        print(f"    First /health HTTP 200: {t_first_health:.3f} s")
            except Exception:
                pass

        # Check search
        try:
            req = urllib.request.Request(f"{base_url}/search", data=query_payload, headers=headers)
            with urllib.request.urlopen(req, timeout=1) as resp:
                if resp.status == 200:
                    t_first_search = time.perf_counter() - t_start
                    first_search_response = json.loads(resp.read().decode())
                    print(f"    First /search HTTP 200: {t_first_search:.3f} s")
                    break
        except Exception:
            pass

        time.sleep(0.05)

    if t_first_search is None:
        subprocess.run(["docker", "logs", container_name])
        subprocess.run(["docker", "rm", "-f", container_name])
        raise TimeoutError(f"Container failed to respond to /search within {timeout}s")

    print("\n" + "=" * 70)
    print("COLD START RESULTS (--cpus=0.25 throttled):")
    print(f"  Container boot to /health ok:  {t_first_health:.3f} s")
    print(f"  Container boot to first search:{t_first_search:.3f} s")
    print(f"  First search results returned: {len(first_search_response.get('results', []))} vectors")
    print("=" * 70)

    # 3. Run full smoke test against the container
    print("\n[3] Running scripts/smoke_test.py against the live throttled container...")
    from scripts.smoke_test import run_smoke_test
    smoke_passed = run_smoke_test(base_url=base_url, write_key=write_key, read_key=read_key)

    # Cleanup
    print("\nCleaning up throttled test container...")
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

    if not smoke_passed:
        raise RuntimeError("Smoke test suite failed!")
        
    print("\n>>> ALL THROTTLED COLD START AND SMOKE TEST CHECKS PASSED! <<<")

if __name__ == "__main__":
    main()

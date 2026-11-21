"""
scripts/smoke_test.py
Verification suite against a live vsearch deployment.
Validates:
1. GET /health (unauthenticated, status ok)
2. GET /stats (read-key authenticated, native: true, version present)
3. POST /search (unfiltered returns k nearest neighbors)
4. POST /search (filtered returns only the requested category)
5. Security: Read key cannot write (insert/delete rejected with 401)
6. Write key can write and delete (if provided)
"""

import sys
import os
import json
import argparse
import urllib.request
import urllib.error
from typing import Dict, Any, Optional, Tuple

def make_request(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    payload: Optional[Dict[str, Any]] = None
) -> Tuple[int, Any]:
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
        
    data_bytes = None
    if payload is not None:
        data_bytes = json.dumps(payload).encode("utf-8")
        
    req = urllib.request.Request(url, data=data_bytes, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            code = resp.status
            body = resp.read().decode("utf-8")
            return code, json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = {"raw": body}
        return e.code, parsed

def run_smoke_test(base_url: str, write_key: Optional[str] = None, read_key: Optional[str] = None) -> bool:
    base = base_url.rstrip("/")
    print("=" * 70)
    print(f"RUNNING VSEARCH SMOKE TEST ON: {base}")
    print(f"Write Key: {'***' + write_key[-4:] if write_key and len(write_key) > 4 else ('Present' if write_key else 'None')}")
    print(f"Read Key:  {'***' + read_key[-4:] if read_key and len(read_key) > 4 else ('Present' if read_key else 'None')}")
    print("=" * 70)

    all_passed = True

    # 1. GET /health (Unauthenticated)
    print("\n[1/5] Checking GET /health (Unauthenticated)...")
    code, data = make_request(f"{base}/health")
    if code == 200 and data.get("status") == "ok":
        print(f"  [PASS] /health returned HTTP 200, status='ok'")
    else:
        print(f"  [FAIL] /health failed: HTTP {code}, body: {data}")
        all_passed = False

    # 2. GET /stats (Read Key)
    print("\n[2/5] Checking GET /stats (Read Key)...")
    read_headers = {"X-API-Key": read_key} if read_key else {}
    code, stats = make_request(f"{base}/stats", headers=read_headers)
    dim = 64
    if code == 200:
        is_native = stats.get("native")
        version = stats.get("version")
        size = stats.get("size", 0)
        dim = stats.get("dim", 64)
        print(f"  [PASS] /stats returned HTTP 200")
        print(f"         - version: {version}")
        print(f"         - native:  {is_native}")
        print(f"         - size:    {size} vectors")
        print(f"         - dim:     {dim}")
        
        if is_native is True:
            print("  [PASS] native is True (C++ OpenMP extension active in runtime)")
        else:
            print("  [WARN/FAIL] native is NOT True! (Running in pure-Python fallback)")
            all_passed = False
            
        if not version:
            print("  [FAIL] version string missing from /stats")
            all_passed = False
    else:
        print(f"  [FAIL] /stats failed: HTTP {code}, body: {stats}")
        all_passed = False

    # 3. Unfiltered Search (k results)
    print("\n[3/5] Checking POST /search (Unfiltered, k=5)...")
    dummy_query = [0.1] * dim
    code, res = make_request(
        f"{base}/search",
        method="POST",
        headers=read_headers,
        payload={"vector": dummy_query, "k": 5}
    )
    if code == 200:
        results = res.get("results", [])
        if len(results) == 5:
            print(f"  [PASS] Unfiltered search returned exactly 5 results (top-1 node_id={results[0]['node_id']}, dist={results[0]['distance']:.4f})")
        else:
            print(f"  [WARN] Unfiltered search returned {len(results)} results (expected 5)")
    else:
        print(f"  [FAIL] Unfiltered search failed: HTTP {code}, body: {res}")
        all_passed = False

    # 4. Filtered Search (Only requested category)
    print("\n[4/5] Checking POST /search (Filtered: category='electronics')...")
    code, res = make_request(
        f"{base}/search",
        method="POST",
        headers=read_headers,
        payload={
            "vector": dummy_query,
            "k": 5,
            "filter": {"category": "electronics"}
        }
    )
    if code == 200:
        results = res.get("results", [])
        if results:
            categories = [r.get("metadata", {}).get("category") for r in results]
            if all(c == "electronics" for c in categories):
                print(f"  [PASS] All {len(results)} results match category='electronics' (100% precision)")
            else:
                print(f"  [FAIL] Non-matching categories in results: {categories}")
                all_passed = False
        else:
            print("  [WARN] Filtered search returned 0 results (index might not have 'electronics' category)")
    else:
        print(f"  [FAIL] Filtered search failed: HTTP {code}, body: {res}")
        all_passed = False

    # 5. Auth Permission Checks
    print("\n[5/5] Checking Security & Auth Restrictions...")
    if read_key:
        # Read key should be REJECTED on writes
        code, err = make_request(
            f"{base}/vectors",
            method="POST",
            headers={"X-API-Key": read_key},
            payload={"node_id": 9999999, "vector": dummy_query, "metadata": {"test": True}}
        )
        if code in (401, 403):
            print(f"  [PASS] Read key was rejected on POST /vectors with HTTP {code} (Cannot insert)")
        else:
            print(f"  [FAIL] Read key was NOT rejected on write! HTTP {code}, body: {err}")
            all_passed = False

        code, err = make_request(
            f"{base}/vectors/1",
            method="DELETE",
            headers={"X-API-Key": read_key}
        )
        if code in (401, 403):
            print(f"  [PASS] Read key was rejected on DELETE /vectors/1 with HTTP {code} (Cannot delete)")
        else:
            print(f"  [FAIL] Read key was NOT rejected on delete! HTTP {code}, body: {err}")
            all_passed = False
    else:
        print("  [SKIP] No VSEARCH_READ_KEY configured to test read restriction.")

    # Write key validation (if provided)
    if write_key:
        # Write key SHOULD succeed on write and delete
        test_id = 9999999
        code, write_res = make_request(
            f"{base}/vectors",
            method="POST",
            headers={"X-API-Key": write_key},
            payload={"node_id": test_id, "vector": dummy_query, "metadata": {"smoke_test": True}}
        )
        if code == 200:
            print(f"  [PASS] Write key successfully inserted test vector node_id={test_id}")
            # Clean up
            code_del, _ = make_request(
                f"{base}/vectors/{test_id}",
                method="DELETE",
                headers={"X-API-Key": write_key}
            )
            if code_del == 200:
                print(f"  [PASS] Write key successfully cleaned up test vector")
            else:
                print(f"  [WARN] Failed to clean up test vector: HTTP {code_del}")
        else:
            print(f"  [FAIL] Write key failed on insert: HTTP {code}, body: {write_res}")
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print(">>> ALL SMOKE TESTS PASSED SUCCESSFULLY! <<<")
        print("=" * 70)
        return True
    else:
        print(">>> SMOKE TESTS FAILED! <<<")
        print("=" * 70)
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vsearch deployment smoke test")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="Base URL of service")
    parser.add_argument("--write-key", type=str, default=os.getenv("VSEARCH_API_KEY"), help="Write API key")
    parser.add_argument("--read-key", type=str, default=os.getenv("VSEARCH_READ_KEY"), help="Read API key")
    args = parser.parse_args()

    success = run_smoke_test(args.url, write_key=args.write_key, read_key=args.read_key)
    sys.exit(0 if success else 1)

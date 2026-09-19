# Phase 8: Hardening & Honesty Pass

Phase 8 was entirely dedicated to closing the gap between the project's documented claims and its actual implementation. Rather than introducing new features, this phase focused on hardening the codebase, improving its robustness, and ensuring absolute transparency in what the engine does (and doesn't) do.

## Overview of Changes

### 1. Document Honesty
* **Removed gRPC Claims:** The initial design planned for both a gRPC and REST API. However, gRPC was deferred in Phase 4. All claims of gRPC support have been removed from the `README.md` to accurately reflect that the service currently operates exclusively over REST (via FastAPI).
* **Live Deployment Clarification:** The documentation previously implied the existence of a live deployment URL. Since this is an open-source template project, those claims were removed. The engine remains fully Dockerized and *ready* for deployment, but we no longer claim it is hosted live out-of-the-box.

### 2. Dependency Pinning & Local Testing
* **Reproducible Builds:** The `requirements.txt` file was completely rebuilt in a clean virtual environment and all dependencies were explicitly pinned using `pip freeze`. This resolves historical conflicts (e.g., `web3`/`pkg_resources` issues) and guarantees a reproducible environment.
* **Local Test Suite:** The full `pytest` suite—including the `hypothesis` and concurrency tests—was verified to pass successfully in the clean local environment. 

### 3. Continuous Integration (CI)
* **GitHub Actions:** Added a `.github/workflows/test.yml` workflow to automatically run the full pytest suite on every push and pull request to the `main` branch.
* **CI Badge:** Added a live "tests passing" status badge to the top of the `README.md`.

### 4. API Authentication
* **Basic Security:** Added a lightweight API key authentication mechanism to the FastAPI service. 
* **Implementation:** The `POST /vectors`, `DELETE /vectors/{node_id}`, and `POST /search` endpoints now require an `X-API-Key` header if the `VSEARCH_API_KEY` environment variable is set. If the variable is unset, the API remains open (ideal for local development). This prevents the service from being fully exposed if deployed to a public URL.

### 5. WAL Performance: Fsync Batching
* **The Problem:** Previously, the Write-Ahead Log (WAL) called `os.fsync()` on every single insertion or deletion. While maximizing durability, this resulted in nearly 98.5% overhead on insert operations due to continuous disk I/O blocking.
* **The Solution:** Implemented a batched group-commit approach. The `HNSWConfig` now accepts a `wal_batch_size` parameter (default: 100). The WAL will buffer writes and only call `fsync` after `N` operations.
* **The Tradeoff:** This significantly increases insert throughput at the cost of a small durability window. If the server crashes, up to the last `N` un-fsynced operations may be lost. This is a standard database tradeoff (similar to PostgreSQL's `commit_delay` or Redis's AOF `everysec`).

### 6. Tombstone Compaction
* **The Problem:** Deletions in the engine were previously purely logical (tombstones). The node was marked as deleted and skipped during search, but its vector remained in the memory-mapped file, and its ID remained in the HNSW graph's adjacency lists. Over time, this wasted disk space and degraded graph connectivity.
* **The Solution:** Added a `compact()` method to the `HNSWIndex`. When called, it:
  1. Acquires a global write lock.
  2. Physically rewrites the memory-mapped `VectorStore`, dropping all vectors marked as deleted, shrinking the file size.
  3. Iterates through the entire HNSW graph, removing deleted nodes from all adjacency lists, healing the graph.
  4. Truncates the WAL, as all current state is now durably compacted.
* **Verification:** Added `tests/test_compaction.py` to guarantee that file sizes shrink, deleted nodes are cleared, and recall remains intact for the remaining vectors.

### 7. CI Packaging and Import Paths
* **The Problem:** The GitHub Actions CI workflow failed because `vsearch` was not a properly installable package. Tests and benchmarks were relying on local `sys.path.insert()` hacks or implicitly resolving `src.vsearch`, which failed on a clean runner.
* **The Solution:** Added a `pyproject.toml` using `setuptools` to make `vsearch` installable. Replaced all `from src.vsearch...` with `from vsearch...` and removed `sys.path.insert()` hacks. Updated `.github/workflows/test.yml` to run `pip install -e .` before testing.
* **The Lesson:** Relying on `PYTHONPATH` manipulation or relative paths for imports breaks continuous integration and makes a project difficult for others to consume. Defining a project as an installable package from day one avoids "works on my machine" failures.

### 8. Dependency Management: The `pip freeze` Pitfall
* **The Problem:** The initial `requirements.txt` was generated using a raw `pip freeze`. This pinned every transitive dependency (e.g., `contourpy`, `fonttools`) to an exact version specific to the developer's local Python version. When the GitHub Actions CI (running Python 3.10) tried to install these, it failed because some of those exact transitive versions required Python >=3.12.
* **The Solution:** Rewrote `requirements.txt` to only include top-level, directly imported dependencies (e.g., `numpy`, `fastapi`, `pytest`) using compatible-range pins (e.g., `>=1.26,<3.0`). 
* **The Lesson:** Never use a raw `pip freeze > requirements.txt` for library or template project distribution. It tightly couples the environment to the local machine and breaks cross-platform/cross-version compatibility. Only pin direct dependencies with compatible ranges, allowing `pip` to resolve valid transitive dependencies for the target environment.

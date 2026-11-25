# vsearch Project & Architecture Facts

## Overview
- **What is vsearch**: `vsearch` is an open-source, single-process, in-memory vector search engine implementing the Hierarchical Navigable Small World (HNSW) algorithm.
- **Core Purpose**: Provide fast, low-latency approximate nearest neighbor (ANN) search and filtered vector search with persistence and minimal memory overhead.
- **Implementation**: Built with Python for the REST API (FastAPI/Uvicorn) and C++ (with AVX2/FMA SIMD optimizations and OpenMP multi-threading) for compute-intensive distance calculation and graph search.
- **Automated Test Suite**: Verified by 33 comprehensive automated tests covering graph indexing, persistence, filtering, cache budgeting, API authorization, and assistant interactions.

## Vector Search & HNSW Basics
- **What is a Vector**: A list of numerical coordinates (embeddings) representing data points such as text, images, or audio in high-dimensional space.
- **Distance Metric**: `vsearch` measures similarity primarily via Euclidean (L2) distance and cosine similarity, where smaller distance values mean vectors are more similar.
- **What is HNSW (Hierarchical Navigable Small World)**: A multi-layer graph data structure where upper layers act as express lanes (skipping long distances) and lower layers provide fine-grained nearest neighbor connections. Search starts at the top layer, greedily traverses to the closest node, descends to the next layer, and finishes with a beam search on layer 0.
- **Why Items Are Similar**: Two items are similar if their vector embeddings lie close to each other in vector space (low distance), reflecting shared features or semantic proximity.

## Filtering Capabilities & Proven Benchmarks
- **Filtered Search**: Allows filtering vectors by metadata attributes (e.g., category).
- **Dual Execution Path & 2,000-Match Fallback**:
  - `graph`: The standard HNSW in-graph search path with visited-node bitset filtering.
  - `exact_fallback`: When the matching subset has 2,000 or fewer items (crossover threshold where brute-force is faster than graph traversal) or during escalation, `vsearch` executes exact brute-force search over the matching vectors.
- **Filter Caching**: Employs an LRU cache with byte budgeting for roaring bitmaps representing filtered node sets.
- **Proven Filtered Recall**: In benchmarks on 100,000 128-dimensional SIFT vectors, `vsearch` achieves 96.1% to 100.0% filtered recall across all tested selectivity levels.

## The Public Demo
- **Dataset**: 20,000 synthetic 64-dimensional vectors grouped into 5 clustered categories (`electronics`, `clothing`, `home`, `books`, `sports`).
- **Synthetic & Resets on Restart**: All demo vectors are synthetic; any runtime additions reset to the clean baked snapshot whenever the service restarts.
- **Interactive UI**:
  - "Pick random item": Samples a random query vector from the dataset (`GET /demo/random`).
  - Nearest Neighbors: Searches for the top-$k$ closest vectors (`GET /similar/{node_id}`).
  - Category Filter: Restricts queries to specific categories.
  - Telemetry: Displays real-time query latency in milliseconds and the search path used (`graph` vs `exact_fallback`).

## Architecture & Operational Constraints
- **Single-Process Safety**: `vsearch` must run as a single process (`--workers 1`). Multi-worker deployments (e.g. `WEB_CONCURRENCY > 1`) are explicitly blocked at startup to prevent memory corruption of the in-memory graph and memory-mapped write-ahead log (WAL).
- **Persistence**: Employs a binary Write-Ahead Log (`wal.log`) for durability and atomic snapshot checkpoints (`snapshot.bin`) for instant restarts.
- **Memory Footprint**: Uses ~105 MB resident memory for 20,000 vectors; easily fits inside a 512 MB container.
- **Security & Three-Tier Authentication**:
  - `VSEARCH_API_KEY` (Write Key): Secret key required for mutations (`POST/DELETE /vectors`).
  - `VSEARCH_READ_KEY` (Read Key): Secret key for raw vector search (`POST /search`) and statistics (`GET /stats`).
  - `VSEARCH_PUBLIC_KEY` (Public Demo Key): Dedicated public key injected into the frontend demo, authorized only on `/similar`, `/demo/random`, and `/assistant`. Strictly rejected with HTTP 401 on `/search`, `/stats`, and write routes.

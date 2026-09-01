# Phase 4: API Layer & Metadata Storage

Phase 4 introduces a fully functional REST API over the core `HNSWIndex`, effectively turning it into a standalone vector search service. This phase also introduces hybrid search capabilities by allowing metadata to be stored and filtered alongside vectors.

## Overview of Changes

1. **FastAPI Service (`src/vsearch/api/`)**: A robust REST API using FastAPI.
2. **Metadata Storage**: Extended the core `HNSWIndex` and custom binary WAL (`WALManager`) to persist arbitrary JSON-serializable dictionaries alongside vectors.
3. **Hybrid Search (Post-Filtering)**: Added the ability to apply exact-match filters on metadata during search queries.
4. **Lifecycle Management**: The service automatically loads the latest snapshot on startup, saves a final snapshot on shutdown, and performs background snapshots every 5 minutes to prevent unbounded WAL growth.

## REST Endpoints

### 1. `POST /vectors`
Insert a new vector with an optional metadata dictionary.

**Request:**
```json
{
  "node_id": 1,
  "vector": [0.1, 0.2, 0.3, 0.4],
  "metadata": {
    "category": "electronics",
    "price": 100
  }
}
```

**Response:**
```json
{
  "node_id": 1,
  "status": "inserted"
}
```

### 2. `DELETE /vectors/{node_id}`
Delete a vector by its node ID.

**Response:**
```json
{
  "status": "deleted",
  "node_id": 1
}
```

### 3. `POST /search`
Search for the nearest neighbors to a query vector, with an optional exact-match metadata filter.

**Request:**
```json
{
  "vector": [0.1, 0.2, 0.3, 0.4],
  "k": 10,
  "ef": 50,
  "filter": {
    "category": "electronics"
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "node_id": 1,
      "distance": 0.0,
      "metadata": {
        "category": "electronics",
        "price": 100
      }
    }
  ]
}
```

### 4. `GET /stats`
Retrieve current index metrics and configuration.

**Response:**
```json
{
  "size": 1000,
  "dim": 4,
  "max_level": 3,
  "entry_point": 1,
  "config": {
    "M": 16,
    "ef_construction": 100,
    "ef_search": 50
  }
}
```

## Architectural Decisions

### Metadata WAL Serialization
To maintain the high throughput of the append-only WAL without breaking the existing binary format, we introduced a new WAL Opcode (`0x03` = `INSERT_WITH_METADATA`). This allows vectors that carry metadata to be logged in a single contiguous write:
`[Opcode (1 byte) | Node ID (4 bytes) | Vector Bytes | Metadata JSON String Length (4 bytes) | Metadata JSON String]`

### Hybrid Search: Pre-Filtering vs. Post-Filtering
When implementing hybrid search (combining vector similarity with scalar filtering), there are two primary approaches:

1. **Pre-Filtering:** 
   - *How it works:* The graph traversal logic is modified to only consider nodes that match the filter. 
   - *Pros:* Guarantees `k` results (if they exist) and is highly accurate.
   - *Cons:* Extremely complex to implement in HNSW. If a filter is highly restrictive, the search algorithm can hit "dead ends" in the graph because the neighbors of a node might not match the filter, breaking the traversal path. It often requires maintaining multiple graphs or applying complex heuristics to jump across non-matching nodes.
2. **Post-Filtering:**
   - *How it works:* The search algorithm runs normally to retrieve a large pool of candidate neighbors. The filter is then applied to these candidates, discarding non-matches until `k` results are found.
   - *Pros:* Simple to implement and doesn't interfere with the heavily optimized HNSW traversal logic.
   - *Cons:* Can suffer from "candidate starvation." If a filter is highly restrictive, the candidate pool might not contain `k` matching results, even if they exist elsewhere in the index.

**Decision:** For Phase 4, we implemented **Post-Filtering**. To mitigate candidate starvation, the API automatically inflates the internal `ef` (exploration factor) when a `filter` is provided (e.g., `ef = max(ef, k * 5)`). This forces the HNSW index to return a much wider candidate pool before filtering, significantly improving the chances of finding `k` matches without modifying the core traversal invariants. Pre-filtering is left as an advanced optimization for a future phase.

## Running the API

You can start the FastAPI server locally:

```bash
uvicorn src.vsearch.api.app:app --host 0.0.0.0 --port 8000 --reload
```

The service will automatically create a `vsearch_data/` directory in your current working directory to store the WAL, memmap array, and snapshots. Interactive API documentation is available at `http://localhost:8000/docs`.

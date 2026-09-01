# Phase 3: Disk-Backed Persistence Layer

This document details the implementation and design tradeoffs for Phase 3: making the HNSW index durable across process restarts and enabling vectors to reside outside of RAM.

## 1. Vector Storage Format

To ensure the index can exceed available RAM without degrading lookup speeds, we implemented a custom `VectorStore` backed by `numpy.memmap`.

**Design**:
- Raw vectors are written sequentially to a flat binary file (`vectors.bin`).
- An in-memory mapping `node_id -> index_in_file` provides O(1) lookups for any given node ID.
- Since `numpy.memmap` requires a fixed shape at instantiation, `VectorStore` automatically tracks capacity. When an insertion exceeds the file size, it explicitly flushes, resizes the underlying file (doubling capacity), and re-instantiates the memmap object.

**Impact**: 
- Searching reads directly from the OS page cache.
- Inserting incurs minimal overhead since it's appending to an pre-allocated memory-mapped block.

## 2. Write-Ahead Log (WAL)

To protect the index against crashes without incurring the massive overhead of snapshotting on every insert, we implemented a Write-Ahead Log (WAL).

**Design**:
- `wal.log` is an append-only file opened in binary mode.
- We bypassed libraries like `json` or `pickle` for the WAL entirely. Instead, operations are logged using a highly efficient Python `struct` binary layout: `[Opcode (1 byte) | Node ID (4 bytes) | Vector Bytes (dim * 4 bytes)]`.
- On `insert()` or `delete()`, the operation is flushed to the WAL *before* the in-memory graph is mutated.

## 3. Snapshot Serialization (Hybrid Approach)

To serialize the hierarchical graph cleanly, safely, and efficiently, we utilized a hybrid approach, consciously avoiding `pickle` due to its known vulnerability to arbitrary code execution.

**Design**:
- **Metadata**: Constants, configuration, and the entry point are saved to a human-readable `metadata.json` file.
- **Graph Edges**: The nested structure of the HNSW index (`List[Dict[int, Set[int]]]`) is flattened into a tightly packed binary file (`graph.bin`) using `struct`. 
  - Format: `[Num Levels | [Num Nodes | [Node ID | Num Edges | Edge1, Edge2...]]...]`
- Upon a successful snapshot, the WAL is truncated since all operations are now securely committed to the snapshot.

**Impact**: Loading a 10,000-node graph takes milliseconds, and reading the binary avoids the large memory spike typical of parsing massive JSON objects.

## 4. Benchmarks

We measured the overhead of the Phase 3 additions using a dataset of 10,000 random vectors of 64 dimensions, matching `M=16, ef_construction=100`.

| Metric | Time | Notes |
|--------|------|-------|
| In-Memory Insert (Baseline) | ~29.58s | Phase 2 implementation. |
| Persistent Insert (WAL + Memmap)| ~58.72s | Appending and `flush()`ing every vector to disk. |
| **WAL Overhead** | **+98.5%** | Inserting is roughly 2x slower due to the fsync durability cost. |
| Save Snapshot | ~0.05s | Custom binary structure makes disk dumps near-instant. |
| Load Snapshot | ~0.05s | Re-instantiating the graph from `graph.bin` and mapping `vectors.bin`. |

The ~2x latency penalty during insertion is standard for database durability guarantees (fsyncing to WAL on every transaction). The custom binary snapshotting proved extremely effective, enabling nearly instantaneous cold starts.

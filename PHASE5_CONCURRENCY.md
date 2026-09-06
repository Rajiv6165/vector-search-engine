# Phase 5: Concurrency & Consistency

This document outlines the changes made to the Vector Search Engine to support concurrent operations safely without regressing recall or persistence guarantees.

## 1. Identified Race Conditions

In the single-threaded implementation of `HNSWIndex`, several race conditions arise under concurrent access:

- **Race Condition 1 (Incomplete Edges / Iteration Crash):** 
  `insert()` mutates the graph layers (`self.graphs[lc]`) by adding new nodes and updating neighbor edge lists. A concurrent `search()` reading these dictionaries could encounter missing keys or trigger `RuntimeError: dictionary changed size during iteration` if Python resizes the dictionary or set mid-iteration.
- **Race Condition 2 (Dangling References from Delete):**
  Phase 1's `delete()` reconnected neighbors immediately, physically removing the `node_id` from `self.nodes` and `self.graphs`. A concurrent `search()` traversing via that deleted node could hit a `KeyError` on `self.nodes[node_id]`. 
- **Race Condition 3 (Lost Updates in Insert):**
  When two `insert()` calls execute concurrently on adjacent nodes, they may both attempt to update the same neighbor's edge list. Although CPython's GIL protects simple dictionary assignments, read-modify-write patterns like neighbor pruning (`self.graphs[lc][n_id] = pruned_set`) are not atomic, leading to lost edges.
- **Race Condition 4 (Entry Point and Max Level Inconsistency):**
  Concurrent inserts can both determine they should increase `self.max_level` and update `self.entry_point`. Without synchronization, the index could be left with a mismatched entry point or max level, making parts of the graph unreachable.

## 2. Locking Strategy: Reader-Writer Lock

To handle concurrency, we implemented a custom thread-safe **Reader-Writer Lock (RWLock)**.
- **Multiple Readers:** `search()` calls acquire the lock in *read mode*, allowing many simultaneous searches to execute in parallel without blocking each other. This is critical for high read QPS.
- **Exclusive Writers:** `insert()` and `delete()` calls acquire the lock in *write mode*. This ensures mutual exclusion: no other readers or writers can run concurrently, which protects graph mutations, WAL appending, and dictionary resizing.
- **Atomicity:** The write lock spans both the memory update and the WAL log append, ensuring that a crash won't leave the in-memory graph and the persistence layer out of sync.

### Why not fine-grained locking?
Fine-grained (per-node) locking is extremely complex for HNSW due to the highly interconnected nature of the graph and the risk of deadlocks during neighbor pruning. A global RW lock provides a substantial concurrency improvement for read-heavy workloads (the most common vector database use case) with minimal code complexity overhead.

## 3. Tombstone Deletion Semantics

We revised the `delete()` semantics from Phase 1. Doing immediate neighbor reconnection holding an exclusive write lock would cause unacceptable latency spikes for all concurrent readers and writers, as reconnection is $O(degree^2)$.

- **New Approach (Tombstones):** `delete()` simply marks the `node_id` as deleted by adding it to a `self.deleted_nodes` set, logs a `DELETE` operation to the WAL, and removes it from `metadata_store`.
- **Search Interaction:** `search()` can still safely traverse through a deleted node (since its vector and edges remain in `self.nodes` and `self.graphs`), but it filters out deleted nodes before returning the final top-$k$ results.
- **Compaction:** In a future phase, a background compaction process will physically remove tombstoned nodes and reconnect their edges during low-traffic periods.

## 4. Benchmark Results

Concurrency benchmarks were executed to evaluate the impact of the `RWLock` and Python's Global Interpreter Lock (GIL) on search and insertion throughput.

### Read Throughput (Search QPS)
- **1 Thread:** 156.75 QPS
- **4 Threads:** 51.98 QPS
- **16 Threads:** 45.49 QPS

*Observation:* As expected in Python, purely CPU-bound tasks (like iterating over lists and computing vector distances) experience **negative scaling** with multithreading due to the GIL. Context switching overhead causes 4 threads to be significantly slower than a single thread. For production use, multiprocessing (running multiple worker processes) is highly recommended for scaling read throughput in Python.

### Write Throughput Degradation
- **Baseline (1 Writer, 0 Readers):** 73.20 inserts/sec
- **Concurrent (1 Writer, 4 Readers):** 28.99 inserts/sec

*Observation:* Write throughput degrades when concurrent readers are active. This is primarily because `insert()` requires a Write Lock, and our `RWLock` design causes the writer to wait for active readers to finish. We added `pending_writers` logic to prevent complete writer starvation, but writers must still yield to in-progress reads.

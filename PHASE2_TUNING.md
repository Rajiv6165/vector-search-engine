# Phase 2 Tuning and Tradeoffs

This document summarizes the parameter tuning experiments for the HNSW index and explains the tradeoffs between recall, latency, and memory footprint.

## Parameters Explained

1. **`M` (Maximum edges per node)**
   - **What it controls**: The maximum number of outgoing edges a node can have in the graph (except for layer 0, where it is `2*M`).
   - **Tradeoff**: Increasing `M` makes the graph denser. This significantly improves recall by providing more routing options and reducing the chance of getting stuck in local minima during search. However, a higher `M` increases memory footprint linearly (more edges to store) and increases search latency (more neighbors to evaluate at each step).
   - **Recommendation**: `M=16` is a standard starting point. Use `M=32` or higher only if data is highly clustered or dimensionality is very high.

2. **`ef_construction` (Size of the dynamic candidate list during insert)**
   - **What it controls**: How deeply the graph explores candidate neighbors when inserting a new node.
   - **Tradeoff**: Higher `ef_construction` builds a higher-quality graph by finding better edges, which improves recall later on. However, it increases index build time (insertion latency) linearly. It has virtually no impact on search latency or memory footprint.
   - **Recommendation**: Set this as high as your indexing time budget allows (e.g., 100-200). 

3. **`ef_search` (Size of the dynamic candidate list during search)**
   - **What it controls**: How thoroughly the algorithm explores the graph during a search query.
   - **Tradeoff**: Higher `ef_search` improves recall directly by exploring more paths. It increases search latency linearly. It does not affect memory footprint or build time.
   - **Recommendation**: This is the primary knob to tune at query time. It should be at least equal to `k` (the number of neighbors requested).

## Parameter Sweep Results

We benchmarked 2000 vectors of 64 dimensions, searching for `k=10` neighbors against an exact brute-force baseline.

| M | ef_con | ef_sch | Recall | Insert(s) | Search(ms) | Edges |
|---|--------|--------|--------|-----------|------------|-------|
|  8|    100 |     50 | 0.8455 |     4.460 |      1.224 | 24665 |
|  8|    100 |    100 | 0.9365 |     4.460 |      2.036 | 24665 |
| 16|    100 |     50 | 0.9615 |     6.564 |      1.518 | 47519 |
| 16|    100 |    100 | 0.9855 |     6.564 |      2.403 | 47519 |
| 16|    200 |     50 | 0.9630 |     8.244 |      1.565 | 47559 |
| 32|    100 |     50 | 0.9920 |    12.256 |      1.869 | 95101 |
| 32|    100 |    100 | 0.9990 |    12.256 |      2.911 | 95101 |

## Pareto-Optimal Configuration & Default Choice

The Pareto front reveals that `M=16`, `ef_construction=100`, and `ef_search=50` is a sweet spot. 
- It achieves **96.15% recall**, well above the 90% threshold for good ANN performance.
- Search latency is extremely low (1.5ms per query).
- Memory footprint is reasonable (~47k edges for 2k nodes, roughly 24 edges per node on average across all layers).

Pushing `M=32` doubles the memory footprint and slows insertion considerably for only a 3% absolute gain in recall. Pushing `ef_search=100` adds 1ms latency per query for a ~2.4% recall gain, which could be toggled dynamically by the user but isn't necessary for the default.

**Chosen Default**:
```python
HNSWConfig(M=16, ef_construction=100, ef_search=50)
```

## Parallel Construction Constraints

During this phase, we investigated parallelizing the bulk insert process. 
- True parallel insertion (using Python threads or multiprocessing) into a shared HNSW graph is generally **unsafe** without complex fine-grained locking on individual nodes. Because `insert()` mutates adjacency lists of existing nodes (bidirectional connections and edge pruning), concurrent inserts can easily corrupt the graph or cause race conditions.
- Vectorizing the distance computation (batching `numpy.linalg.norm` over the candidate list) yielded a nearly **2x speedup** on insertion and a **3x speedup** on search, providing the necessary performance gains without the concurrency complexity.

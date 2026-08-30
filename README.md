# Vector Search Engine

A from-scratch implementation of a vector database and Approximate Nearest Neighbor (ANN) search engine. 

## Pitch
This project exists to demonstrate a deep, systems-level understanding of vector search algorithms. Instead of wrapping existing libraries like FAISS, hnswlib, or Annoy, the core Hierarchical Navigable Small World (HNSW) graph algorithm is implemented entirely from scratch in Python. It strictly follows the algorithms presented in Malkov & Yashunin (2016), complete with a brute-force baseline for rigorous recall validation.

## Phase 1 Features
- **In-Memory HNSW Index**: Core graph structure with probabilistic layer assignment.
- **Search & Insert**: Greedy best-first layer traversal with bounded candidate heuristics (`ef_construction` and `ef_search`).
- **Graph Healing Deletions**: Deletions maintain connectivity by mutually reconnecting former neighbors.
- **Distance Metrics**: Pluggable L2 and Cosine distance functions.

*(Note: Persistence, concurrency, API layer, and SIMD/C++ performance tuning are planned for future phases.)*

## Testing
To run the algorithmic correctness and recall tests locally, you need `pytest` and `numpy`:

```bash
pip install -r requirements.txt
python -m pytest tests/test_hnsw_recall.py -v
```

import os
import asyncio
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import ValidationError
from typing import Optional

from src.vsearch.hnsw import HNSWIndex, HNSWConfig, VectorStore
from src.vsearch.api.models import InsertRequest, InsertResponse, SearchRequest, SearchResponse, SearchResult, StatsResponse

PERSIST_DIR = os.getenv("VSEARCH_PERSIST_DIR", "./vsearch_data")
SNAPSHOT_INTERVAL_SECONDS = int(os.getenv("VSEARCH_SNAPSHOT_INTERVAL", "300"))

# Global index instance
index: Optional[HNSWIndex] = None

async def snapshot_loop():
    """Background task to periodically snapshot the index."""
    while True:
        await asyncio.sleep(SNAPSHOT_INTERVAL_SECONDS)
        if index is not None:
            print("Running periodic background snapshot...")
            index.save_snapshot()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global index
    print(f"Starting vsearch service. Persist dir: {PERSIST_DIR}")
    
    # Initialize or load index
    metadata_path = os.path.join(PERSIST_DIR, "metadata.json")
    if os.path.exists(metadata_path):
        print("Found existing snapshot, loading...")
        index = HNSWIndex.load_snapshot(path=PERSIST_DIR, persist_dir=PERSIST_DIR)
        print("Snapshot and WAL loaded.")
    else:
        print("No existing snapshot found, initializing empty index.")
        config = HNSWConfig(M=16, ef_construction=100, ef_search=50)
        index = HNSWIndex(config=config, persist_dir=PERSIST_DIR)
        
    # Start background snapshot task
    task = asyncio.create_task(snapshot_loop())
    
    yield  # Run application
    
    print("Shutting down vsearch service...")
    task.cancel()
    if index is not None:
        print("Saving final snapshot...")
        index.save_snapshot()
        if hasattr(index.nodes, "close"):
            index.nodes.close()
        if index.wal:
            index.wal.close()
        print("Shutdown complete.")

app = FastAPI(title="Vector Search Engine", lifespan=lifespan)

@app.post("/vectors", response_model=InsertResponse)
async def insert_vector(req: InsertRequest):
    if index is None:
        raise HTTPException(status_code=500, detail="Index not initialized")
        
    vec_array = np.array(req.vector, dtype=np.float32)
    
    # Dimension check
    current_dim = 0
    if isinstance(index.nodes, VectorStore):
        current_dim = index.nodes.dim
    elif len(index.nodes) > 0:
        current_dim = len(next(iter(index.nodes.values())))
        
    if current_dim > 0 and len(vec_array) != current_dim:
        raise HTTPException(status_code=400, detail=f"Vector dimension mismatch. Expected {current_dim}, got {len(vec_array)}")
        
    try:
        index.insert(req.node_id, vec_array, metadata=req.metadata)
        return InsertResponse(node_id=req.node_id, status="inserted")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/vectors/{node_id}")
async def delete_vector(node_id: int):
    if index is None:
        raise HTTPException(status_code=500, detail="Index not initialized")
        
    if node_id not in index.nodes:
        raise HTTPException(status_code=404, detail="Node ID not found")
        
    index.delete(node_id)
    return {"status": "deleted", "node_id": node_id}

@app.post("/search", response_model=SearchResponse)
async def search_vectors(req: SearchRequest):
    if index is None:
        raise HTTPException(status_code=500, detail="Index not initialized")
        
    vec_array = np.array(req.vector, dtype=np.float32)
    
    current_dim = 0
    if isinstance(index.nodes, VectorStore):
        current_dim = index.nodes.dim
    elif len(index.nodes) > 0:
        current_dim = len(next(iter(index.nodes.values())))
        
    if current_dim > 0 and len(vec_array) != current_dim:
        raise HTTPException(status_code=400, detail=f"Vector dimension mismatch. Expected {current_dim}, got {len(vec_array)}")
        
    results = index.search(vec_array, k=req.k, ef=req.ef, filter_dict=req.filter)
    
    formatted = [SearchResult(node_id=n, distance=d, metadata=m) for n, d, m in results]
    return SearchResponse(results=formatted)

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    if index is None:
        raise HTTPException(status_code=500, detail="Index not initialized")
        
    current_dim = 0
    size = 0
    if isinstance(index.nodes, VectorStore):
        current_dim = index.nodes.dim
        size = len(index.nodes.node_id_to_index)
    else:
        size = len(index.nodes)
        if size > 0:
            current_dim = len(next(iter(index.nodes.values())))
            
    return StatsResponse(
        size=size,
        dim=current_dim,
        max_level=index.max_level,
        entry_point=index.entry_point,
        config={
            "M": index.config.M,
            "ef_construction": index.config.ef_construction,
            "ef_search": index.config.ef_search
        }
    )

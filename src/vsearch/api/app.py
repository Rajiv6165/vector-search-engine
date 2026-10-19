import os
import asyncio
import numpy as np
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import ValidationError
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("vsearch.api")

from vsearch.hnsw import HNSWIndex, HNSWConfig, VectorStore, ContiguousVectorStore
from vsearch.api.models import (
    InsertRequest, InsertResponse,
    BatchInsertRequest, BatchInsertResponse,
    SearchRequest, SearchResponse,
    BatchSearchRequest, BatchSearchResponse,
    SearchResult, StatsResponse
)

PERSIST_DIR = os.getenv("VSEARCH_PERSIST_DIR", "./vsearch_data")
SNAPSHOT_INTERVAL_SECONDS = int(os.getenv("VSEARCH_SNAPSHOT_INTERVAL", "300"))
VSEARCH_API_KEY = os.getenv("VSEARCH_API_KEY")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key_header: str = Security(api_key_header)):
    if VSEARCH_API_KEY:
        if api_key_header != VSEARCH_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return api_key_header

# Global index instance
index: Optional[HNSWIndex] = None

async def snapshot_loop():
    """Background task to periodically snapshot the index."""
    while True:
        await asyncio.sleep(SNAPSHOT_INTERVAL_SECONDS)
        if index is not None:
            logger.info("Running periodic background snapshot...")
            index.save_snapshot()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global index
    logger.info(f"Starting vsearch service. Persist dir: {PERSIST_DIR}")
    
    # Initialize or load index
    metadata_path = os.path.join(PERSIST_DIR, "metadata.json")
    if os.path.exists(metadata_path):
        logger.info("Found existing snapshot, loading...")
        index = HNSWIndex.load_snapshot(path=PERSIST_DIR, persist_dir=PERSIST_DIR)
        logger.info("Snapshot and WAL loaded.")
    else:
        logger.info("No existing snapshot found, initializing empty index.")
        config = HNSWConfig(M=16, ef_construction=100, ef_search=50)
        index = HNSWIndex(config=config, persist_dir=PERSIST_DIR)
        
    # Start background snapshot task
    task = asyncio.create_task(snapshot_loop())
    
    yield  # Run application
    
    logger.info("Shutting down vsearch service...")
    task.cancel()
    if index is not None:
        logger.info("Saving final snapshot...")
        index.save_snapshot()
        if hasattr(index.nodes, "close"):
            index.nodes.close()
        if index.wal:
            index.wal.close()
        logger.info("Shutdown complete.")

app = FastAPI(title="Vector Search Engine", lifespan=lifespan)

@app.post("/vectors", response_model=InsertResponse)
def insert_vector(req: InsertRequest, api_key: str = Depends(verify_api_key)):
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

@app.post("/vectors/batch", response_model=BatchInsertResponse)
def insert_vectors_batch(req: BatchInsertRequest, api_key: str = Depends(verify_api_key)):
    if index is None:
        raise HTTPException(status_code=500, detail="Index not initialized")
        
    if not req.vectors:
        return BatchInsertResponse(inserted_count=0, node_ids=[], status="inserted")
        
    vecs_array = np.array(req.vectors, dtype=np.float32)
    current_dim = 0
    if isinstance(index.nodes, VectorStore):
        current_dim = index.nodes.dim
    elif len(index.nodes) > 0:
        current_dim = len(next(iter(index.nodes.values())))
        
    if current_dim > 0 and vecs_array.shape[1] != current_dim:
        raise HTTPException(status_code=400, detail=f"Vector dimension mismatch. Expected {current_dim}, got {vecs_array.shape[1]}")
        
    node_ids = req.node_ids
    if node_ids is None:
        start_id = len(index.nodes)
        node_ids = list(range(start_id, start_id + len(vecs_array)))
        
    try:
        index.batch_insert(vecs_array, node_ids=node_ids, metadatas=req.metadatas)
        return BatchInsertResponse(inserted_count=len(node_ids), node_ids=node_ids, status="inserted")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/vectors/{node_id}")
def delete_vector(node_id: int, api_key: str = Depends(verify_api_key)):
    if index is None:
        raise HTTPException(status_code=500, detail="Index not initialized")
        
    if node_id not in index.nodes:
        raise HTTPException(status_code=404, detail="Node ID not found")
        
    index.delete(node_id)
    return {"status": "deleted", "node_id": node_id}

@app.post("/search", response_model=SearchResponse)
def search_vectors(req: SearchRequest, api_key: str = Depends(verify_api_key)):
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

@app.post("/search/batch", response_model=BatchSearchResponse)
def search_vectors_batch(req: BatchSearchRequest, api_key: str = Depends(verify_api_key)):
    if index is None:
        raise HTTPException(status_code=500, detail="Index not initialized")
        
    if not req.vectors:
        return BatchSearchResponse(results=[])
        
    queries_array = np.array(req.vectors, dtype=np.float32)
    current_dim = 0
    if isinstance(index.nodes, VectorStore):
        current_dim = index.nodes.dim
    elif len(index.nodes) > 0:
        current_dim = len(next(iter(index.nodes.values())))
        
    if current_dim > 0 and queries_array.shape[1] != current_dim:
        raise HTTPException(status_code=400, detail=f"Vector dimension mismatch. Expected {current_dim}, got {queries_array.shape[1]}")
        
    batch_results = index.batch_search(
        queries_array,
        k=req.k,
        ef=req.ef,
        filter_dict=req.filter,
        num_threads=req.num_threads
    )
    
    formatted_batch = [
        [SearchResult(node_id=n, distance=d, metadata=m) for n, d, m in row]
        for row in batch_results
    ]
    return BatchSearchResponse(results=formatted_batch)

@app.get("/stats", response_model=StatsResponse)
def get_stats():
    if index is None:
        raise HTTPException(status_code=500, detail="Index not initialized")
        
    current_dim = 0
    size = 0
    if isinstance(index.nodes, (VectorStore, ContiguousVectorStore)):
        current_dim = index.nodes.dim
        size = len(index.nodes.node_id_to_index) - len(index.deleted_nodes)
    else:
        size = len(index.nodes) - len(index.deleted_nodes)
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

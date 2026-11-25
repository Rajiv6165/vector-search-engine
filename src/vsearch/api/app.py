import os
import sys
import time
import random
import shutil
import asyncio
import secrets
import threading
from collections import defaultdict, deque
import numpy as np
import logging
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, Security, Depends, Request, Query
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, HTMLResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("vsearch.api")

import vsearch
from vsearch.hnsw import HNSWIndex, HNSWConfig, VectorStore, ContiguousVectorStore, HAS_NATIVE
from vsearch.api.models import (
    InsertRequest, InsertResponse,
    BatchInsertRequest, BatchInsertResponse,
    SearchRequest, SearchResponse,
    BatchSearchRequest, BatchSearchResponse,
    SearchResult, StatsResponse, HealthResponse,
    SimilarResponse, RandomNodeResponse
)

# ---------------------------------------------------------------------------
# Single-process safety verification (WEB_CONCURRENCY)
# ---------------------------------------------------------------------------
def check_single_process_safety():
    web_concurrency = os.getenv("WEB_CONCURRENCY")
    if web_concurrency is not None:
        try:
            if int(web_concurrency) > 1:
                raise RuntimeError(
                    f"vsearch must run as a single process (WEB_CONCURRENCY={web_concurrency} detected). "
                    "Multi-worker configurations corrupt in-memory HNSW graphs and memmap WAL state. "
                    "Use single-process concurrency."
                )
        except ValueError:
            pass

check_single_process_safety()

# ---------------------------------------------------------------------------
# Exclusive Data Directory Lock File
# ---------------------------------------------------------------------------
class DataDirLock:
    """Exclusive lockfile to prevent multiple processes from attaching to the same data directory."""
    def __init__(self, data_dir: str):
        self.lock_path = os.path.join(data_dir, "vsearch.lock")
        self._file = None

    def acquire(self):
        os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)
        self._file = open(self.lock_path, "a+")
        try:
            if sys.platform == "win32":
                import msvcrt
                self._file.seek(0)
                msvcrt.locking(self._file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (IOError, OSError, PermissionError) as e:
            raise RuntimeError(
                f"Another vsearch process is currently running on data directory '{os.path.dirname(self.lock_path)}'. "
                "Exclusive lock acquisition failed; only one process may access the data directory at a time."
            ) from e

    def release(self):
        if self._file:
            try:
                if sys.platform == "win32":
                    import msvcrt
                    self._file.seek(0)
                    msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

# ---------------------------------------------------------------------------
# Per-IP Rate Limiting for Search Endpoints
# ---------------------------------------------------------------------------
class IPRateLimiter:
    """Sliding-window per-IP rate limiter."""
    def __init__(self, max_requests: int = 100, window_seconds: float = 1.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check_rate_limit(self, client_ip: str) -> bool:
        if self.max_requests <= 0:
            return True
        now = time.time()
        with self._lock:
            queue = self._requests[client_ip]
            while queue and now - queue[0] > self.window_seconds:
                queue.popleft()
            if len(queue) >= self.max_requests:
                return False
            queue.append(now)
            return True

def get_client_ip(request: Request) -> str:
    """Extract client IP respecting X-Forwarded-For when behind a reverse proxy."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        # First IP in comma-separated chain is the originating client IP
        return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"

# ---------------------------------------------------------------------------
# Configuration & Limits
# ---------------------------------------------------------------------------
PORT = int(os.getenv("PORT", "8000"))
PERSIST_DIR = os.getenv("VSEARCH_DATA_DIR") or os.getenv("VSEARCH_PERSIST_DIR", "./vsearch_data")
SNAPSHOT_INTERVAL_SECONDS = int(os.getenv("VSEARCH_SNAPSHOT_INTERVAL", "300"))
BAKED_SNAPSHOT_DIR = os.getenv("VSEARCH_BAKED_SNAPSHOT_DIR", "/app/demo_snapshot")

VSEARCH_API_KEY = os.getenv("VSEARCH_API_KEY")
VSEARCH_READ_KEY = os.getenv("VSEARCH_READ_KEY")
VSEARCH_PUBLIC_KEY = os.getenv("VSEARCH_PUBLIC_KEY")
VSEARCH_ALLOW_INSECURE = os.getenv("VSEARCH_ALLOW_INSECURE", "0") in ("1", "true", "True")

MAX_VECTORS_ENV = os.getenv("VSEARCH_MAX_VECTORS")
MAX_VECTORS = int(MAX_VECTORS_ENV) if MAX_VECTORS_ENV is not None else None

MAX_K_ENV = os.getenv("VSEARCH_MAX_K")
MAX_K = int(MAX_K_ENV) if MAX_K_ENV is not None else 1000

MAX_EF_ENV = os.getenv("VSEARCH_MAX_EF")
MAX_EF = int(MAX_EF_ENV) if MAX_EF_ENV is not None else 500

MAX_BATCH_ENV = os.getenv("VSEARCH_MAX_BATCH")
MAX_BATCH = int(MAX_BATCH_ENV) if MAX_BATCH_ENV is not None else 1000

MAX_REQUEST_BYTES_ENV = os.getenv("VSEARCH_MAX_REQUEST_BYTES")
MAX_REQUEST_BYTES = int(MAX_REQUEST_BYTES_ENV) if MAX_REQUEST_BYTES_ENV is not None else 10 * 1024 * 1024

RATE_LIMIT_RPS_ENV = os.getenv("VSEARCH_RATE_LIMIT_RPS")
RATE_LIMIT_RPS = int(RATE_LIMIT_RPS_ENV) if RATE_LIMIT_RPS_ENV is not None else 100
rate_limiter = IPRateLimiter(max_requests=RATE_LIMIT_RPS, window_seconds=1.0)

SEED_ON_START = os.getenv("VSEARCH_SEED_ON_START", "0").lower() in ("1", "true", "yes")

# Service ready state and background seeding control
is_ready: bool = True
seeding_progress: Optional[float] = None
ready_event = threading.Event()
ready_event.set()

# Active directory lock instance
dir_lock: Optional[DataDirLock] = None

# ---------------------------------------------------------------------------
# Auth Handlers (Fails Closed with secrets.compare_digest)
# ---------------------------------------------------------------------------
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_write_key(api_key: Optional[str] = Security(api_key_header)):
    if VSEARCH_API_KEY:
        if not api_key or not secrets.compare_digest(api_key, VSEARCH_API_KEY):
            raise HTTPException(status_code=401, detail="Invalid or missing write API Key")
    elif not VSEARCH_ALLOW_INSECURE:
        raise HTTPException(status_code=401, detail="Authentication required: VSEARCH_API_KEY is not configured")
    return api_key

def verify_read_key(api_key: Optional[str] = Security(api_key_header)):
    if not VSEARCH_API_KEY and not VSEARCH_READ_KEY and VSEARCH_ALLOW_INSECURE:
        return api_key

    valid = False
    if api_key:
        if VSEARCH_API_KEY and secrets.compare_digest(api_key, VSEARCH_API_KEY):
            valid = True
        if VSEARCH_READ_KEY and secrets.compare_digest(api_key, VSEARCH_READ_KEY):
            valid = True

    if not valid:
        raise HTTPException(status_code=401, detail="Invalid or missing read API Key")
    return api_key

def verify_demo_key(api_key: Optional[str] = Security(api_key_header)):
    """Verifies access to public demo read routes (allows VSEARCH_PUBLIC_KEY, read key, or write key)."""
    if not VSEARCH_API_KEY and not VSEARCH_READ_KEY and not VSEARCH_PUBLIC_KEY and VSEARCH_ALLOW_INSECURE:
        return api_key

    valid = False
    if api_key:
        if VSEARCH_API_KEY and secrets.compare_digest(api_key, VSEARCH_API_KEY):
            valid = True
        if VSEARCH_READ_KEY and secrets.compare_digest(api_key, VSEARCH_READ_KEY):
            valid = True
        if VSEARCH_PUBLIC_KEY and secrets.compare_digest(api_key, VSEARCH_PUBLIC_KEY):
            valid = True

    if not valid:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key for demo routes")
    return api_key

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "templates", "index.html")

def render_demo_html(public_key: str = "") -> str:
    if os.path.exists(TEMPLATE_PATH):
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            content = f.read()
            return content.replace("{{ PUBLIC_KEY }}", public_key or "")
    return "<html><body><h1>vsearch demo</h1><p>Demo template not found.</p></body></html>"

# Global index instance
index: Optional[HNSWIndex] = None

async def snapshot_loop():
    """Background task to periodically snapshot the index."""
    while True:
        await asyncio.sleep(SNAPSHOT_INTERVAL_SECONDS)
        if index is not None and is_ready:
            logger.info("Running periodic background snapshot...")
            index.save_snapshot()

def _run_background_seeding():
    """Seeds ~20,000 demo vectors in a background thread without blocking port serving."""
    global is_ready, seeding_progress
    logger.info("Starting background demo seeding thread...")
    try:
        from scripts.seed_demo import seed_index
        def on_progress(fraction: float):
            global seeding_progress
            seeding_progress = fraction
        seed_index(index, total_vectors=20000, dim=64, batch_size=1000, progress_callback=on_progress)
    except Exception as e:
        logger.error(f"Error during background demo seeding: {e}", exc_info=True)
    finally:
        is_ready = True
        seeding_progress = 1.0
        ready_event.set()
        logger.info("Background demo seeding finished. Service marked ready: true.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global index, dir_lock, is_ready, seeding_progress
    check_single_process_safety()

    # Auth fails closed verification
    current_api_key = os.getenv("VSEARCH_API_KEY")
    current_allow_insecure = os.getenv("VSEARCH_ALLOW_INSECURE", "0") in ("1", "true", "True")
    if not current_api_key and not current_allow_insecure:
        raise RuntimeError(
            "VSEARCH_API_KEY is not set. For security, vsearch refuses to start without an API key. "
            "Set VSEARCH_API_KEY='your-secret-key' or set VSEARCH_ALLOW_INSECURE=1 for local development."
        )

    logger.info(f"Starting vsearch service. Data dir: {PERSIST_DIR}")
    
    # Acquire exclusive lockfile on data directory
    dir_lock = DataDirLock(PERSIST_DIR)
    dir_lock.acquire()

    # Check if a pre-baked snapshot is available for instant cold-starts
    metadata_path = os.path.join(PERSIST_DIR, "metadata.json")
    baked_metadata = os.path.join(BAKED_SNAPSHOT_DIR, "metadata.json")
    if not os.path.exists(metadata_path) and os.path.exists(baked_metadata):
        logger.info(f"Copying pre-baked demo snapshot from {BAKED_SNAPSHOT_DIR} to {PERSIST_DIR}...")
        for fname in os.listdir(BAKED_SNAPSHOT_DIR):
            shutil.copy2(os.path.join(BAKED_SNAPSHOT_DIR, fname), os.path.join(PERSIST_DIR, fname))

    # Initialize or load index
    if os.path.exists(metadata_path):
        logger.info("Found existing snapshot, loading...")
        index = HNSWIndex.load_snapshot(path=PERSIST_DIR, persist_dir=PERSIST_DIR)
        logger.info(f"Snapshot and WAL loaded ({len(index.nodes)} vectors).")
        is_ready = True
        seeding_progress = 1.0
        ready_event.set()
    else:
        logger.info("No existing snapshot found, initializing empty index.")
        config = HNSWConfig(M=16, ef_construction=100, ef_search=50)
        index = HNSWIndex(config=config, persist_dir=PERSIST_DIR)

        current_size = _get_index_size()
        auto_seed = os.getenv("VSEARCH_AUTO_SEED", "1").lower() in ("1", "true", "yes")
        if auto_seed and (current_size == 0 or SEED_ON_START):
            logger.info("Scheduling non-blocking background demo seeding...")
            is_ready = False
            seeding_progress = 0.0
            ready_event.clear()
            seeder_thread = threading.Thread(target=_run_background_seeding, daemon=True)
            seeder_thread.start()
        else:
            is_ready = True
            seeding_progress = 1.0
            ready_event.set()

    # Start background snapshot task
    task = asyncio.create_task(snapshot_loop())
    
    yield  # Port is served immediately

    logger.info("Shutting down vsearch service...")
    task.cancel()
    if index is not None:
        logger.info("Saving final snapshot...")
        index.save_snapshot()
        if hasattr(index.nodes, "close"):
            index.nodes.close()
        if index.wal:
            index.wal.close()

    if dir_lock:
        dir_lock.release()
        dir_lock = None
    logger.info("Shutdown complete.")

app = FastAPI(title="Vector Search Engine", lifespan=lifespan)

# Request size limiter middleware
class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_bytes: Optional[int] = None):
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        limit = MAX_REQUEST_BYTES if MAX_REQUEST_BYTES is not None else self.max_bytes
        content_length = request.headers.get("content-length")
        if content_length and limit is not None:
            try:
                if int(content_length) > limit:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": f"Request body exceeds maximum allowed size ({int(content_length)} > {limit} bytes)"}
                    )
            except ValueError:
                pass
        return await call_next(request)

app.add_middleware(RequestSizeLimitMiddleware, max_bytes=MAX_REQUEST_BYTES)

def _get_current_dim() -> int:
    if index is None:
        return 0
    if isinstance(index.nodes, (VectorStore, ContiguousVectorStore)):
        return index.nodes.dim
    elif len(index.nodes) > 0:
        return len(next(iter(index.nodes.values())))
    return 0

def _get_index_size() -> int:
    if index is None:
        return 0
    if isinstance(index.nodes, (VectorStore, ContiguousVectorStore)):
        return len(index.nodes.node_id_to_index) - len(index.deleted_nodes)
    return len(index.nodes) - len(index.deleted_nodes)

def _node_exists(node_id: int) -> bool:
    if index is None:
        return False
    if isinstance(index.nodes, (VectorStore, ContiguousVectorStore)):
        return node_id in index.nodes.node_id_to_index
    return node_id in index.nodes

# ---------------------------------------------------------------------------
# Public Endpoints (No Auth)
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def get_health():
    """Unauthenticated healthcheck endpoint. Available immediately on server listen."""
    return HealthResponse(status="ok")

# ---------------------------------------------------------------------------
# Write Endpoints (Require Write Key)
# ---------------------------------------------------------------------------
@app.post("/vectors", response_model=InsertResponse)
def insert_vector(req: InsertRequest, api_key: str = Depends(verify_write_key)):
    if index is None:
        raise HTTPException(status_code=500, detail="Index not initialized")
        
    vec_array = np.array(req.vector, dtype=np.float32)
    if vec_array.size == 0:
        raise HTTPException(status_code=400, detail="Vector cannot be empty")
    if not np.all(np.isfinite(vec_array)):
        raise HTTPException(status_code=400, detail="Vector contains non-finite values (NaN or Inf)")

    current_dim = _get_current_dim()
    if current_dim > 0 and len(vec_array) != current_dim:
        raise HTTPException(status_code=400, detail=f"Vector dimension mismatch. Expected {current_dim}, got {len(vec_array)}")

    if MAX_VECTORS is not None:
        current_size = _get_index_size()
        if not _node_exists(req.node_id) and current_size >= MAX_VECTORS:
            raise HTTPException(status_code=400, detail=f"Index size limit reached (MAX_VECTORS={MAX_VECTORS})")
        
    try:
        index.insert(req.node_id, vec_array, metadata=req.metadata)
        return InsertResponse(node_id=req.node_id, status="inserted")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/vectors/batch", response_model=BatchInsertResponse)
def insert_vectors_batch(req: BatchInsertRequest, api_key: str = Depends(verify_write_key)):
    if index is None:
        raise HTTPException(status_code=500, detail="Index not initialized")
        
    if not req.vectors:
        return BatchInsertResponse(inserted_count=0, node_ids=[], status="inserted")

    if MAX_BATCH is not None and len(req.vectors) > MAX_BATCH:
        raise HTTPException(status_code=400, detail=f"Batch size {len(req.vectors)} exceeds maximum allowed (MAX_BATCH={MAX_BATCH})")

    vecs_array = np.array(req.vectors, dtype=np.float32)
    if vecs_array.ndim != 2 or vecs_array.shape[1] == 0:
        raise HTTPException(status_code=400, detail="Invalid vectors dimensions")
    if not np.all(np.isfinite(vecs_array)):
        raise HTTPException(status_code=400, detail="Vectors contain non-finite values (NaN or Inf)")

    current_dim = _get_current_dim()
    if current_dim > 0 and vecs_array.shape[1] != current_dim:
        raise HTTPException(status_code=400, detail=f"Vector dimension mismatch. Expected {current_dim}, got {vecs_array.shape[1]}")
        
    node_ids = req.node_ids
    if node_ids is None:
        start_id = len(index.nodes)
        node_ids = list(range(start_id, start_id + len(vecs_array)))

    if MAX_VECTORS is not None:
        current_size = _get_index_size()
        new_nodes_count = sum(1 for nid in node_ids if not _node_exists(nid))
        if current_size + new_nodes_count > MAX_VECTORS:
            raise HTTPException(
                status_code=400,
                detail=f"Batch insert exceeds MAX_VECTORS limit ({current_size + new_nodes_count} > {MAX_VECTORS})"
            )
        
    try:
        index.batch_insert(vecs_array, node_ids=node_ids, metadatas=req.metadatas)
        return BatchInsertResponse(inserted_count=len(node_ids), node_ids=node_ids, status="inserted")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/vectors/{node_id}")
def delete_vector(node_id: int, api_key: str = Depends(verify_write_key)):
    if index is None:
        raise HTTPException(status_code=500, detail="Index not initialized")
        
    if not _node_exists(node_id):
        raise HTTPException(status_code=404, detail="Node ID not found")
        
    index.delete(node_id)
    return {"status": "deleted", "node_id": node_id}

# ---------------------------------------------------------------------------
# Read Endpoints (Require Read Key or Write Key, IP Rate-Limited)
# ---------------------------------------------------------------------------
@app.post("/search", response_model=SearchResponse)
def search_vectors(req: SearchRequest, request: Request, api_key: str = Depends(verify_read_key)):
    if index is None:
        raise HTTPException(status_code=500, detail="Index not initialized")

    # Rate limiting
    client_ip = get_client_ip(request)
    if not rate_limiter.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Too many search requests from this IP.")

    # Readiness check
    if not is_ready:
        if not ready_event.wait(timeout=0.05):
            pct = round((seeding_progress or 0.0) * 100)
            raise HTTPException(status_code=503, detail=f"Index is initializing demo dataset ({pct}% complete). Please retry shortly.")

    if MAX_K is not None and req.k > MAX_K:
        raise HTTPException(status_code=400, detail=f"Requested k={req.k} exceeds maximum allowed (MAX_K={MAX_K})")

    if req.ef is not None and MAX_EF is not None and req.ef > MAX_EF:
        raise HTTPException(status_code=400, detail=f"Requested ef={req.ef} exceeds maximum allowed (MAX_EF={MAX_EF})")
        
    vec_array = np.array(req.vector, dtype=np.float32)
    if vec_array.size == 0:
        raise HTTPException(status_code=400, detail="Query vector cannot be empty")
    if not np.all(np.isfinite(vec_array)):
        raise HTTPException(status_code=400, detail="Query vector contains non-finite values (NaN or Inf)")

    current_dim = _get_current_dim()
    if current_dim > 0 and len(vec_array) != current_dim:
        raise HTTPException(status_code=400, detail=f"Vector dimension mismatch. Expected {current_dim}, got {len(vec_array)}")
        
    results = index.search(vec_array, k=req.k, ef=req.ef, filter_dict=req.filter)
    formatted = [SearchResult(node_id=n, distance=d, metadata=m) for n, d, m in results]
    return SearchResponse(results=formatted)

@app.post("/search/batch", response_model=BatchSearchResponse)
def search_vectors_batch(req: BatchSearchRequest, request: Request, api_key: str = Depends(verify_read_key)):
    if index is None:
        raise HTTPException(status_code=500, detail="Index not initialized")

    # Rate limiting
    client_ip = get_client_ip(request)
    if not rate_limiter.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Too many search requests from this IP.")

    # Readiness check
    if not is_ready:
        if not ready_event.wait(timeout=0.05):
            pct = round((seeding_progress or 0.0) * 100)
            raise HTTPException(status_code=503, detail=f"Index is initializing demo dataset ({pct}% complete). Please retry shortly.")
        
    if not req.vectors:
        return BatchSearchResponse(results=[])

    if MAX_BATCH is not None and len(req.vectors) > MAX_BATCH:
        raise HTTPException(status_code=400, detail=f"Batch query count {len(req.vectors)} exceeds maximum allowed (MAX_BATCH={MAX_BATCH})")

    if MAX_K is not None and req.k > MAX_K:
        raise HTTPException(status_code=400, detail=f"Requested k={req.k} exceeds maximum allowed (MAX_K={MAX_K})")

    if req.ef is not None and MAX_EF is not None and req.ef > MAX_EF:
        raise HTTPException(status_code=400, detail=f"Requested ef={req.ef} exceeds maximum allowed (MAX_EF={MAX_EF})")

    queries_array = np.array(req.vectors, dtype=np.float32)
    if queries_array.ndim != 2 or queries_array.shape[1] == 0:
        raise HTTPException(status_code=400, detail="Invalid query vectors array")
    if not np.all(np.isfinite(queries_array)):
        raise HTTPException(status_code=400, detail="Query vectors contain non-finite values (NaN or Inf)")

    current_dim = _get_current_dim()
    if current_dim > 0 and queries_array.shape[1] != current_dim:
        raise HTTPException(status_code=400, detail=f"Vector dimension mismatch. Expected {current_dim}, got {queries_array.shape[1]}")
        
    if req.filters is not None and len(req.filters) != len(req.vectors):
        raise HTTPException(status_code=400, detail="Length of filters list must match length of vectors list")

    batch_results = index.batch_search(
        queries_array,
        k=req.k,
        ef=req.ef,
        filter_dict=req.filter,
        filters=req.filters,
        num_threads=req.num_threads
    )
    
    formatted_batch = [
        [SearchResult(node_id=n, distance=d, metadata=m) for n, d, m in row]
        for row in batch_results
    ]
    return BatchSearchResponse(results=formatted_batch)

@app.get("/stats", response_model=StatsResponse)
def get_stats(api_key: str = Depends(verify_read_key)):
    if index is None:
        raise HTTPException(status_code=500, detail="Index not initialized")
        
    return StatsResponse(
        size=_get_index_size(),
        dim=_get_current_dim(),
        max_level=index.max_level,
        entry_point=index.entry_point,
        config={
            "M": index.config.M,
            "ef_construction": index.config.ef_construction,
            "ef_search": index.config.ef_search
        },
        native=HAS_NATIVE,
        version=getattr(vsearch, "__version__", "0.1.0"),
        ready=is_ready,
        progress=seeding_progress
    )

# ---------------------------------------------------------------------------
# Phase 12: Public Demo Endpoints
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def get_demo_page():
    """Serves the interactive single-page public demo."""
    key = VSEARCH_PUBLIC_KEY or ""
    return HTMLResponse(content=render_demo_html(public_key=key))

@app.get("/similar/{node_id}", response_model=SimilarResponse)
def get_similar_vectors(
    node_id: int,
    k: int = Query(10, ge=1, description="Number of nearest neighbors to return"),
    category: Optional[str] = Query(None, description="Optional category filter"),
    request: Request = None,
    api_key: str = Depends(verify_demo_key)
):
    if index is None:
        raise HTTPException(status_code=500, detail="Index not initialized")

    # Rate limiting
    if request is not None:
        client_ip = get_client_ip(request)
        if not rate_limiter.check_rate_limit(client_ip):
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Too many search requests from this IP.")

    # Readiness check
    if not is_ready:
        if not ready_event.wait(timeout=0.05):
            pct = round((seeding_progress or 0.0) * 100)
            raise HTTPException(status_code=503, detail=f"Index is initializing demo dataset ({pct}% complete). Please retry shortly.")

    if MAX_K is not None and k > MAX_K:
        raise HTTPException(status_code=400, detail=f"Requested k={k} exceeds maximum allowed (MAX_K={MAX_K})")

    # Check node existence
    if node_id not in index.nodes or node_id in index.deleted_nodes:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

    vec = index.nodes[node_id]
    filter_dict = {"category": category} if category else None

    start_time = time.perf_counter()
    results = index.search(vec, k=k, filter_dict=filter_dict)
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    search_path = getattr(index, "last_search_path", "graph")
    used_exact_fallback = (search_path == "exact_fallback")

    formatted = [SearchResult(node_id=n, distance=d, metadata=m) for n, d, m in results]
    return SimilarResponse(
        node_id=node_id,
        k=k,
        category=category,
        results=formatted,
        latency_ms=round(latency_ms, 3),
        search_path=search_path,
        used_exact_fallback=used_exact_fallback
    )

@app.get("/demo/random", response_model=RandomNodeResponse)
def get_random_node(
    category: Optional[str] = Query(None, description="Optional category filter"),
    request: Request = None,
    api_key: str = Depends(verify_demo_key)
):
    if index is None:
        raise HTTPException(status_code=500, detail="Index not initialized")

    # Rate limiting
    if request is not None:
        client_ip = get_client_ip(request)
        if not rate_limiter.check_rate_limit(client_ip):
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Too many search requests from this IP.")

    # Readiness check
    if not is_ready:
        if not ready_event.wait(timeout=0.05):
            pct = round((seeding_progress or 0.0) * 100)
            raise HTTPException(status_code=503, detail=f"Index is initializing demo dataset ({pct}% complete). Please retry shortly.")

    if category:
        matching = index.metadata_index.get_matching_ids({"category": category})
        if matching is None:
            candidates = [n for n, m in index.metadata_store.items() if m.get("category") == category and n not in index.deleted_nodes and n in index.nodes]
        else:
            candidates = [n for n in matching if n not in index.deleted_nodes and n in index.nodes]
        if not candidates:
            raise HTTPException(status_code=404, detail=f"No vectors found in category '{category}'")
    else:
        candidates = [n for n in index.nodes.node_id_to_index.keys() if n not in index.deleted_nodes]
        if not candidates:
            raise HTTPException(status_code=404, detail="Index contains no active vectors")

    chosen_id = random.choice(candidates)
    meta = index.metadata_store.get(chosen_id, {})
    return RandomNodeResponse(node_id=chosen_id, metadata=meta)


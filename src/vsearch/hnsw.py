import math
import random
import heapq
import os
import json
import struct
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Callable, Optional, Union
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor

try:
    from vsearch._hnsw_native import (
        search_layer_native,
        greedy_search_layer_native,
        select_neighbors_heuristic_native,
        batch_search_layer_native,
        compute_distance_native,
    )
    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False

class RWLock:
    def __init__(self):
        self._condition = threading.Condition(threading.Lock())
        self._readers = 0
        self._writers = 0
        self._pending_writers = 0
        
    def acquire_read(self):
        with self._condition:
            while self._writers > 0 or self._pending_writers > 0:
                self._condition.wait()
            self._readers += 1
            
    def release_read(self):
        with self._condition:
            self._readers -= 1
            if not self._readers:
                self._condition.notify_all()
                
    def acquire_write(self):
        with self._condition:
            self._pending_writers += 1
            while self._writers > 0 or self._readers > 0:
                self._condition.wait()
            self._pending_writers -= 1
            self._writers += 1
            
    def release_write(self):
        with self._condition:
            self._writers -= 1
            self._condition.notify_all()

    class _ReadContext:
        def __init__(self, lock):
            self.lock = lock
        def __enter__(self):
            self.lock.acquire_read()
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.lock.release_read()
            
    class _WriteContext:
        def __init__(self, lock):
            self.lock = lock
        def __enter__(self):
            self.lock.acquire_write()
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.lock.release_write()
            
    def read(self):
        return self._ReadContext(self)
        
    def write(self):
        return self._WriteContext(self)

def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return float(1.0 - dot / (norm_a * norm_b))

@dataclass
class HNSWConfig:
    M: int = 16
    ef_construction: int = 100
    ef_search: int = 50
    m_L: float = None  # Normalization factor for level generation
    wal_batch_size: int = 100 # Flush WAL after this many writes to reduce fsync overhead
    heuristic: bool = True # Paper Algorithm 4 diversity-aware heuristic neighbor selection
    extend_candidates: bool = False
    keep_pruned_connections: bool = True
    use_native: bool = True
    seed: Optional[int] = None

    def __post_init__(self):
        if self.m_L is None:
            self.m_L = 1.0 / math.log(self.M)


class ContiguousVectorStore:
    """
    Contiguous in-memory float32 matrix store.
    Keeps vectors in a single contiguous float32 2D array to enable
    zero-copy native SIMD distance computation without per-node Python object access.
    """
    def __init__(self, initial_capacity: int = 1000, dtype=np.float32):
        self.dtype = dtype
        self.initial_capacity = initial_capacity
        self.capacity = initial_capacity
        self.size = 0
        self.dim = 0
        self.vectors: Optional[np.ndarray] = None
        self.node_id_to_index: Dict[int, int] = {}
        self.index_to_node_id: Dict[int, int] = {}
        self.mmap = None

    def _init_vectors(self, dim: int):
        self.dim = dim
        self.capacity = self.initial_capacity
        self.vectors = np.zeros((self.capacity, self.dim), dtype=self.dtype)

    def _resize(self, new_capacity: int):
        new_vecs = np.zeros((new_capacity, self.dim), dtype=self.dtype)
        if self.vectors is not None and self.size > 0:
            new_vecs[:self.size] = self.vectors[:self.size]
        self.vectors = new_vecs
        self.capacity = new_capacity

    def __setitem__(self, node_id: int, vector: np.ndarray):
        if self.vectors is None:
            self._init_vectors(len(vector))
        if node_id not in self.node_id_to_index:
            if self.size >= self.capacity:
                self._resize(self.capacity * 2)
            idx = self.size
            self.node_id_to_index[node_id] = idx
            self.index_to_node_id[idx] = node_id
            self.size += 1
        else:
            idx = self.node_id_to_index[node_id]
        self.vectors[idx] = vector

    def __getitem__(self, node_id: int) -> np.ndarray:
        return self.vectors[self.node_id_to_index[node_id]]

    def __delitem__(self, node_id: int):
        if node_id in self.node_id_to_index:
            idx = self.node_id_to_index[node_id]
            del self.node_id_to_index[node_id]
            if idx in self.index_to_node_id:
                del self.index_to_node_id[idx]

    def __contains__(self, node_id: int) -> bool:
        return node_id in self.node_id_to_index

    def __len__(self) -> int:
        return len(self.node_id_to_index)

    def values(self):
        return [self.vectors[idx] for idx in self.node_id_to_index.values()]

    def items(self):
        return [(node_id, self.vectors[idx]) for node_id, idx in self.node_id_to_index.items()]

    def get_matrix(self) -> np.ndarray:
        if self.vectors is None or self.size == 0:
            return np.empty((0, self.dim), dtype=self.dtype)
        return self.vectors[:self.size]

    def flush(self):
        pass

    def close(self):
        pass


class VectorStore:
    def __init__(self, filepath: str, dtype=np.float32, initial_capacity: int = 1000):
        self.filepath = filepath
        self.dtype = dtype
        self.initial_capacity = initial_capacity
        self.node_id_to_index: Dict[int, int] = {}
        self.index_to_node_id: Dict[int, int] = {}
        self.size = 0
        self.capacity = 0
        self.dim = 0
        self.mmap = None

    def _init_memmap(self, dim: int):
        self.dim = dim
        self.capacity = self.initial_capacity
        bytes_per_vec = self.dim * np.dtype(self.dtype).itemsize
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'wb') as f:
                f.seek(self.capacity * bytes_per_vec - 1)
                f.write(b'\0')
        else:
            file_size = os.path.getsize(self.filepath)
            self.capacity = max(self.initial_capacity, file_size // bytes_per_vec)
        self.mmap = np.memmap(self.filepath, dtype=self.dtype, mode='r+', shape=(self.capacity, self.dim))

    def _resize(self, new_capacity: int):
        if self.mmap is not None:
            self.mmap.flush()
            if hasattr(self.mmap, '_mmap'):
                self.mmap._mmap.close()
            del self.mmap
            
        bytes_per_vec = self.dim * np.dtype(self.dtype).itemsize
        with open(self.filepath, 'a+b') as f:
            f.truncate(new_capacity * bytes_per_vec)
        self.capacity = new_capacity
        self.mmap = np.memmap(self.filepath, dtype=self.dtype, mode='r+', shape=(self.capacity, self.dim))

    def __setitem__(self, node_id: int, vector: np.ndarray):
        if self.mmap is None:
            self._init_memmap(len(vector))
        if node_id not in self.node_id_to_index:
            if self.size >= self.capacity:
                self._resize(self.capacity * 2)
            idx = self.size
            self.node_id_to_index[node_id] = idx
            self.index_to_node_id[idx] = node_id
            self.size += 1
        else:
            idx = self.node_id_to_index[node_id]
        self.mmap[idx] = vector

    def __getitem__(self, node_id: int) -> np.ndarray:
        return self.mmap[self.node_id_to_index[node_id]]

    def __delitem__(self, node_id: int):
        if node_id in self.node_id_to_index:
            idx = self.node_id_to_index[node_id]
            del self.node_id_to_index[node_id]
            if idx in self.index_to_node_id:
                del self.index_to_node_id[idx]

    def __contains__(self, node_id: int) -> bool:
        return node_id in self.node_id_to_index

    def __len__(self) -> int:
        return len(self.node_id_to_index)
        
    def values(self):
        return [self.mmap[idx] for idx in self.node_id_to_index.values()]

    def items(self):
        return [(node_id, self.mmap[idx]) for node_id, idx in self.node_id_to_index.items()]

    def get_matrix(self) -> np.ndarray:
        if self.mmap is None or self.size == 0:
            return np.empty((0, self.dim), dtype=self.dtype)
        return np.asarray(self.mmap[:self.size])

    def flush(self):
        if self.mmap is not None:
            self.mmap.flush()

    def close(self):
        self.flush()
        if self.mmap is not None:
            if hasattr(self.mmap, '_mmap'):
                self.mmap._mmap.close()
            del self.mmap
            self.mmap = None


class WALManager:
    def __init__(self, filepath: str, batch_size: int = 100):
        self.filepath = filepath
        self.file = None
        self.batch_size = batch_size
        self.writes_since_flush = 0
        self._open()

    def _open(self):
        if self.file is None:
            self.file = open(self.filepath, "ab")

    def log_insert(self, node_id: int, vector: np.ndarray, metadata: dict = None):
        if metadata:
            header = struct.pack('<BI', 3, node_id)
            vec_bytes = vector.astype(np.float32).tobytes()
            meta_json = json.dumps(metadata).encode('utf-8')
            meta_len = struct.pack('<I', len(meta_json))
            self.file.write(header + vec_bytes + meta_len + meta_json)
        else:
            header = struct.pack('<BI', 1, node_id)
            vec_bytes = vector.astype(np.float32).tobytes()
            self.file.write(header + vec_bytes)
        self.writes_since_flush += 1
        if self.writes_since_flush >= self.batch_size:
            self.flush()

    def log_delete(self, node_id: int):
        header = struct.pack('<BI', 2, node_id)
        self.file.write(header)
        self.writes_since_flush += 1
        if self.writes_since_flush >= self.batch_size:
            self.flush()

    def flush(self):
        if self.file:
            self.file.flush()
            if hasattr(os, 'fsync'):
                os.fsync(self.file.fileno())
        self.writes_since_flush = 0

    def truncate(self):
        if self.file:
            self.file.close()
        self.file = open(self.filepath, "wb")
        self.file.close()
        self.file = open(self.filepath, "ab")
        
    def close(self):
        if self.file:
            self.file.close()
            self.file = None

    def replay(self, insert_fn, delete_fn, dim: int):
        if not os.path.exists(self.filepath):
            return
            
        bytes_per_vec = dim * 4 # Assuming float32
        with open(self.filepath, "rb") as f:
            while True:
                op_buf = f.read(1)
                if not op_buf:
                    break
                op = struct.unpack('<B', op_buf)[0]
                if op == 1: # INSERT
                    id_buf = f.read(4)
                    node_id = struct.unpack('<I', id_buf)[0]
                    vec_buf = f.read(bytes_per_vec)
                    vector = np.frombuffer(vec_buf, dtype=np.float32).copy()
                    insert_fn(node_id, vector, None)
                elif op == 2: # DELETE
                    id_buf = f.read(4)
                    node_id = struct.unpack('<I', id_buf)[0]
                    delete_fn(node_id)
                elif op == 3: # INSERT_WITH_METADATA
                    id_buf = f.read(4)
                    node_id = struct.unpack('<I', id_buf)[0]
                    vec_buf = f.read(bytes_per_vec)
                    vector = np.frombuffer(vec_buf, dtype=np.float32).copy()
                    meta_len_buf = f.read(4)
                    meta_len = struct.unpack('<I', meta_len_buf)[0]
                    meta_json = f.read(meta_len).decode('utf-8')
                    metadata = json.loads(meta_json)
                    insert_fn(node_id, vector, metadata)
                else:
                    raise ValueError(f"Corrupt WAL: unknown op {op}")


class HNSWIndex:
    def __init__(self, config: HNSWConfig = None, metric: str = "l2", persist_dir: str = None):
        self.config = config if config is not None else HNSWConfig()
        self.metric = metric
        if metric == "l2":
            self.distance_fn = l2_distance
            self.metric_code = 0
        elif metric == "cosine":
            self.distance_fn = cosine_distance
            self.metric_code = 1
        else:
            raise ValueError("metric must be 'l2' or 'cosine'")
        
        self.persist_dir = persist_dir
        self.max_level = -1
        self.entry_point = None
        self.metadata_store: Dict[int, dict] = {}
        
        self.use_native = self.config.use_native and HAS_NATIVE
        self.rng = random.Random(self.config.seed) if self.config.seed is not None else random
        
        if self.persist_dir:
            if not os.path.exists(self.persist_dir):
                os.makedirs(self.persist_dir)
            self.nodes = VectorStore(os.path.join(self.persist_dir, "vectors.bin"))
            self.wal = WALManager(os.path.join(self.persist_dir, "wal.log"), batch_size=self.config.wal_batch_size)
        else:
            self.nodes = ContiguousVectorStore()
            self.wal = None
            
        self.graphs: List[Dict[int, Set[int]]] = []
        self.layer_edges: List[np.ndarray] = []
        self.layer_degrees: List[np.ndarray] = []
        self.lock = RWLock()
        self.deleted_nodes: Set[int] = set()
        self.last_visited_count: int = 0

    def seed_level_generator(self, seed: int):
        """Seed the level generator RNG for deterministic index construction."""
        self.rng = random.Random(seed)

    def _ensure_graph_capacity(self, level: int):
        while len(self.graphs) <= level:
            self.graphs.append({})
            M_max = self.config.M if len(self.graphs) > 1 else self.config.M * 2
            cap = max(1000, self.nodes.capacity)
            self.layer_edges.append(np.zeros((cap, M_max), dtype=np.int32))
            self.layer_degrees.append(np.zeros(cap, dtype=np.int32))
            
        cap = self.nodes.capacity
        for lc in range(len(self.layer_edges)):
            if self.layer_edges[lc].shape[0] < cap:
                M_max = self.layer_edges[lc].shape[1]
                new_edges = np.zeros((cap, M_max), dtype=np.int32)
                new_edges[:self.layer_edges[lc].shape[0]] = self.layer_edges[lc]
                self.layer_edges[lc] = new_edges
                
                new_degrees = np.zeros(cap, dtype=np.int32)
                new_degrees[:self.layer_degrees[lc].shape[0]] = self.layer_degrees[lc]
                self.layer_degrees[lc] = new_degrees

    def _update_node_edges(self, lc: int, node_id: int):
        if lc >= len(self.layer_edges):
            return
        idx = self.nodes.node_id_to_index.get(node_id)
        if idx is None:
            return
        edges_set = self.graphs[lc].get(node_id, set())
        edges_indices = [self.nodes.node_id_to_index[n] for n in edges_set if n in self.nodes.node_id_to_index]
        deg = len(edges_indices)
        M_max = self.layer_edges[lc].shape[1]
        if deg > M_max:
            edges_indices = edges_indices[:M_max]
            deg = M_max
        if deg > 0:
            self.layer_edges[lc][idx, :deg] = edges_indices
        self.layer_degrees[lc][idx] = deg

    def _rebuild_layer_arrays(self):
        self.layer_edges = []
        self.layer_degrees = []
        cap = max(1000, self.nodes.capacity)
        for lc, level_graph in enumerate(self.graphs):
            M_max = self.config.M if lc > 0 else self.config.M * 2
            edges = np.zeros((cap, M_max), dtype=np.int32)
            degrees = np.zeros(cap, dtype=np.int32)
            for node_id, nbr_set in level_graph.items():
                idx = self.nodes.node_id_to_index.get(node_id)
                if idx is None:
                    continue
                nbr_indices = [self.nodes.node_id_to_index[n] for n in nbr_set if n in self.nodes.node_id_to_index]
                deg = len(nbr_indices)
                if deg > M_max:
                    nbr_indices = nbr_indices[:M_max]
                    deg = M_max
                if deg > 0:
                    edges[idx, :deg] = nbr_indices
                degrees[idx] = deg
            self.layer_edges.append(edges)
            self.layer_degrees.append(degrees)

    def save_snapshot(self, path: str = None):
        if path is None:
            if self.persist_dir is None:
                raise ValueError("Must provide path if persist_dir is not set")
            path = self.persist_dir
            
        if not os.path.exists(path):
            os.makedirs(path)
            
        if hasattr(self.nodes, "flush"):
            self.nodes.flush()
            
        dim = 0
        if isinstance(self.nodes, (VectorStore, ContiguousVectorStore)):
            dim = self.nodes.dim
        elif len(self.nodes) > 0:
            dim = len(next(iter(self.nodes.values())))
            
        metadata = {
            "config": {
                "M": self.config.M,
                "ef_construction": self.config.ef_construction,
                "ef_search": self.config.ef_search,
                "m_L": self.config.m_L,
                "heuristic": self.config.heuristic,
                "extend_candidates": self.config.extend_candidates,
                "keep_pruned_connections": self.config.keep_pruned_connections
            },
            "metric": self.metric,
            "max_level": self.max_level,
            "entry_point": self.entry_point,
            "dim": dim,
            "deleted_nodes": list(self.deleted_nodes)
        }
        
        if isinstance(self.nodes, (VectorStore, ContiguousVectorStore)):
            metadata["node_id_to_index"] = self.nodes.node_id_to_index
            metadata["size"] = self.nodes.size
            
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f)
            
        with open(os.path.join(path, "metadata_store.json"), "w") as f:
            json.dump(self.metadata_store, f)
            
        with open(os.path.join(path, "graph.bin"), "wb") as f:
            f.write(struct.pack('<I', len(self.graphs)))
            for level_graph in self.graphs:
                f.write(struct.pack('<I', len(level_graph)))
                for node_id, edges in level_graph.items():
                    f.write(struct.pack('<I', node_id))
                    f.write(struct.pack('<I', len(edges)))
                    if edges:
                        f.write(struct.pack(f'<{len(edges)}I', *edges))
                        
        if self.wal:
            self.wal.truncate()

    @classmethod
    def load_snapshot(cls, path: str, persist_dir: str = None):
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)
            
        config = HNSWConfig(**metadata["config"])
        idx = cls(config=config, metric=metadata["metric"], persist_dir=persist_dir)
        idx.max_level = metadata["max_level"]
        idx.entry_point = metadata["entry_point"]
        dim = metadata["dim"]
        idx.deleted_nodes = set(metadata.get("deleted_nodes", []))
        
        meta_store_path = os.path.join(path, "metadata_store.json")
        if os.path.exists(meta_store_path):
            with open(meta_store_path, "r") as f:
                raw_store = json.load(f)
                idx.metadata_store = {int(k): v for k, v in raw_store.items()}
        
        if persist_dir and isinstance(idx.nodes, VectorStore) and dim > 0:
            idx.nodes.dim = dim
            idx.nodes.node_id_to_index = {int(k): v for k, v in metadata.get("node_id_to_index", {}).items()}
            idx.nodes.index_to_node_id = {v: k for k, v in idx.nodes.node_id_to_index.items()}
            idx.nodes.size = metadata.get("size", 0)
            idx.nodes._init_memmap(dim)
        elif isinstance(idx.nodes, ContiguousVectorStore) and dim > 0:
            idx.nodes.dim = dim
            idx.nodes.node_id_to_index = {int(k): v for k, v in metadata.get("node_id_to_index", {}).items()}
            idx.nodes.index_to_node_id = {v: k for k, v in idx.nodes.node_id_to_index.items()}
            idx.nodes.size = metadata.get("size", 0)
            idx.nodes._init_vectors(dim)
            
        idx.graphs = []
        graph_path = os.path.join(path, "graph.bin")
        if os.path.exists(graph_path):
            with open(graph_path, "rb") as f:
                num_levels_buf = f.read(4)
                if num_levels_buf:
                    num_levels = struct.unpack('<I', num_levels_buf)[0]
                    for _ in range(num_levels):
                        level_graph = {}
                        num_nodes_buf = f.read(4)
                        if not num_nodes_buf: break
                        num_nodes = struct.unpack('<I', num_nodes_buf)[0]
                        for _ in range(num_nodes):
                            node_id = struct.unpack('<I', f.read(4))[0]
                            num_edges = struct.unpack('<I', f.read(4))[0]
                            if num_edges > 0:
                                edges = struct.unpack(f'<{num_edges}I', f.read(4 * num_edges))
                                level_graph[node_id] = set(edges)
                            else:
                                level_graph[node_id] = set()
                        idx.graphs.append(level_graph)
                        
        idx._rebuild_layer_arrays()
        
        if idx.wal and dim > 0:
            idx.wal.replay(
                insert_fn=lambda n, v, m: idx._replay_insert(n, v, m),
                delete_fn=lambda n: idx._replay_delete(n),
                dim=dim
            )
            
        return idx

    def _replay_insert(self, node_id: int, vector: np.ndarray, metadata: dict):
        old_wal = self.wal
        self.wal = None
        try:
            self.insert(node_id, vector, metadata)
        finally:
            self.wal = old_wal

    def _replay_delete(self, node_id: int):
        old_wal = self.wal
        self.wal = None
        try:
            self.delete(node_id)
        finally:
            self.wal = old_wal

    def _batch_distance(self, query: np.ndarray, node_ids: List[int]) -> np.ndarray:
        if not node_ids:
            return np.array([], dtype=np.float32)
        vecs = np.array([self.nodes[n] for n in node_ids], dtype=np.float32)
        if self.metric == "l2":
            return np.linalg.norm(vecs - query, axis=1)
        elif self.metric == "cosine":
            dots = np.dot(vecs, query)
            query_norm = np.linalg.norm(query)
            if query_norm == 0:
                query_norm = 1.0
            norms = np.linalg.norm(vecs, axis=1) * query_norm
            norms[norms == 0] = 1.0
            return 1.0 - (dots / norms)

    def _get_random_level(self) -> int:
        return math.floor(-math.log(self.rng.uniform(0.0, 1.0)) * self.config.m_L)

    def _insert_unlocked(self, node_id: int, vector: np.ndarray, metadata: dict = None):
        if node_id in self.nodes and node_id not in self.deleted_nodes:
            raise ValueError(f"Node {node_id} already exists in the index.")
            
        if self.wal:
            self.wal.log_insert(node_id, vector, metadata)
        
        vec_array = np.ascontiguousarray(vector, dtype=np.float32)
        self.nodes[node_id] = vec_array
        if node_id in self.deleted_nodes:
            self.deleted_nodes.remove(node_id)
        if metadata:
            self.metadata_store[node_id] = metadata
        
        if self.entry_point is None:
            self.entry_point = node_id
            self.max_level = self._get_random_level()
            self._ensure_graph_capacity(self.max_level)
            for lc in range(self.max_level + 1):
                self.graphs[lc][node_id] = set()
                self._update_node_edges(lc, node_id)
            return
        
        level = self._get_random_level()
        self._ensure_graph_capacity(max(self.max_level, level))
        
        curr_obj = self.entry_point
        curr_dist = self.distance_fn(vec_array, self.nodes[curr_obj])
        
        # Greedy descent through upper layers
        if self.use_native and len(self.layer_edges) > 0 and self.max_level > level:
            vec_matrix = self.nodes.get_matrix()
            curr_idx = self.nodes.node_id_to_index[curr_obj]
            for lc in range(self.max_level, level, -1):
                curr_idx, curr_dist = greedy_search_layer_native(
                    vec_array,
                    curr_idx,
                    curr_dist,
                    vec_matrix,
                    self.layer_edges[lc],
                    self.layer_degrees[lc],
                    self.metric_code
                )
            curr_obj = self.nodes.index_to_node_id[curr_idx]
        else:
            for lc in range(self.max_level, level, -1):
                changed = True
                while changed:
                    changed = False
                    neighbors = list(self.graphs[lc].get(curr_obj, set()))
                    if not neighbors:
                        continue
                        
                    dists = self._batch_distance(vec_array, neighbors)
                    min_idx = np.argmin(dists)
                    if dists[min_idx] < curr_dist:
                        curr_dist = float(dists[min_idx])
                        curr_obj = neighbors[min_idx]
                        changed = True
        
        entry_points = [curr_obj]
        for lc in range(min(self.max_level, level), -1, -1):
            if node_id not in self.graphs[lc]:
                self.graphs[lc][node_id] = set()
                
            W = self._search_layer(vec_array, entry_points, self.config.ef_construction, lc)
            
            M_max = self.config.M if lc > 0 else self.config.M * 2
            neighbors = self._select_neighbors(vec_array, W, self.config.M)
            
            for n_id, n_dist in neighbors:
                self.graphs[lc][node_id].add(n_id)
                if n_id not in self.graphs[lc]:
                    self.graphs[lc][n_id] = set()
                self.graphs[lc][n_id].add(node_id)
                self._update_node_edges(lc, n_id)
                
                if len(self.graphs[lc][n_id]) > M_max:
                    n_neighbors = [(v, self.distance_fn(self.nodes[n_id], self.nodes[v])) 
                                   for v in self.graphs[lc][n_id]]
                    n_neighbors_pruned = self._select_neighbors(self.nodes[n_id], n_neighbors, M_max)
                    self.graphs[lc][n_id] = {v for v, _ in n_neighbors_pruned}
                    self._update_node_edges(lc, n_id)
            
            self._update_node_edges(lc, node_id)
            entry_points = [n_id for n_id, _ in W]

        if level > self.max_level:
            for lc in range(self.max_level + 1, level + 1):
                self.graphs[lc][node_id] = set()
                self._update_node_edges(lc, node_id)
            self.max_level = level
            self.entry_point = node_id

    def insert(self, node_id: int, vector: np.ndarray, metadata: dict = None):
        with self.lock.write():
            self._insert_unlocked(node_id, vector, metadata)

    def batch_insert(self, vectors: np.ndarray, node_ids: List[int] = None, metadatas: List[dict] = None):
        """
        Batched vector insertion to amortize lock overhead and WAL synchronization.
        """
        n_vecs = len(vectors)
        if n_vecs == 0:
            return
            
        if node_ids is None:
            start_id = len(self.nodes)
            node_ids = list(range(start_id, start_id + n_vecs))
            
        with self.lock.write():
            for i in range(n_vecs):
                meta = metadatas[i] if metadatas and i < len(metadatas) else None
                self._insert_unlocked(node_ids[i], vectors[i], meta)
                
            if self.wal:
                self.wal.flush()

    def _matches_filter(self, node_id: int, filter_dict: dict) -> bool:
        if not filter_dict:
            return True
        meta = self.metadata_store.get(node_id, {})
        for k, v in filter_dict.items():
            if meta.get(k) != v:
                return False
        return True

    def _search_unlocked(self, query: np.ndarray, k: int, ef: int = None, filter_dict: dict = None, return_visited: bool = False, use_native: bool = None):
        if self.entry_point is None:
            return ([], 0) if return_visited else []
        
        if ef is None:
            ef = self.config.ef_search
            
        ef = max(ef, k)
        active_use_native = self.use_native if use_native is None else (use_native and HAS_NATIVE)
            
        query_c = np.ascontiguousarray(query, dtype=np.float32)
        curr_obj = self.entry_point
        curr_dist = self.distance_fn(query_c, self.nodes[curr_obj])
        
        # Greedy search in upper layers
        if active_use_native and len(self.layer_edges) > 1 and self.max_level > 0:
            vec_matrix = self.nodes.get_matrix()
            curr_idx = self.nodes.node_id_to_index[curr_obj]
            for lc in range(self.max_level, 0, -1):
                curr_idx, curr_dist = greedy_search_layer_native(
                    query_c,
                    curr_idx,
                    curr_dist,
                    vec_matrix,
                    self.layer_edges[lc],
                    self.layer_degrees[lc],
                    self.metric_code
                )
            curr_obj = self.nodes.index_to_node_id[curr_idx]
        else:
            for lc in range(self.max_level, 0, -1):
                changed = True
                while changed:
                    changed = False
                    neighbors = list(self.graphs[lc].get(curr_obj, set()))
                    if not neighbors:
                        continue
                        
                    dists = self._batch_distance(query_c, neighbors)
                    min_idx = np.argmin(dists)
                    if dists[min_idx] < curr_dist:
                        curr_dist = float(dists[min_idx])
                        curr_obj = neighbors[min_idx]
                        changed = True
                            
        if return_visited:
            W, visited_count = self._search_layer(query_c, [curr_obj], ef, 0, return_visited=True, use_native=active_use_native)
            self.last_visited_count = visited_count
        else:
            W = self._search_layer(query_c, [curr_obj], ef, 0, return_visited=False, use_native=active_use_native)
            self.last_visited_count = 0
            visited_count = 0

        W.sort(key=lambda x: x[1])
        
        results = []
        for n_id, dist in W:
            if n_id in self.deleted_nodes:
                continue
            if self._matches_filter(n_id, filter_dict):
                meta = self.metadata_store.get(n_id, {})
                results.append((n_id, dist, meta))
                if len(results) == k:
                    break
                        
        if return_visited:
            return results, visited_count
        return results

    def search(self, query: np.ndarray, k: int, ef: int = None, filter_dict: dict = None, return_visited: bool = False, use_native: bool = None):
        with self.lock.read():
            return self._search_unlocked(query, k=k, ef=ef, filter_dict=filter_dict, return_visited=return_visited, use_native=use_native)

    def batch_search(
        self,
        queries: np.ndarray,
        k: int = 10,
        ef: int = None,
        filter_dict: dict = None,
        num_threads: int = None,
        return_visited: bool = False,
        use_native: bool = None
    ):
        """
        Batched search across multiple query vectors.
        Releases GIL in native hot path to execute searches concurrently across threads.
        """
        with self.lock.read():
            if self.entry_point is None or len(queries) == 0:
                return [([], 0) if return_visited else [] for _ in range(len(queries))]
                
            queries_arr = np.ascontiguousarray(queries, dtype=np.float32)
            n_queries = len(queries_arr)
            
            if n_queries == 1 or num_threads == 1:
                return [self._search_unlocked(queries_arr[i], k=k, ef=ef, filter_dict=filter_dict, return_visited=return_visited, use_native=use_native) for i in range(n_queries)]
                
            workers = num_threads if num_threads is not None else min(32, os.cpu_count() or 4)
            chunk_size = max(1, n_queries // (workers * 4))
            chunks = [queries_arr[i:i+chunk_size] for i in range(0, n_queries, chunk_size)]
            
            def _search_chunk(chunk):
                return [self._search_unlocked(q, k=k, ef=ef, filter_dict=filter_dict, return_visited=return_visited, use_native=use_native) for q in chunk]
                
            with ThreadPoolExecutor(max_workers=workers) as executor:
                nested = list(executor.map(_search_chunk, chunks))
                return [item for sublist in nested for item in sublist]

    def _search_layer(self, query: np.ndarray, entry_points: List[int], ef: int, layer: int, return_visited: bool = False, use_native: bool = None):
        active_use_native = self.use_native if use_native is None else (use_native and HAS_NATIVE)
        if active_use_native and len(self.layer_edges) > layer:
            vec_matrix = self.nodes.get_matrix()
            if len(vec_matrix) > 0 and len(entry_points) > 0:
                ep_indices = np.array(
                    [self.nodes.node_id_to_index[ep] for ep in entry_points if ep in self.nodes.node_id_to_index],
                    dtype=np.int32
                )
                if len(ep_indices) > 0:
                    query_c = np.ascontiguousarray(query, dtype=np.float32)
                    if return_visited:
                        raw_res, visited_count = search_layer_native(
                            query_c,
                            ep_indices,
                            ef,
                            vec_matrix,
                            self.layer_edges[layer],
                            self.layer_degrees[layer],
                            self.metric_code,
                            return_visited=True
                        )
                        idx_to_id = self.nodes.index_to_node_id
                        return [(idx_to_id[idx], dist) for idx, dist in raw_res if idx in idx_to_id], visited_count
                    else:
                        raw_res = search_layer_native(
                            query_c,
                            ep_indices,
                            ef,
                            vec_matrix,
                            self.layer_edges[layer],
                            self.layer_degrees[layer],
                            self.metric_code,
                            return_visited=False
                        )
                        idx_to_id = self.nodes.index_to_node_id
                        return [(idx_to_id[idx], dist) for idx, dist in raw_res if idx in idx_to_id]

        return self._search_layer_python(query, entry_points, ef, layer, return_visited=return_visited)

    def _search_layer_python(self, query: np.ndarray, entry_points: List[int], ef: int, layer: int, return_visited: bool = False):
        visited = set(entry_points)
        C = []
        W = []
        
        for ep in entry_points:
            d = self.distance_fn(query, self.nodes[ep])
            heapq.heappush(C, (d, ep))
            heapq.heappush(W, (-d, ep))
            
        while len(C) > 0:
            c_dist, c_id = heapq.heappop(C)
            
            f_dist_neg, f_id = W[0]
            f_dist = -f_dist_neg
            
            if c_dist > f_dist:
                break
                
            neighbors = self.graphs[layer].get(c_id, set())
            unvisited = [n for n in neighbors if n not in visited]
            if not unvisited:
                continue
                
            for n in unvisited:
                visited.add(n)
                
            dists = self._batch_distance(query, unvisited)
            
            for i, neighbor in enumerate(unvisited):
                f_dist_neg, _ = W[0]
                f_dist = -f_dist_neg
                
                d = float(dists[i])
                
                if len(W) < ef or d < f_dist:
                    heapq.heappush(C, (d, neighbor))
                    heapq.heappush(W, (-d, neighbor))
                    
                    if len(W) > ef:
                        heapq.heappop(W)
                            
        res = [(n_id, -d_neg) for d_neg, n_id in W]
        if return_visited:
            return res, len(visited)
        return res

    def _select_neighbors(self, base_vector: np.ndarray, candidates: List[Tuple[int, float]], M: int) -> List[Tuple[int, float]]:
        if not self.config.heuristic:
            candidates_sorted = sorted(candidates, key=lambda x: x[1])
            return candidates_sorted[:M]
            
        # Diversity-aware heuristic (paper Algorithm 4)
        if self.use_native:
            vec_matrix = self.nodes.get_matrix()
            cand_indices = [
                (self.nodes.node_id_to_index[c_id], dist)
                for c_id, dist in candidates if c_id in self.nodes.node_id_to_index
            ]
            base_c = np.ascontiguousarray(base_vector, dtype=np.float32)
            raw_res = select_neighbors_heuristic_native(
                base_c,
                cand_indices,
                M,
                vec_matrix,
                self.metric_code,
                self.config.keep_pruned_connections
            )
            idx_to_id = self.nodes.index_to_node_id
            return [(idx_to_id[idx], dist) for idx, dist in raw_res if idx in idx_to_id]

        return self._select_neighbors_heuristic_python(base_vector, candidates, M)

    def _select_neighbors_heuristic_python(self, base_vector: np.ndarray, candidates: List[Tuple[int, float]], M: int) -> List[Tuple[int, float]]:
        candidates_sorted = sorted(candidates, key=lambda x: x[1])
        R: List[Tuple[int, float]] = []
        W_d: List[Tuple[int, float]] = []
        
        for cand in candidates_sorted:
            if len(R) >= M:
                break
            e_id, dist_e_q = cand
            e_vec = self.nodes[e_id]
            is_diverse = True
            for r_id, dist_r_q in R:
                r_vec = self.nodes[r_id]
                dist_e_r = self.distance_fn(e_vec, r_vec)
                if dist_e_r < dist_e_q:
                    is_diverse = False
                    break
            if is_diverse:
                R.append(cand)
            else:
                W_d.append(cand)
                
        if self.config.keep_pruned_connections and len(R) < M:
            for cand in W_d:
                if len(R) >= M:
                    break
                R.append(cand)
                
        return R

    def delete(self, node_id: int):
        with self.lock.write():
            if node_id not in self.nodes or node_id in self.deleted_nodes:
                return
                
            if self.wal:
                self.wal.log_delete(node_id)
                
            if node_id in self.metadata_store:
                del self.metadata_store[node_id]
                
            self.deleted_nodes.add(node_id)

    def compact(self):
        """
        Physically removes tombstoned (deleted) vectors from the store, 
        rebuilds affected graph regions, and reclaims space.
        """
        with self.lock.write():
            if not self.deleted_nodes:
                return

            if isinstance(self.nodes, VectorStore):
                new_filepath = self.nodes.filepath + ".compact"
                new_store = VectorStore(new_filepath, dtype=self.nodes.dtype, initial_capacity=self.nodes.capacity)
                new_store._init_memmap(self.nodes.dim)
                
                for node_id, idx in self.nodes.node_id_to_index.items():
                    if node_id not in self.deleted_nodes:
                        new_store[node_id] = self.nodes.mmap[idx]
                        
                self.nodes.close()
                new_store.close()
                if os.path.exists(self.nodes.filepath):
                    os.remove(self.nodes.filepath)
                os.rename(new_filepath, self.nodes.filepath)
                new_store.filepath = self.nodes.filepath
                new_store._init_memmap(self.nodes.dim)
                self.nodes = new_store
            elif isinstance(self.nodes, ContiguousVectorStore):
                new_store = ContiguousVectorStore(initial_capacity=self.nodes.capacity, dtype=self.nodes.dtype)
                new_store._init_vectors(self.nodes.dim)
                for node_id, idx in self.nodes.node_id_to_index.items():
                    if node_id not in self.deleted_nodes:
                        new_store[node_id] = self.nodes.vectors[idx]
                self.nodes = new_store
            else:
                for node_id in self.deleted_nodes:
                    if node_id in self.nodes:
                        del self.nodes[node_id]

            # Heal graph by removing deleted nodes
            for lc in range(len(self.graphs)):
                for del_id in self.deleted_nodes:
                    if del_id in self.graphs[lc]:
                        del self.graphs[lc][del_id]
                
                for node_id, edges in self.graphs[lc].items():
                    self.graphs[lc][node_id] = {e for e in edges if e not in self.deleted_nodes}

            # Update entry_point if current one was deleted
            if self.entry_point in self.deleted_nodes:
                new_ep = None
                for lc in range(self.max_level, -1, -1):
                    valid_nodes = [n for n in self.graphs[lc] if n not in self.deleted_nodes]
                    if valid_nodes:
                        new_ep = valid_nodes[0]
                        self.max_level = lc
                        break
                self.entry_point = new_ep

            self.deleted_nodes.clear()
            self._rebuild_layer_arrays()

            if self.wal:
                self.wal.truncate()

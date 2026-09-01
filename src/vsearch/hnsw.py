import math
import random
import heapq
import os
import json
import struct
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Callable, Optional
import numpy as np

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

    def __post_init__(self):
        if self.m_L is None:
            self.m_L = 1.0 / math.log(self.M)


class VectorStore:
    def __init__(self, filepath: str, dtype=np.float32, initial_capacity: int = 1000):
        self.filepath = filepath
        self.dtype = dtype
        self.initial_capacity = initial_capacity
        self.node_id_to_index: Dict[int, int] = {}
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
            # close memmap properly if possible, else garbage collection handles it
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
            self.size += 1
        else:
            idx = self.node_id_to_index[node_id]
        self.mmap[idx] = vector

    def __getitem__(self, node_id: int) -> np.ndarray:
        return self.mmap[self.node_id_to_index[node_id]]

    def __delitem__(self, node_id: int):
        if node_id in self.node_id_to_index:
            del self.node_id_to_index[node_id]

    def __contains__(self, node_id: int) -> bool:
        return node_id in self.node_id_to_index
        
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
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = None
        self._open()

    def _open(self):
        if self.file is None:
            self.file = open(self.filepath, "ab")

    def log_insert(self, node_id: int, vector: np.ndarray):
        # Format: Opcode(1 byte), NodeID(4 bytes)
        header = struct.pack('<BI', 1, node_id)
        vec_bytes = vector.astype(np.float32).tobytes()
        self.file.write(header + vec_bytes)
        self.file.flush()

    def log_delete(self, node_id: int):
        header = struct.pack('<BI', 2, node_id)
        self.file.write(header)
        self.file.flush()

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
                if op == 1:
                    id_buf = f.read(4)
                    node_id = struct.unpack('<I', id_buf)[0]
                    vec_buf = f.read(bytes_per_vec)
                    vector = np.frombuffer(vec_buf, dtype=np.float32).copy()
                    insert_fn(node_id, vector)
                elif op == 2:
                    id_buf = f.read(4)
                    node_id = struct.unpack('<I', id_buf)[0]
                    delete_fn(node_id)
                else:
                    raise ValueError(f"Corrupt WAL: unknown op {op}")


class HNSWIndex:
    def __init__(self, config: HNSWConfig = HNSWConfig(), metric: str = "l2", persist_dir: str = None):
        self.config = config
        self.metric = metric
        if metric == "l2":
            self.distance_fn = l2_distance
        elif metric == "cosine":
            self.distance_fn = cosine_distance
        else:
            raise ValueError("metric must be 'l2' or 'cosine'")
        
        self.persist_dir = persist_dir
        self.max_level = -1
        self.entry_point = None
        
        if self.persist_dir:
            if not os.path.exists(self.persist_dir):
                os.makedirs(self.persist_dir)
            self.nodes = VectorStore(os.path.join(self.persist_dir, "vectors.bin"))
            self.wal = WALManager(os.path.join(self.persist_dir, "wal.log"))
        else:
            self.nodes: Dict[int, np.ndarray] = {}
            self.wal = None
            
        # graphs is a list of dicts, one for each level. 
        # graphs[level][node_id] = set of neighbor node_ids
        self.graphs: List[Dict[int, Set[int]]] = []

    def save_snapshot(self, path: str = None):
        if path is None:
            if self.persist_dir is None:
                raise ValueError("Must provide path if persist_dir is not set")
            path = self.persist_dir
            
        if not os.path.exists(path):
            os.makedirs(path)
            
        # Flush vectors if using VectorStore
        if hasattr(self.nodes, "flush"):
            self.nodes.flush()
            
        dim = 0
        if isinstance(self.nodes, VectorStore):
            dim = self.nodes.dim
        elif len(self.nodes) > 0:
            dim = len(next(iter(self.nodes.values())))
            
        metadata = {
            "config": {
                "M": self.config.M,
                "ef_construction": self.config.ef_construction,
                "ef_search": self.config.ef_search,
                "m_L": self.config.m_L
            },
            "metric": self.metric,
            "max_level": self.max_level,
            "entry_point": self.entry_point,
            "dim": dim
        }
        
        if isinstance(self.nodes, VectorStore):
            metadata["node_id_to_index"] = self.nodes.node_id_to_index
            metadata["size"] = self.nodes.size
            
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f)
            
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
        
        if persist_dir and isinstance(idx.nodes, VectorStore) and dim > 0:
            idx.nodes.dim = dim
            idx.nodes.node_id_to_index = {int(k): v for k, v in metadata.get("node_id_to_index", {}).items()}
            idx.nodes.size = metadata.get("size", 0)
            idx.nodes._init_memmap(dim)
            
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
                        
        if idx.wal and dim > 0:
            idx.wal.replay(
                insert_fn=lambda n, v: idx._replay_insert(n, v),
                delete_fn=lambda n: idx._replay_delete(n),
                dim=dim
            )
            
        return idx

    def _replay_insert(self, node_id: int, vector: np.ndarray):
        old_wal = self.wal
        self.wal = None
        try:
            self.insert(node_id, vector)
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
            return np.array([])
        # Use a list comprehension to gather vectors, then stack
        # This is typically faster than vstack
        vecs = np.array([self.nodes[n] for n in node_ids])
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
        return math.floor(-math.log(random.uniform(0.0, 1.0)) * self.config.m_L)

    def insert(self, node_id: int, vector: np.ndarray):
        """
        Inserts a new vector into the HNSW graph.
        """
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id} already exists in the index.")
            
        if self.wal:
            self.wal.log_insert(node_id, vector)
        
        self.nodes[node_id] = vector
        
        if self.entry_point is None:
            self.entry_point = node_id
            self.max_level = self._get_random_level()
            for _ in range(self.max_level + 1):
                self.graphs.append({node_id: set()})
            return
        
        level = self._get_random_level()
        
        while len(self.graphs) <= level:
            self.graphs.append({})
            
        curr_obj = self.entry_point
        curr_dist = self.distance_fn(vector, self.nodes[curr_obj])
        
        for lc in range(self.max_level, level, -1):
            changed = True
            while changed:
                changed = False
                neighbors = list(self.graphs[lc].get(curr_obj, set()))
                if not neighbors:
                    continue
                    
                dists = self._batch_distance(vector, neighbors)
                min_idx = np.argmin(dists)
                if dists[min_idx] < curr_dist:
                    curr_dist = float(dists[min_idx])
                    curr_obj = neighbors[min_idx]
                    changed = True
        
        entry_points = [curr_obj]
        for lc in range(min(self.max_level, level), -1, -1):
            if node_id not in self.graphs[lc]:
                self.graphs[lc][node_id] = set()
                
            W = self._search_layer(vector, entry_points, self.config.ef_construction, lc)
            
            M_max = self.config.M if lc > 0 else self.config.M * 2
            neighbors = self._select_neighbors(W, self.config.M)
            
            for n_id, n_dist in neighbors:
                self.graphs[lc][node_id].add(n_id)
                if n_id not in self.graphs[lc]:
                    self.graphs[lc][n_id] = set()
                self.graphs[lc][n_id].add(node_id)
                
                if len(self.graphs[lc][n_id]) > M_max:
                    n_neighbors = [(v, self.distance_fn(self.nodes[n_id], self.nodes[v])) 
                                   for v in self.graphs[lc][n_id]]
                    n_neighbors_pruned = self._select_neighbors(n_neighbors, M_max)
                    self.graphs[lc][n_id] = {v for v, _ in n_neighbors_pruned}
            
            entry_points = [n_id for n_id, _ in W]

        if level > self.max_level:
            for lc in range(self.max_level + 1, level + 1):
                self.graphs[lc][node_id] = set()
            self.max_level = level
            self.entry_point = node_id

    def search(self, query: np.ndarray, k: int, ef: int = None) -> List[Tuple[int, float]]:
        if self.entry_point is None:
            return []
        
        if ef is None:
            ef = self.config.ef_search
            
        ef = max(ef, k)
            
        curr_obj = self.entry_point
        curr_dist = self.distance_fn(query, self.nodes[curr_obj])
        
        for lc in range(self.max_level, 0, -1):
            changed = True
            while changed:
                changed = False
                neighbors = list(self.graphs[lc].get(curr_obj, set()))
                if not neighbors:
                    continue
                    
                dists = self._batch_distance(query, neighbors)
                min_idx = np.argmin(dists)
                if dists[min_idx] < curr_dist:
                    curr_dist = float(dists[min_idx])
                    curr_obj = neighbors[min_idx]
                    changed = True
                        
        W = self._search_layer(query, [curr_obj], ef, 0)
        
        W.sort(key=lambda x: x[1])
        return W[:k]

    def _search_layer(self, query: np.ndarray, entry_points: List[int], ef: int, layer: int) -> List[Tuple[int, float]]:
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
                            
        return [(n_id, -d_neg) for d_neg, n_id in W]

    def _select_neighbors(self, candidates: List[Tuple[int, float]], M: int) -> List[Tuple[int, float]]:
        candidates_sorted = sorted(candidates, key=lambda x: x[1])
        return candidates_sorted[:M]

    def delete(self, node_id: int):
        if node_id not in self.nodes:
            return
            
        if self.wal:
            self.wal.log_delete(node_id)
            
        for lc in range(len(self.graphs)):
            if node_id in self.graphs[lc]:
                neighbors = list(self.graphs[lc][node_id])
                
                for other_node, other_neighbors in self.graphs[lc].items():
                    if node_id in other_neighbors:
                        other_neighbors.remove(node_id)
                
                del self.graphs[lc][node_id]
                
                M_max = self.config.M if lc > 0 else self.config.M * 2
                for n_id in neighbors:
                    current_connections = list(self.graphs[lc][n_id])
                    candidates = [(v, self.distance_fn(self.nodes[n_id], self.nodes[v])) 
                                  for v in current_connections]
                    
                    for other_n in neighbors:
                        if other_n != n_id and other_n not in current_connections:
                            candidates.append((other_n, self.distance_fn(self.nodes[n_id], self.nodes[other_n])))
                            
                    pruned = self._select_neighbors(candidates, M_max)
                    self.graphs[lc][n_id] = {v for v, _ in pruned}

        if self.entry_point == node_id:
            new_entry_point = None
            for lc in range(self.max_level, -1, -1):
                if len(self.graphs[lc]) > 0:
                    new_entry_point = next(iter(self.graphs[lc].keys()))
                    self.max_level = lc
                    break
            self.entry_point = new_entry_point
            
            if new_entry_point is None:
                self.max_level = -1
                self.graphs = []
                
        while len(self.graphs) > 0 and len(self.graphs[-1]) == 0:
            self.graphs.pop()
            self.max_level -= 1
            
        del self.nodes[node_id]

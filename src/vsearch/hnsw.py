import math
import random
import heapq
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Callable
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

class HNSWIndex:
    def __init__(self, config: HNSWConfig = HNSWConfig(), metric: str = "l2"):
        self.config = config
        self.metric = metric
        if metric == "l2":
            self.distance_fn = l2_distance
        elif metric == "cosine":
            self.distance_fn = cosine_distance
        else:
            raise ValueError("metric must be 'l2' or 'cosine'")
        
        self.max_level = -1
        self.entry_point = None
        self.nodes: Dict[int, np.ndarray] = {}
        # graphs is a list of dicts, one for each level. 
        # graphs[level][node_id] = set of neighbor node_ids
        self.graphs: List[Dict[int, Set[int]]] = []

    def _get_random_level(self) -> int:
        return math.floor(-math.log(random.uniform(0.0, 1.0)) * self.config.m_L)

    def insert(self, node_id: int, vector: np.ndarray):
        """
        Inserts a new vector into the HNSW graph.
        Corresponds to Algorithm 1 (INSERT) in Malkov & Yashunin (2016).
        """
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id} already exists in the index.")
        
        self.nodes[node_id] = vector
        
        if self.entry_point is None:
            # First node in the index
            self.entry_point = node_id
            self.max_level = self._get_random_level()
            for _ in range(self.max_level + 1):
                self.graphs.append({node_id: set()})
            return
        
        level = self._get_random_level()
        
        # Ensure we have enough layers in our graph structure
        while len(self.graphs) <= level:
            self.graphs.append({})
            
        curr_obj = self.entry_point
        curr_dist = self.distance_fn(vector, self.nodes[curr_obj])
        
        # Phase 1: Greedily descend from the top layer to level + 1 (ef = 1)
        # We only do this if the new node's level is less than the current max_level
        for lc in range(self.max_level, level, -1):
            changed = True
            while changed:
                changed = False
                neighbors = self.graphs[lc].get(curr_obj, set())
                for neighbor in neighbors:
                    d = self.distance_fn(vector, self.nodes[neighbor])
                    if d < curr_dist:
                        curr_dist = d
                        curr_obj = neighbor
                        changed = True
        
        # Phase 2: From min(max_level, level) down to 0, do ef-based search and connect
        entry_points = [curr_obj]
        for lc in range(min(self.max_level, level), -1, -1):
            # Init empty graph for this level if the node doesn't exist yet
            if node_id not in self.graphs[lc]:
                self.graphs[lc][node_id] = set()
                
            # Search layer to find candidates
            W = self._search_layer(vector, entry_points, self.config.ef_construction, lc)
            
            # Select neighbors
            M_max = self.config.M if lc > 0 else self.config.M * 2  # Often M_max0 = 2*M
            neighbors = self._select_neighbors(W, self.config.M)
            
            # Add bidirectional edges
            for n_id, n_dist in neighbors:
                self.graphs[lc][node_id].add(n_id)
                if n_id not in self.graphs[lc]:
                    self.graphs[lc][n_id] = set()
                self.graphs[lc][n_id].add(node_id)
                
                # Prune connections of the neighbor if they exceed M_max
                if len(self.graphs[lc][n_id]) > M_max:
                    n_neighbors = [(v, self.distance_fn(self.nodes[n_id], self.nodes[v])) 
                                   for v in self.graphs[lc][n_id]]
                    n_neighbors_pruned = self._select_neighbors(n_neighbors, M_max)
                    self.graphs[lc][n_id] = {v for v, _ in n_neighbors_pruned}
            
            # The entry points for the next layer down are the nodes in W
            entry_points = [n_id for n_id, _ in W]

        # Update entry point if the new node has a higher level
        if level > self.max_level:
            for lc in range(self.max_level + 1, level + 1):
                self.graphs[lc][node_id] = set()
            self.max_level = level
            self.entry_point = node_id

    def search(self, query: np.ndarray, k: int, ef: int = None) -> List[Tuple[int, float]]:
        """
        Searches for the k nearest neighbors of the query vector.
        Corresponds to Algorithm 5 (K-NN-SEARCH) in Malkov & Yashunin (2016).
        """
        if self.entry_point is None:
            return []
        
        if ef is None:
            ef = self.config.ef_search
            
        ef = max(ef, k)
            
        curr_obj = self.entry_point
        curr_dist = self.distance_fn(query, self.nodes[curr_obj])
        
        # Greedily descend to layer 1
        for lc in range(self.max_level, 0, -1):
            changed = True
            while changed:
                changed = False
                neighbors = self.graphs[lc].get(curr_obj, set())
                for neighbor in neighbors:
                    d = self.distance_fn(query, self.nodes[neighbor])
                    if d < curr_dist:
                        curr_dist = d
                        curr_obj = neighbor
                        changed = True
                        
        # Full search at layer 0
        W = self._search_layer(query, [curr_obj], ef, 0)
        
        # W contains tuples of (node_id, distance). 
        # We want the k nearest, so we can sort them by distance.
        W.sort(key=lambda x: x[1])
        return W[:k]

    def _search_layer(self, query: np.ndarray, entry_points: List[int], ef: int, layer: int) -> List[Tuple[int, float]]:
        """
        Greedy best-first search on a single layer.
        Corresponds to Algorithm 2 (SEARCH-LAYER) in Malkov & Yashunin (2016).
        Returns a list of (node_id, distance) for the closest found nodes.
        """
        visited = set(entry_points)
        
        # C is a min-heap of candidates to evaluate: (distance, node_id)
        C = []
        # W is a max-heap of the best neighbors found so far: (-distance, node_id)
        # (We use negative distance because Python's heapq is a min-heap)
        W = []
        
        for ep in entry_points:
            d = self.distance_fn(query, self.nodes[ep])
            heapq.heappush(C, (d, ep))
            heapq.heappush(W, (-d, ep))
            
        while len(C) > 0:
            # Extract closest candidate
            c_dist, c_id = heapq.heappop(C)
            
            # Farthest element in W
            f_dist_neg, f_id = W[0]
            f_dist = -f_dist_neg
            
            # If the closest candidate is further than the farthest result, we can stop
            if c_dist > f_dist:
                break
                
            # Evaluate neighbors of candidate
            neighbors = self.graphs[layer].get(c_id, set())
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    
                    f_dist_neg, f_id = W[0]
                    f_dist = -f_dist_neg
                    
                    d = self.distance_fn(query, self.nodes[neighbor])
                    
                    # If W is not full (len < ef) or neighbor is closer than the farthest element in W
                    if len(W) < ef or d < f_dist:
                        heapq.heappush(C, (d, neighbor))
                        heapq.heappush(W, (-d, neighbor))
                        
                        # Maintain W size to be at most ef
                        if len(W) > ef:
                            heapq.heappop(W)
                            
        # Convert W back to a regular list of (node_id, distance)
        return [(n_id, -d_neg) for d_neg, n_id in W]

    def _select_neighbors(self, candidates: List[Tuple[int, float]], M: int) -> List[Tuple[int, float]]:
        """
        Selects M neighbors from the candidate list.
        Corresponds to Algorithm 3 (SELECT-NEIGHBORS-SIMPLE).
        Phase 2 Upgrade: Implement Algorithm 4 (SELECT-NEIGHBORS-HEURISTIC) for diversity.
        """
        # Sort candidates by distance and take the top M
        candidates_sorted = sorted(candidates, key=lambda x: x[1])
        return candidates_sorted[:M]

    def delete(self, node_id: int):
        """
        Deletes a node from the graph and re-connects its former neighbors to 
        maintain the small-world property and connectivity.
        """
        if node_id not in self.nodes:
            return
            
        # For each layer, remove the node and heal the graph
        for lc in range(len(self.graphs)):
            if node_id in self.graphs[lc]:
                neighbors = list(self.graphs[lc][node_id])
                
                # Because edges can become directed due to pruning, we must remove 
                # node_id from ALL adjacency lists in this layer to prevent dangling references.
                for other_node, other_neighbors in self.graphs[lc].items():
                    if node_id in other_neighbors:
                        other_neighbors.remove(node_id)
                
                # Remove the node itself from this layer
                del self.graphs[lc][node_id]
                
                # Heal the graph: reconnect the former OUT-neighbors.
                # A simple strategy is to run a local selection for each neighbor
                # using the other neighbors as candidates, or simply pairwise connect them
                # up to their M limit.
                M_max = self.config.M if lc > 0 else self.config.M * 2
                for n_id in neighbors:
                    # Current connections + other former neighbors
                    current_connections = list(self.graphs[lc][n_id])
                    candidates = [(v, self.distance_fn(self.nodes[n_id], self.nodes[v])) 
                                  for v in current_connections]
                    
                    # Add other former neighbors of the deleted node as candidates
                    for other_n in neighbors:
                        if other_n != n_id and other_n not in current_connections:
                            candidates.append((other_n, self.distance_fn(self.nodes[n_id], self.nodes[other_n])))
                            
                    # Re-select the best M_max connections
                    pruned = self._select_neighbors(candidates, M_max)
                    self.graphs[lc][n_id] = {v for v, _ in pruned}
                    
                    # Note: this makes edges directed temporarily during the loop, 
                    # but since we do this for all neighbors, it largely balances out. 
                    # A more robust Phase 2 implementation would ensure bidirectional consistency.

        # If the deleted node was the entry point, we need to find a new one.
        if self.entry_point == node_id:
            # Find the highest level that still has nodes
            new_entry_point = None
            for lc in range(self.max_level, -1, -1):
                if len(self.graphs[lc]) > 0:
                    new_entry_point = next(iter(self.graphs[lc].keys()))
                    self.max_level = lc
                    break
            self.entry_point = new_entry_point
            
            if new_entry_point is None:
                # Graph is empty
                self.max_level = -1
                self.graphs = []
                
        # Remove empty top layers
        while len(self.graphs) > 0 and len(self.graphs[-1]) == 0:
            self.graphs.pop()
            self.max_level -= 1
            
        del self.nodes[node_id]

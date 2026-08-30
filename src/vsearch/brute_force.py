import numpy as np

def l2_distance_matrix(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Computes L2 distance between a query and a matrix of vectors."""
    # (x-y)^2 = x^2 + y^2 - 2xy
    # Using np.linalg.norm is simpler and exact.
    # vectors is of shape (N, D), query is (D,)
    diff = vectors - query
    return np.linalg.norm(diff, axis=1)

def cosine_distance_matrix(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Computes cosine distance (1 - cosine_similarity)."""
    query_norm = np.linalg.norm(query)
    vecs_norm = np.linalg.norm(vectors, axis=1)
    
    # Avoid division by zero
    query_norm = np.where(query_norm == 0, 1e-10, query_norm)
    vecs_norm = np.where(vecs_norm == 0, 1e-10, vecs_norm)
    
    dot_products = np.dot(vectors, query)
    cosine_sim = dot_products / (query_norm * vecs_norm)
    # Cosine distance is 1 - similarity
    return 1.0 - cosine_sim

class ExactKNN:
    def __init__(self, metric: str = "l2"):
        if metric not in ["l2", "cosine"]:
            raise ValueError("metric must be 'l2' or 'cosine'")
        self.metric = metric
        self.vectors = []
        self.ids = []

    def add(self, node_id: int, vector: np.ndarray):
        self.ids.append(node_id)
        self.vectors.append(vector)

    def search(self, query: np.ndarray, k: int):
        if not self.vectors:
            return []
        
        vecs_matrix = np.array(self.vectors)
        if self.metric == "l2":
            dists = l2_distance_matrix(query, vecs_matrix)
        else:
            dists = cosine_distance_matrix(query, vecs_matrix)
        
        # Get the indices of the top k smallest distances
        k_actual = min(k, len(self.vectors))
        # np.argpartition is faster than argsort for top k
        if k_actual < len(self.vectors):
            top_k_idx = np.argpartition(dists, k_actual - 1)[:k_actual]
            # sort the top k exactly
            top_k_dists = dists[top_k_idx]
            sorted_k_idx = top_k_idx[np.argsort(top_k_dists)]
        else:
            sorted_k_idx = np.argsort(dists)
            
        results = [(self.ids[i], float(dists[i])) for i in sorted_k_idx]
        return results

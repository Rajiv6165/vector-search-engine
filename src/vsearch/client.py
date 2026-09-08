import requests
from typing import List, Dict, Any, Optional

class VSearchClient:
    """
    A simple Python client for the Vector Search Engine API.
    """
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        
    def insert(self, node_id: int, vector: List[float], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Insert a vector into the index.
        """
        url = f"{self.base_url}/vectors"
        payload = {
            "node_id": node_id,
            "vector": vector,
        }
        if metadata is not None:
            payload["metadata"] = metadata
            
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()
        
    def delete(self, node_id: int) -> Dict[str, Any]:
        """
        Delete a vector by its node_id.
        """
        url = f"{self.base_url}/vectors/{node_id}"
        resp = requests.delete(url)
        resp.raise_for_status()
        return resp.json()
        
    def search(self, vector: List[float], k: int = 10, ef: int = 50, filter_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Search for the top-k nearest neighbors of a query vector.
        """
        url = f"{self.base_url}/search"
        payload = {
            "vector": vector,
            "k": k,
            "ef": ef
        }
        if filter_dict is not None:
            payload["filter"] = filter_dict
            
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()
        
    def stats(self) -> Dict[str, Any]:
        """
        Retrieve index statistics (size, dimension, config, etc).
        """
        url = f"{self.base_url}/stats"
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.json()

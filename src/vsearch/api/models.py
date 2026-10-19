from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class InsertRequest(BaseModel):
    node_id: int
    vector: List[float] = Field(..., description="The vector to insert.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata to store with the vector.")

class InsertResponse(BaseModel):
    node_id: int
    status: str

class BatchInsertRequest(BaseModel):
    vectors: List[List[float]] = Field(..., description="List of vectors to insert.")
    node_ids: Optional[List[int]] = Field(None, description="Optional explicit node IDs.")
    metadatas: Optional[List[Dict[str, Any]]] = Field(None, description="Optional metadata per vector.")

class BatchInsertResponse(BaseModel):
    inserted_count: int
    node_ids: List[int]
    status: str

class SearchRequest(BaseModel):
    vector: List[float] = Field(..., description="The query vector.")
    k: int = Field(10, gt=0, description="Number of nearest neighbors to return.")
    ef: Optional[int] = Field(None, gt=0, description="Optional ef override for search exploration.")
    filter: Optional[Dict[str, Any]] = Field(None, description="Optional exact-match metadata filter.")

class SearchResult(BaseModel):
    node_id: int
    distance: float
    metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    results: List[SearchResult]

class BatchSearchRequest(BaseModel):
    vectors: List[List[float]] = Field(..., description="List of query vectors.")
    k: int = Field(10, gt=0, description="Number of nearest neighbors to return per query.")
    ef: Optional[int] = Field(None, gt=0, description="Optional ef override for search exploration.")
    filter: Optional[Dict[str, Any]] = Field(None, description="Optional exact-match metadata filter.")
    num_threads: Optional[int] = Field(None, gt=0, description="Optional number of parallel threads.")

class BatchSearchResponse(BaseModel):
    results: List[List[SearchResult]]

class StatsResponse(BaseModel):
    size: int
    dim: int
    max_level: int
    entry_point: Optional[int]
    config: Dict[str, Any]

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, root_validator

class InsertRequest(BaseModel):
    node_id: int
    vector: List[float] = Field(..., description="The vector to insert.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata to store with the vector.")

class InsertResponse(BaseModel):
    node_id: int
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

class StatsResponse(BaseModel):
    size: int
    dim: int
    max_level: int
    entry_point: Optional[int]
    config: Dict[str, Any]

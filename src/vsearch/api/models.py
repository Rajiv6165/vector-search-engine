from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator

def _validate_filter_dict(f: Any) -> Optional[Dict[str, Any]]:
    if f is None:
        return None
    if not isinstance(f, dict):
        raise ValueError("Filter must be a dictionary")
    for k, v in f.items():
        if not isinstance(k, str) or not k.strip():
            raise ValueError("Filter keys must be non-empty strings")
        if not isinstance(v, (str, int, float, bool)):
            raise ValueError(f"Filter value for '{k}' must be a primitive scalar (str, int, float, bool), got {type(v).__name__}")
    return f

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

    @field_validator("filter")
    @classmethod
    def validate_filter(cls, v):
        return _validate_filter_dict(v)

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
    filter: Optional[Dict[str, Any]] = Field(None, description="Optional shared exact-match metadata filter.")
    filters: Optional[List[Optional[Dict[str, Any]]]] = Field(None, description="Optional per-query exact-match metadata filters.")
    num_threads: Optional[int] = Field(None, gt=0, description="Optional number of parallel threads.")

    @field_validator("filter")
    @classmethod
    def validate_filter(cls, v):
        return _validate_filter_dict(v)

    @field_validator("filters")
    @classmethod
    def validate_filters(cls, v):
        if v is None:
            return None
        return [_validate_filter_dict(item) for item in v]

class BatchSearchResponse(BaseModel):
    results: List[List[SearchResult]]

class StatsResponse(BaseModel):
    size: int
    dim: int
    max_level: int
    entry_point: Optional[int]
    config: Dict[str, Any]

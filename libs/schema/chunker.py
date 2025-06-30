from pydantic import BaseModel
from typing import List, Optional

class Chunk(BaseModel):
    chunk_id: str
    text: str
    start_page: Optional[int] = None
    end_page:   Optional[int] = None
    token_count: int

class ChunkerResult(BaseModel):
    chunks: List[Chunk]

    # summary fields
    num_chunks: int
    total_tokens: int
    embedding_model: Optional[str] = None

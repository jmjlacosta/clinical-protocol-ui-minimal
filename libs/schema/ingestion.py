from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class IngestionResult(BaseModel):
    text: str
    page_map: List[Dict] = Field(default_factory=list)
    engine: str
    confidence: float

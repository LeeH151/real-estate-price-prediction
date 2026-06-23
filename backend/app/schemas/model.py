from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class ModelItem(BaseModel):
    model_id: str
    name: str
    version: str
    status: str  # active / inactive
    accuracy: Optional[float] = None
    created_at: datetime


class ModelResponse(BaseModel):
    items: list[ModelItem]
    total: int
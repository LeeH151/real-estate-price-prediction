from pydantic import BaseModel
from typing import Optional


class PropertyCreate(BaseModel):
    title: Optional[str] = None
    district: str
    ward: str
    price: Optional[float] = None
    area: Optional[float] = None
    type: Optional[str] = "house"


class PropertyResponse(BaseModel):
    id: str
    title: str
    district: str
    ward: str
    price: Optional[float]
    area: Optional[float]
    type: Optional[str]

    class Config:
        from_attributes = True
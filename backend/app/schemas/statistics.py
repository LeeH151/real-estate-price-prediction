from pydantic import BaseModel
from typing import List


class HeatmapItem(BaseModel):
    district: str
    lat: float
    lng: float
    avg_price: float
    intensity: float


class TopDistrictItem(BaseModel):
    district: str
    avg_price: float


class PriceDistributionItem(BaseModel):
    range: str
    count: int


class HourlyItem(BaseModel):
    hour: str
    count: int
    is_peak: bool


class StatisticsResponse(BaseModel):
    heatmap_areas: List[HeatmapItem]
    top_districts: List[TopDistrictItem]
    price_distribution: List[PriceDistributionItem]
    hourly_predictions: List[HourlyItem]
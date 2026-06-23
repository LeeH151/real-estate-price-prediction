from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Literal
from enum import Enum


class QuarterEnum(str, Enum):
    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"


class MarketStats(BaseModel):
    avg_price: float = Field(..., ge=0)
    total_listings: int = Field(..., ge=0)
    prediction_count: int = Field(default=0, ge=0)

    model_config = ConfigDict(extra="forbid")


class HeatmapPoint(BaseModel):
    district: str
    lat: float
    lng: float
    avg_price_million_per_m2: float
    growth_pct: float

    model_config = ConfigDict(extra="forbid")


class HeatmapData(BaseModel):
    points: List[HeatmapPoint] = []
    max_value: float = 0

    model_config = ConfigDict(extra="forbid")


class TrendPoint(BaseModel):
    period: str
    price: float

    model_config = ConfigDict(extra="forbid")


class DistrictChartItem(BaseModel):
    district: str
    price: float

    model_config = ConfigDict(extra="forbid")


class TopGrowthItem(BaseModel):
    district: str
    growth: float
    price: float

    model_config = ConfigDict(extra="forbid")


class MarketAnalysis(BaseModel):
    year: int
    quarter: QuarterEnum

    avg_price: float = 0
    growth: float = 0
    transactions: int = 0

    trend: Literal["up", "down", "stable"] = "stable"

    price_index: float = 0

    line_chart: List[TrendPoint] = []

    top_growth: List[TopGrowthItem] = []

    district_chart: List[DistrictChartItem] = []

    heatmap: List[HeatmapPoint] = []

    @field_validator("quarter", mode="before")
    @classmethod
    def normalize_quarter(cls, v):
        if isinstance(v, str):
            return v.upper()
        return v

    model_config = ConfigDict(extra="forbid")
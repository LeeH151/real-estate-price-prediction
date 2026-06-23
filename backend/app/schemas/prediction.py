from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ConfigDict,
)

from typing import (
    Optional,
    List,
    Dict,
    Any,
    Generic,
    TypeVar,
    Literal,
)

from datetime import datetime
import pandas as pd


# =========================================================
# CONSTANT MAPPINGS (SOURCE OF TRUTH)
# =========================================================

HOUSE_TYPE_MAP = {
    "front_house": "Nhà mặt tiền",
    "alley_house": "Nhà hẻm, ngõ",
    "villa": "Biệt thự, Villa",
    "residential_land": "Đất thổ cư",
    "project_land": "Đất dự án, Khu dân cư",
    "apartment": "Chung cư"  
}
LEGAL_MAP = {
    "pink book": "so hong",
    "sổ hồng": "so hong",
    "so hong": "so hong",

    "red book": "so do",
    "sổ đỏ": "so do",
    "so do": "so do",

    "unknown": "unknown",
}


# =========================================================
# INPUT
# =========================================================
class PredictionInput(BaseModel):
    # =====================================================
    # LOCATION
    # =====================================================
    province: str = Field(
        default="Hồ Chí Minh",
        max_length=100
    )

    district: str = Field(
        ...,
        min_length=2,
        max_length=100
    )

    ward: str = Field(
        default="Unknown",
        max_length=100
    )

    # =====================================================
    # PROPERTY
    # =====================================================
    area_m2: float = Field(
        ...,
        gt=5,
        lt=10000
    )

    width: float = Field(
        default=0,
        ge=0,
        le=1000
    )

    length: float = Field(
        default=0,
        ge=0,
        le=1000
    )

    bedrooms: int = Field(
        default=0,
        ge=0,
        le=100
    )

    bathrooms: int = Field(
        default=0,
        ge=0,
        le=100
    )

    floors: int = Field(
        default=1,
        ge=0,
        le=100
    )

    # =====================================================
    # HOUSE TYPE
    # MUST MATCH DATASET
    # =====================================================
    house_type: Literal[
        "front_house",
        "alley_house",
        "villa",
        "residential_land",
        "project_land",
        "apartment"
    ]

    # =====================================================
    # LEGAL
    # =====================================================
    legal: str = Field(
        default="unknown",
        max_length=100
    )

    # =====================================================
    # DIRECTIONS
    # =====================================================
    main_direction: str = Field(
        default="Unknown",
        max_length=50
    )

    balcony_direction: str = Field(
        default="Unknown",
        max_length=50
    )

    # =====================================================
    # CONFIG
    # =====================================================
    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore",
        str_strip_whitespace=True,
    )

    # =====================================================
    # NORMALIZE
    # =====================================================
    @field_validator(
        "province",
        "district",
        "ward",
        "legal",
        "main_direction",
        "balcony_direction",
        mode="before"
    )
    @classmethod
    def normalize_text(cls, v):

        if v is None:
            return "Unknown"

        value = str(v).strip()
        value = " ".join(value.split())

        if value == "":
            return "Unknown"

        return value

    # =====================================================
    # BUSINESS RULES
    # =====================================================
    @model_validator(mode="after")
    def validate_logic(self):

        if self.house_type == "villa" and self.area_m2 < 50:
            raise ValueError(
                "Villa area too small"
            )

        if self.bathrooms > self.bedrooms + 8:
            raise ValueError(
                "Invalid bathrooms count"
            )

        if (
            self.width > 0
            and self.length > 0
        ):
            estimated_area = (
                self.width * self.length
            )

            diff = abs(
                estimated_area
                - self.area_m2
            )

            if diff > self.area_m2 * 0.5:
                raise ValueError(
                    "Width/Length mismatch area"
                )

        return self

    # =====================================================
    # MODEL INPUT
    # =====================================================
    def to_model_input(self) -> pd.DataFrame:

        legal_key = (
            self.legal
            .strip()
            .lower()
        )

        legal = LEGAL_MAP.get(
            legal_key,
            self.legal
        )

        data = {

            "Location": [
                f"{self.ward}, {self.district}"
            ],

            "Land Area": [
                f"{float(self.area_m2)} m²"
            ],

            "Bedrooms": [
                int(self.bedrooms)
            ],

            "Toilets": [
                int(self.bathrooms)
            ],

            "Total Floors": [
                int(self.floors)
            ],

            "Type of House": [
                HOUSE_TYPE_MAP[
                    self.house_type
                ]
            ],

            "Main Door Direction": [
                self.main_direction
            ],

            "Balcony Direction": [
                self.balcony_direction
            ],

            "Legal Documents": [
                legal
            ],
        }

        return pd.DataFrame(data)

# =========================================================
# RESULT
# =========================================================
class PredictionResult(BaseModel):

    predicted_price_billion_vnd: float = Field(default=0, ge=0)
    min_price: float = Field(default=0, ge=0)
    max_price: float = Field(default=0, ge=0)

    confidence_score: float = Field(default=0, ge=0, le=1)


# =========================================================
# OUTPUT
# =========================================================
class PredictionOutput(PredictionResult):

    model_version: str = "v3.0.0"

    comparison: Optional[List[Dict[str, Any]]] = None
    trend: Optional[List[Dict[str, Any]]] = None

    model_config = ConfigDict(extra="allow")


# =========================================================
# HISTORY ITEM
# =========================================================
class PredictionHistoryItem(BaseModel):

    prediction_id: str
    user_id: Optional[str] = None
    user_name: Optional[str] = None

    input_data: Optional[PredictionInput] = None
    result: Optional[PredictionResult] = None

    created_at: Optional[datetime] = None


# =========================================================
# PAGINATION
# =========================================================
T = TypeVar("T")


class Page(BaseModel, Generic[T]):

    items: List[T] = Field(default_factory=list)
    total: int = 0
    page: int = 1
    size: int = 10

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def total_pages(self):

        if self.size <= 0:
            return 0

        return (self.total + self.size - 1) // self.size


# =========================================================
# SAVE REQUEST
# =========================================================
class SavePredictionRequest(BaseModel):

    input_data: PredictionInput
    result: Dict[str, Any]

    model_config = ConfigDict(extra="ignore")
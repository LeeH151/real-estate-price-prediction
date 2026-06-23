from fastapi import APIRouter, Query, HTTPException, Depends, status
import logging
from typing import Optional

from app.schemas.market import MarketStats, HeatmapData, MarketAnalysis
from app.services.market_service import get_market_stats, get_heatmap, get_market_analysis
from app.core.security import get_current_user_optional

router = APIRouter(prefix="/market", tags=["Market"])
logger = logging.getLogger(__name__)

QUARTERS = {"Q1", "Q2", "Q3", "Q4"}


# =========================
# MARKET STATS
# =========================
@router.get("/stats", response_model=MarketStats)
async def stats(current_user: Optional[dict] = Depends(get_current_user_optional)):
    try:
        data = await get_market_stats()
        return MarketStats(
            avg_price=data.get("avg_price", 0),
            total_listings=data.get("total_listings", 0),
            prediction_count=data.get("prediction_count", 0),
        )
    except Exception as e:
        logger.exception(f"MARKET_STATS_ERROR: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load market stats",
        )


# =========================
# HEATMAP
# =========================
@router.get("/heatmap", response_model=HeatmapData)
async def heatmap(current_user: Optional[dict] = Depends(get_current_user_optional)):
    try:
        data = await get_heatmap()
        return HeatmapData(
            points=data.get("points", []),
            max_value=data.get("max_value", 0),
        )
    except Exception as e:
        logger.exception(f"HEATMAP_ERROR: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load heatmap",
        )


# =========================
# MARKET ANALYSIS
# =========================
@router.get("/analysis", response_model=MarketAnalysis)
async def analysis(
    year: int = Query(..., ge=2000, le=2100),
    quarter: str = Query(...),
    current_user: Optional[dict] = Depends(get_current_user_optional),
):
    try:
        quarter = quarter.upper()

        if quarter not in QUARTERS:
            raise HTTPException(
                status_code=400,
                detail="Quarter must be Q1, Q2, Q3, Q4",
            )

        data = await get_market_analysis(year, quarter)

        if not data:
            data = {
                "avg_price": 2.5,
                "growth": 3.0,
                "transactions": 1200,
                "line_chart": [],
                "top_growth": [],
                "district_chart": [],
                "heatmap": [],
            }

        growth = float(data.get("growth", 0))

        trend = (
            "up"
            if growth > 0
            else "down"
            if growth < 0
            else "stable"
        )

        avg_price = float(data.get("avg_price", 0))

        return MarketAnalysis(
            year=year,
            quarter=quarter,
            avg_price=round(avg_price, 2),
            growth=round(growth, 2),
            transactions=int(data.get("transactions", 0)),
            trend=trend,
            price_index=round(avg_price * (1 + growth / 100), 2),
            line_chart=data.get("line_chart", []),
            top_growth=data.get("top_growth", []),
            district_chart=data.get("district_chart", []),
            heatmap=data.get("heatmap", []),
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(f"MARKET_ANALYSIS_ERROR: {e}")

        return MarketAnalysis(
            year=year,
            quarter=quarter,
            avg_price=0,
            growth=0,
            transactions=0,
            trend="stable",
            price_index=0,
            line_chart=[],
            top_growth=[],
            district_chart=[],
            heatmap=[],
        )
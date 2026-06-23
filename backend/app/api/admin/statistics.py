from fastapi import APIRouter, Depends, HTTPException, status
import logging

from app.api.deps import require_admin
from app.schemas.statistics import StatisticsResponse
from app.services.statistics_service import get_statistics_service

router = APIRouter(
    prefix="/admin/statistics",
    tags=["Admin Statistics"]
)

logger = logging.getLogger(__name__)


@router.get(
    "",
    response_model=StatisticsResponse,
    status_code=status.HTTP_200_OK
)
async def get_statistics(admin_user=Depends(require_admin)):
    try:
        data = await get_statistics_service()

        return data

    except Exception as e:
        logger.exception("❌ Statistics API failed")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load statistics"
        )
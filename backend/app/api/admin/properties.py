from fastapi import APIRouter, Depends, Query, status
from app.api.deps import require_admin
from app.schemas.property import PropertyCreate
from app.services.property_service import (
    get_properties_service,
    create_property_service
)

router = APIRouter(
    prefix="/admin/properties",
    tags=["Admin Properties"]
)


# =========================
# GET PROPERTIES
# =========================
@router.get("", status_code=status.HTTP_200_OK)
async def get_properties(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    search: str = "",
    district: str = "",
    ward: str = "",
    user=Depends(require_admin)
):
    return await get_properties_service(
        page, page_size, search, district, ward
    )


# =========================
# CREATE PROPERTY
# =========================
@router.post("", status_code=status.HTTP_201_CREATED)
async def create_property(
    payload: PropertyCreate,
    user=Depends(require_admin)
):
    return await create_property_service(payload)
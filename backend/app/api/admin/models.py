from fastapi import APIRouter, Depends, Query, HTTPException
from app.api.deps import require_admin
from app.services.model_service import get_models_service

router = APIRouter(
    prefix="/admin/models",
    tags=["Admin Models"]
)


@router.get("")
async def get_models(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    search: str = Query("")
):
    try:
        data = await get_models_service(page, page_size, search)

        return {
            "items": data["items"],
            "total": data["total"],
            "page": page,
            "page_size": page_size
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch models: {str(e)}"
        )
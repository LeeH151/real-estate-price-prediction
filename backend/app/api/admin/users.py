from fastapi import APIRouter, Depends, Query, Path, status

from app.api.deps import require_admin
from app.schemas.user import (
    UserCreate,
    UserListResponse
)
from app.services.user_service import (
    get_users_service,
    create_user_service,
    update_user_status_service
)

router = APIRouter(
    prefix="/admin/users",
    tags=["Admin Users"]
)


# =========================
# GET USERS
# =========================
@router.get(
    "",
    response_model=UserListResponse,
    status_code=status.HTTP_200_OK
)
async def get_users(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    search: str = Query("", max_length=100),
    user=Depends(require_admin)
):
    return await get_users_service(page, page_size, search, user)


# =========================
# CREATE USER
# =========================
@router.post(
    "",
    status_code=status.HTTP_201_CREATED
)
async def create_user(
    payload: UserCreate,
    user=Depends(require_admin)
):
    return await create_user_service(payload, user)


# =========================
# UPDATE USER STATUS (FIX 404)
# =========================
@router.patch(
    "/{user_id}/status",
    status_code=status.HTTP_200_OK
)
async def update_user_status(
    user_id: str = Path(..., description="MongoDB ObjectId"),
    user=Depends(require_admin)
):
    return await update_user_status_service(user_id, user)
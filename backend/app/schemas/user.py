from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Literal


# =========================
# CREATE USER
# =========================
class UserCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=100)
    role: Literal["admin", "user"] = "user"


# =========================
# UPDATE USER STATUS
# =========================
class UserUpdateStatus(BaseModel):
    status: Literal["active", "inactive"]


# =========================
# USER RESPONSE
# =========================
class UserResponse(BaseModel):
    id: str
    user_id: Optional[str]
    name: str
    email: EmailStr
    role: Literal["admin", "user"]
    status: Literal["active", "inactive"] = "active"

    class Config:
        from_attributes = True


# =========================
# PAGINATION
# =========================
class Pagination(BaseModel):
    page: int
    page_size: int
    total: int


# =========================
# LIST USERS RESPONSE
# =========================
class UserListResponse(BaseModel):
    data: List[UserResponse]
    pagination: Pagination
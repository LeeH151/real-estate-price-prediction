from pydantic import BaseModel, EmailStr, Field, field_validator
from datetime import datetime
import re

# =========================
# BASE AUTH INPUT
# =========================
class AuthInput(BaseModel):
    email: EmailStr
    password: str

    # Chuẩn hóa email về chữ thường
    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: EmailStr) -> str:
        return v.lower().strip()

    # Kiểm tra password mạnh
    @field_validator("password")
    @classmethod
    def strong_password(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError("Password too short (min 6 characters)")
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least 1 uppercase letter")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain at least 1 number")
        return v


# =========================
# REGISTER INPUT
# =========================
class RegisterInput(AuthInput):
    name: str = Field(..., min_length=2, max_length=80)
    role: str = Field("user", pattern="^(admin|user)$")


# =========================
# USER PUBLIC
# =========================
class UserPublic(BaseModel):
    user_id: str
    email: EmailStr
    name: str
    picture: str | None = None
    role: str
    created_at: datetime

    model_config = {
        "from_attributes": True,  # cho phép chuyển từ ORM hoặc dict
    }

    # Chuyển created_at từ ISO string thành datetime nếu cần
    @field_validator("created_at", mode="before")
    @classmethod
    def parse_created_at(cls, v):
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v)
            except Exception:
                pass
        return v


# =========================
# AUTH RESPONSE
# =========================
class AuthResponse(BaseModel):
    user: UserPublic
    access_token: str
    token_type: str = "bearer"
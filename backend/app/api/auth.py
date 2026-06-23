from fastapi import APIRouter, HTTPException, Depends, status, Response
from fastapi.responses import JSONResponse
from app.schemas.auth import RegisterInput, AuthInput, AuthResponse, UserPublic
from app.services.auth_service import register_user, login_user, get_user_by_id
from app.core.security import create_access_token, get_current_user
from datetime import datetime

router = APIRouter(prefix="/auth", tags=["Auth"])


# =========================
# HÀM CHUYỂN USER SANG JSON SERIALIZABLE
# =========================
def serialize_user(user: dict) -> dict:
    """Chuyển datetime sang string ISO và trả dict"""
    serialized = user.copy()
    if "created_at" in serialized and isinstance(serialized["created_at"], datetime):
        serialized["created_at"] = serialized["created_at"].isoformat()
    if "updated_at" in serialized and isinstance(serialized["updated_at"], datetime):
        serialized["updated_at"] = serialized["updated_at"].isoformat()
    return serialized


# =========================
# REGISTER
# =========================
@router.post("/register", response_model=AuthResponse)
async def register(payload: RegisterInput, response: Response):
    try:
        user = await register_user(payload)

        access_token = create_access_token({
            "sub": user["user_id"],
            "email": user["email"],
            "role": user.get("role", "user"),
        })

        return {
            "user": UserPublic(**serialize_user(user)).dict(),
            "access_token": access_token,
            "token_type": "bearer",
        }

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        print("REGISTER ERROR:", repr(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Register error")


# =========================
# LOGIN
# =========================
@router.post("/login", response_model=AuthResponse)
async def login(payload: AuthInput, response: Response):
    try:
        user = await login_user(payload)

        access_token = create_access_token({
            "sub": user["user_id"],
            "email": user["email"],
            "role": user.get("role", "user"),
        })

        return {
            "user": UserPublic(**serialize_user(user)).dict(),
            "access_token": access_token,
            "token_type": "bearer",
        }

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
    except Exception as e:
        print("LOGIN ERROR:", repr(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Login error")


# =========================
# GET CURRENT USER
# =========================
@router.get("/me", response_model=UserPublic)
async def get_me(current_user: dict = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user_id = current_user.get("user_id")
    user = await get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserPublic(**serialize_user(user))


# =========================
# LOGOUT
# =========================
@router.post("/logout")
async def logout():
    """
    Stateless logout – Frontend phải xóa access token khỏi memory hoặc cookie.
    """
    return {"message": "Logged out successfully"}
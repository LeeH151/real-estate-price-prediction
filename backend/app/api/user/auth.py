'''from fastapi import APIRouter, HTTPException, Depends, status

from app.schemas.auth import RegisterInput, AuthInput, AuthResponse, UserPublic
from app.services.auth_service import register_user, login_user
from app.core.security import create_access_token, get_current_user

router = APIRouter(prefix="/auth", tags=["Auth"])


# =========================
# REGISTER
# =========================
@router.post("/register", response_model=AuthResponse)
async def register(payload: RegisterInput):
    try:
        user = await register_user(payload)

        token = create_access_token({
            "sub": str(user["user_id"]),
            "email": user["email"],
            "role": user.get("role", "user"),
        })

        return {
            "user": UserPublic(**user),
            "access_token": token,
            "token_type": "bearer",
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Register error: {str(e)}",
        )


# =========================
# LOGIN
# =========================
@router.post("/login", response_model=AuthResponse)
async def login(payload: AuthInput):
    try:
        user = await login_user(payload)

        token = create_access_token({
            "sub": str(user["user_id"]),
            "email": user["email"],
            "role": user.get("role", "user"),
        })

        return {
            "user": UserPublic(**user),
            "access_token": token,
            "token_type": "bearer",
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Login error: {str(e)}",
        )


# =========================
# ME (CURRENT USER)
# =========================
@router.get("/me", response_model=UserPublic)
async def get_me(current_user=Depends(get_current_user)):
    return {
        "id": current_user["user_id"],
        "email": current_user["email"],
        "role": current_user["role"],
    }


# =========================
# LOGOUT
# =========================
@router.post("/logout")
async def logout():
    return {"message": "Logged out (client should delete token)"}'''
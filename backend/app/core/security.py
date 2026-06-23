from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from datetime import datetime, timezone
from passlib.context import CryptContext
from app.core.config import settings
import logging

# =========================
# LOGGER
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# PASSWORD HASHING
# =========================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    if not password:
        raise ValueError("Password cannot be empty")
    return pwd_context.hash(password.strip())

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plain password against hashed"""
    if not plain_password or not hashed_password:
        return False
    try:
        return pwd_context.verify(plain_password.strip(), hashed_password)
    except Exception as e:
        logger.error(f"❌ PASSWORD VERIFY ERROR: {e}")
        return False

# =========================
# JWT CREATE TOKEN
# =========================
def create_access_token(data: dict) -> str:
    """
    Create JWT access token without expiry.
    - iat must be integer (UNIX timestamp)
    - role normalized to lowercase
    """
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(data.get("sub")),
        "email": data.get("email"),
        "role": str(data.get("role", "user")).lower(),
        "iat": int(now.timestamp()),  # <-- integer timestamp
    }
    token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return token

# =========================
# SECURITY SCHEME
# =========================
security = HTTPBearer(auto_error=False)

# =========================
# REQUIRED AUTH
# =========================
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    logger.info("🔐 AUTH HEADER RECEIVED")
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
            options={"verify_exp": False},  # ignore expiry
        )
    except JWTError as e:
        logger.error(f"❌ JWT ERROR: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token: missing sub")

    return {
        "user_id": user_id,
        "email": payload.get("email"),
        "role": str(payload.get("role", "user")).lower(),
    }

# =========================
# OPTIONAL AUTH
# =========================
def get_current_user_optional(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials or not credentials.credentials:
        return None
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
            options={"verify_exp": False},
        )
        return {
            "user_id": payload.get("sub"),
            "email": payload.get("email"),
            "role": str(payload.get("role", "user")).lower(),
        }
    except JWTError as e:
        logger.warning(f"⚠️ OPTIONAL AUTH FAILED: {e}")
        return None

# =========================
# ADMIN ONLY
# =========================
def require_admin(current_user: dict = Depends(get_current_user)):
    if current_user.get("role", "user").lower() != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required")
    return current_user
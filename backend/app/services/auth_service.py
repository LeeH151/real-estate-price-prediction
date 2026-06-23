from datetime import datetime, timezone
from typing import Optional
import uuid
from app.core.database import users_col
from app.core.security import hash_password, verify_password
from app.schemas.auth import RegisterInput, AuthInput

# =========================
# HELPERS
# =========================
def now() -> str:
    """Current UTC datetime as ISO string for JSON serialization"""
    return datetime.now(timezone.utc).isoformat()


def generate_user_id() -> str:
    """Generate a new UUID for user_id"""
    return str(uuid.uuid4())


def normalize_email(email: str) -> str:
    """Lowercase and strip email"""
    return email.lower().strip()


def normalize_role(role: Optional[str]) -> str:
    """Normalize role to lowercase, default to 'user'"""
    if not role:
        return "user"
    return role.lower()


# =========================
# REGISTER
# =========================
async def register_user(payload: RegisterInput) -> dict:
    email = normalize_email(payload.email)
    password = payload.password.strip()
    role = normalize_role(payload.role)

    collection = users_col()
    existing = await collection.find_one({"email": email})
    if existing:
        raise ValueError("Email already exists")

    user_doc = {
        "user_id": generate_user_id(),
        "email": email,
        "name": payload.name.strip() if payload.name else None,
        "password_hash": hash_password(password),
        "role": role,
        "status": "active",
        "created_at": now(),
        "updated_at": now(),
    }

    await collection.insert_one(user_doc)

    # Return user data ready for JSON serialization
    return {
        "user_id": user_doc["user_id"],
        "email": user_doc["email"],
        "name": user_doc["name"],
        "role": user_doc["role"],
        "status": user_doc["status"],
        "created_at": user_doc["created_at"],
    }


# =========================
# LOGIN
# =========================
async def login_user(payload: AuthInput) -> dict:
    email = normalize_email(payload.email)
    password = payload.password.strip()

    collection = users_col()
    user = await collection.find_one({"email": email})
    if not user:
        raise ValueError("Invalid credentials")
    if user.get("status") != "active":
        raise ValueError("Account is not active")
    if not verify_password(password, user.get("password_hash")):
        raise ValueError("Invalid credentials")

    role = normalize_role(user.get("role"))

    # Return user data ready for JSON serialization
    return {
        "user_id": user["user_id"],
        "email": user["email"],
        "name": user.get("name"),
        "role": role,
        "status": user.get("status", "active"),
        "created_at": user.get("created_at").isoformat() if isinstance(user.get("created_at"), datetime) else user.get("created_at"),
    }


# =========================
# GET USER BY ID
# =========================
async def get_user_by_id(user_id: str) -> Optional[dict]:
    collection = users_col()
    user = await collection.find_one({"user_id": user_id})
    if not user:
        return None

    # Remove MongoDB internal ID and password hash
    user.pop("_id", None)
    user.pop("password_hash", None)

    # Convert datetime to ISO string for JSON serialization
    if isinstance(user.get("created_at"), datetime):
        user["created_at"] = user["created_at"].isoformat()
    if isinstance(user.get("updated_at"), datetime):
        user["updated_at"] = user["updated_at"].isoformat()

    # Normalize role
    user["role"] = normalize_role(user.get("role"))

    return user
from datetime import datetime
from fastapi import HTTPException
from passlib.context import CryptContext
from bson import ObjectId
import uuid

from app.core.database import get_db

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# =========================
# HASH PASSWORD
# =========================
def hash_password(password: str):
    return pwd_context.hash(password)


# =========================
# NORMALIZE ADMIN USER
# =========================
def get_admin_id(admin_user):
    return admin_user.get("id") or admin_user.get("user_id")


# =========================
# VALIDATE OBJECT ID
# =========================
def parse_object_id(user_id: str):
    try:
        return ObjectId(user_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid user id")


# =========================
# GET USERS
# =========================
async def get_users_service(page, page_size, search, admin_user):
    db = get_db()

    query = {"role": "user"}

    if search:
        query["email"] = {"$regex": search, "$options": "i"}

    users = await db.users.find(
        query,
        {"password_hash": 0}
    ).skip((page - 1) * page_size).limit(page_size).to_list(length=page_size)

    # format
    for u in users:
        u["id"] = str(u["_id"])
        u["user_id"] = u.get("user_id")
        u["status"] = u.get("status", "active")
        del u["_id"]

    total = await db.users.count_documents(query)

    return {
        "data": users,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total
        }
    }


# =========================
# CREATE USER
# =========================
async def create_user_service(payload, admin_user):
    db = get_db()

    existing = await db.users.find_one({"email": payload.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already exists")

    admin_id = get_admin_id(admin_user)

    user_dict = payload.dict()

    # hash password
    user_dict["password_hash"] = hash_password(payload.password)

    # generate UUID
    user_dict["user_id"] = str(uuid.uuid4())

    # audit
    user_dict.update({
        "created_by": admin_id,
        "created_at": datetime.utcnow(),
        "status": "active"
    })

    result = await db.users.insert_one(user_dict)

    # log
    await db.admin_logs.insert_one({
        "admin_id": admin_id,
        "action": "CREATE_USER",
        "target_email": payload.email,
        "timestamp": datetime.utcnow()
    })

    return {
        "message": "User created successfully",
        "id": str(result.inserted_id)
    }


# =========================
# UPDATE USER STATUS (FIXED)
# =========================
async def update_user_status_service(user_id: str, admin_user):
    db = get_db()

    admin_id = get_admin_id(admin_user)

    obj_id = parse_object_id(user_id)

    user = await db.users.find_one({"_id": obj_id})

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    current_status = user.get("status", "active")

    new_status = "inactive" if current_status == "active" else "active"

    await db.users.update_one(
        {"_id": obj_id},
        {
            "$set": {
                "status": new_status,
                "updated_at": datetime.utcnow(),
                "updated_by": admin_id
            }
        }
    )

    # log admin action
    await db.admin_logs.insert_one({
        "admin_id": admin_id,
        "action": "UPDATE_USER_STATUS",
        "target_user_id": user_id,
        "new_status": new_status,
        "timestamp": datetime.utcnow()
    })

    return {
        "message": "User status updated",
        "status": new_status
    }
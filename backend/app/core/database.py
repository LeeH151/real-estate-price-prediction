import certifi
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

client = None
db = None


# =========================
# CONNECT DB
# =========================
async def connect_db():
    global client, db

    print("🔌 Connecting to MongoDB...")

    client = AsyncIOMotorClient(
        settings.MONGO_URL,
        tls=True,
        tlsCAFile=certifi.where(),   # 🔥 FIX SSL
        serverSelectionTimeoutMS=30000
    )

    db = client[settings.DB_NAME]

    await client.admin.command("ping")

    print(f"✅ Connected to DB: {settings.DB_NAME}")


# =========================
# CLOSE DB
# =========================
async def close_db():
    global client

    if client:
        client.close()
        print("🔌 MongoDB connection closed")


# =========================
# GET DB (SAFE)
# =========================
def get_db():
    # ✅ FIX 1: tránh None silent error
    global db

    if db is None:
        raise RuntimeError("❌ Database not connected (check startup event)")

    return db


# =========================
# COLLECTIONS
# =========================
def users_col():
    return get_db()["users"]


def prediction_col():
    return get_db()["prediction_history"]


# =========================
# ✅ FIX BỔ SUNG QUAN TRỌNG (THÊM MỚI)
# =========================

# 🔥 helper: convert ObjectId -> string (tránh lỗi FastAPI 500)
def normalize_id(doc: dict):
    if not doc:
        return doc

    if "_id" in doc:
        doc["_id"] = str(doc["_id"])

    return doc


# 🔥 helper: normalize list docs
def normalize_list(docs: list):
    return [normalize_id(d) for d in docs]

def user_col():
    return db["users"]

def model_col():
    return db["models"]
import asyncio
import logging

from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# =========================
# SEED (SAFE / NO SIDE EFFECT)
# =========================
async def seed():
    client = AsyncIOMotorClient(settings.MONGO_URL)
    db = client[settings.DB_NAME]

    logger.info("🔥 SEED START (SAFE MODE)")

    try:
        # chỉ check connection
        await db.command("ping")
        logger.info("✅ MongoDB connected successfully")

        # chỉ thống kê, KHÔNG thay đổi data
        users_count = await db.users.count_documents({})
        models_count = await db.ai_models.count_documents({})
        pred_count = await db.prediction_history.count_documents({})

        logger.info(f"📦 USERS: {users_count}")
        logger.info(f"📦 AI MODELS: {models_count}")
        logger.info(f"📦 PREDICTIONS: {pred_count}")

        logger.info("🔥 SEED DONE (NO CHANGES)")

    except Exception as e:
        logger.error(f"❌ SEED FAILED: {e}")

    finally:
        client.close()


# =========================
# RUN
# =========================
if __name__ == "__main__":
    asyncio.run(seed())
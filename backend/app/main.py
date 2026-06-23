from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.database import connect_db, close_db

# =========================
# ROUTERS
# =========================
from app.api.auth import router as auth_router
from app.api.user.predict import router as predict_router
from app.api.user.market import router as market_router

from app.api.admin.dashboard import router as dashboard_router
from app.api.admin.users import router as users_router
from app.api.admin.properties import router as properties_router
from app.api.admin.predictions import router as predictions_router
from app.api.admin.models import router as models_router
from app.api.admin.statistics import router as statistics_router

# =========================
# INIT APP
# =========================
app = FastAPI(
    title="FastAPI App",
    version="1.0.0"
)

# =========================
# CORS MIDDLEWARE
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # DEV ONLY, cho phép mọi origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_PREFIX = "/api"

# =========================
# INCLUDE ROUTERS
# =========================
app.include_router(auth_router, prefix=API_PREFIX)
app.include_router(predict_router, prefix=API_PREFIX)
app.include_router(market_router, prefix=API_PREFIX)

app.include_router(dashboard_router, prefix=API_PREFIX)
app.include_router(users_router, prefix=API_PREFIX )
app.include_router(properties_router, prefix=API_PREFIX)
app.include_router(predictions_router, prefix=API_PREFIX)
app.include_router(models_router, prefix=API_PREFIX)
app.include_router(statistics_router, prefix=API_PREFIX)

# =========================
# ROOT ENDPOINT
# =========================
@app.get("/")
def root():
    return {"message": "API is running 🚀"}

# =========================
# STARTUP / SHUTDOWN EVENTS
# =========================
@app.on_event("startup")
async def startup():
    print("🔌 Connecting to MongoDB...")
    await connect_db()
    print("✅ App ready (NO SEED MODE)")

@app.on_event("shutdown")
async def shutdown():
    print("🔌 Closing MongoDB...")
    await close_db()
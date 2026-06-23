from datetime import (
    datetime,
    timedelta,
    timezone,
)
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from pymongo import DESCENDING
from pymongo.errors import PyMongoError

from app.api.deps import require_admin
from app.core.database import get_db

router = APIRouter(
    prefix="/admin/dashboard",
    tags=["Admin Dashboard"],
)

# =========================================================
# HELPERS
# =========================================================
def success_response(data: Any):
    return {
        "success": True,
        "data": data,
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }


def handle_exception(
    error: Exception,
    context: str,
):
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"{context}: {str(error)}",
    )


# =========================================================
# SUMMARY KPI
# =========================================================
@router.get(
    "/summary",
    status_code=status.HTTP_200_OK,
)
async def summary(
    user=Depends(require_admin),
    db=Depends(get_db),
):
    try:
        # =================================================
        # TOTAL USERS
        # =================================================
        total_users_task = (
            db.users.count_documents(
                {"role": "user"}
            )
        )

        # =================================================
        # TOTAL PREDICTIONS
        # =================================================
        total_predictions_task = (
            db.prediction_history.count_documents(
                {}
            )
        )

        # =================================================
        # MOST SEARCHED AREA
        # =================================================
        most_searched_task = (
            db.prediction_history.aggregate(
                [
                    {
                        "$match": {
                            "input_data.district": {
                                "$nin": [
                                    None,
                                    "",
                                    "N/A",
                                ]
                            }
                        }
                    },
                    {
                        "$group": {
                            "_id": "$input_data.district",
                            "count": {
                                "$sum": 1
                            },
                        }
                    },
                    {
                        "$sort": {
                            "count": DESCENDING
                        }
                    },
                    {"$limit": 1},
                ]
            ).to_list(length=1)
        )

        # =================================================
        # ACTIVE MODEL
        # =================================================
        active_model_task = (
            db.ai_models.find_one(
                {"is_active": True},
                {
                    "_id": 0,
                    "name": 1,
                },
            )
        )

        # =================================================
        # EXECUTE TASKS
        # =================================================
        total_users = (
            await total_users_task
        )

        total_predictions = (
            await total_predictions_task
        )

        most_searched = (
            await most_searched_task
        )

        active_model = (
            await active_model_task
        )

        return success_response(
            {
                "total_users": total_users,
                "total_predictions": (
                    total_predictions
                ),
                "most_searched_area": (
                    most_searched[0]["_id"]
                    if most_searched
                    else None
                ),
                "active_ai_model": (
                    active_model.get("name")
                    if active_model
                    else None
                ),
            }
        )

    except PyMongoError as e:
        handle_exception(
            e,
            "Database error in summary KPI",
        )

    except Exception as e:
        handle_exception(
            e,
            "Unexpected error in summary KPI",
        )


# =========================================================
# DAILY TREND
# =========================================================
@router.get(
    "/trends/daily",
    status_code=status.HTTP_200_OK,
)
async def daily_trend(
    user=Depends(require_admin),
    db=Depends(get_db),
):
    try:
        seven_days_ago = (
            datetime.now(timezone.utc)
            - timedelta(days=7)
        )

        pipeline = [
            {
                "$match": {
                    "created_at": {
                        "$gte": (
                            seven_days_ago
                        )
                    }
                }
            },
            {
                "$group": {
                    "_id": {
                        "$dateToString": {
                            "format": (
                                "%Y-%m-%d"
                            ),
                            "date": (
                                "$created_at"
                            ),
                        }
                    },
                    "count": {
                        "$sum": 1
                    },
                }
            },
            {"$sort": {"_id": 1}},
        ]

        data = (
            await db.prediction_history.aggregate(
                pipeline,
                allowDiskUse=True,
            ).to_list(length=None)
        )

        return success_response(data)

    except PyMongoError as e:
        handle_exception(
            e,
            "Database error in daily trend",
        )

    except Exception as e:
        handle_exception(
            e,
            "Unexpected error in daily trend",
        )


# =========================================================
# TOP DISTRICTS
# =========================================================
@router.get(
    "/trends/districts",
    status_code=status.HTTP_200_OK,
)
async def top_districts(
    user=Depends(require_admin),
    db=Depends(get_db),
):
    try:
        pipeline = [
            {
                "$match": {
                    "input_data.district": {
                        "$nin": [
                            None,
                            "",
                            "N/A",
                        ]
                    }
                }
            },
            {
                "$group": {
                    "_id": (
                        "$input_data.district"
                    ),
                    "count": {
                        "$sum": 1
                    },
                }
            },
            {
                "$sort": {
                    "count": DESCENDING
                }
            },
            {"$limit": 10},
        ]

        data = (
            await db.prediction_history.aggregate(
                pipeline,
                allowDiskUse=True,
            ).to_list(length=None)
        )

        return success_response(data)

    except PyMongoError as e:
        handle_exception(
            e,
            "Database error in top districts",
        )

    except Exception as e:
        handle_exception(
            e,
            "Unexpected error in top districts",
        )


# =========================================================
# HOUSE TYPES
# =========================================================
@router.get(
    "/trends/house-types",
    status_code=status.HTTP_200_OK,
)
async def house_types(
    user=Depends(require_admin),
    db=Depends(get_db),
):
    try:
        pipeline = [
            {
                "$match": {
                    "input_data.house_type": {
                        "$nin": [
                            None,
                            "",
                            "N/A",
                        ]
                    }
                }
            },
            {
                "$group": {
                    "_id": (
                        "$input_data.house_type"
                    ),
                    "count": {
                        "$sum": 1
                    },
                }
            },
            {
                "$sort": {
                    "count": DESCENDING
                }
            },
        ]

        data = (
            await db.prediction_history.aggregate(
                pipeline,
                allowDiskUse=True,
            ).to_list(length=None)
        )

        return success_response(data)

    except PyMongoError as e:
        handle_exception(
            e,
            "Database error in house types",
        )

    except Exception as e:
        handle_exception(
            e,
            "Unexpected error in house types",
        )


# =========================================================
# FULL CHARTS
# =========================================================
@router.get(
    "/charts",
    status_code=status.HTTP_200_OK,
)
async def charts(
    user=Depends(require_admin),
    db=Depends(get_db),
):
    try:
        # =================================================
        # DAILY TREND
        # =================================================
        daily_pipeline = [
            {
                "$match": {
                    "created_at": {
                        "$ne": None
                    }
                }
            },
            {
                "$group": {
                    "_id": {
                        "$dateToString": {
                            "format": (
                                "%Y-%m-%d"
                            ),
                            "date": (
                                "$created_at"
                            ),
                        }
                    },
                    "count": {
                        "$sum": 1
                    },
                }
            },
            {"$sort": {"_id": 1}},
        ]

        # =================================================
        # DISTRICTS
        # =================================================
        districts_pipeline = [
            {
                "$match": {
                    "input_data.district": {
                        "$nin": [
                            None,
                            "",
                            "N/A",
                        ]
                    }
                }
            },
            {
                "$group": {
                    "_id": (
                        "$input_data.district"
                    ),
                    "count": {
                        "$sum": 1
                    },
                }
            },
            {
                "$sort": {
                    "count": DESCENDING
                }
            },
            {"$limit": 10},
        ]

        # =================================================
        # HOUSE TYPES
        # =================================================
        house_types_pipeline = [
            {
                "$match": {
                    "input_data.house_type": {
                        "$nin": [
                            None,
                            "",
                            "N/A",
                        ]
                    }
                }
            },
            {
                "$group": {
                    "_id": (
                        "$input_data.house_type"
                    ),
                    "count": {
                        "$sum": 1
                    },
                }
            },
            {
                "$sort": {
                    "count": DESCENDING
                }
            },
        ]

        # =================================================
        # EXECUTE QUERIES
        # =================================================
        daily = (
            await db.prediction_history.aggregate(
                daily_pipeline,
                allowDiskUse=True,
            ).to_list(length=None)
        )

        districts = (
            await db.prediction_history.aggregate(
                districts_pipeline,
                allowDiskUse=True,
            ).to_list(length=None)
        )

        house_types_data = (
            await db.prediction_history.aggregate(
                house_types_pipeline,
                allowDiskUse=True,
            ).to_list(length=None)
        )

        # DEBUG LOGS
        print(
            "districts =>",
            districts,
        )

        print(
            "house_types =>",
            house_types_data,
        )

        return success_response(
            {
                "daily_prediction_trend": (
                    daily
                ),
                "top_districts": (
                    districts
                ),
                "house_type_distribution": (
                    house_types_data
                ),
            }
        )

    except PyMongoError as e:
        handle_exception(
            e,
            "Database error in charts",
        )

    except Exception as e:
        handle_exception(
            e,
            "Unexpected error in charts",
        )
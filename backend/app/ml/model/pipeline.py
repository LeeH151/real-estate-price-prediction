import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    ExtraTreesRegressor
)

from app.ml.model.feature_builder import FeatureBuilder
from app.ml.model.utils import create_cluster_key

logger = logging.getLogger(__name__)

GLOBAL_RANDOM_STATE = 42


# =========================================================
# FEATURES
# MUST MATCH FeatureBuilder.feature_columns_
# =========================================================
NUMERIC_FEATURES = [

    # raw
    "Land Area",
    "Bedrooms",
    "Toilets",
    "Total Floors",

    # core
    "rooms",
    "ward_freq",

    # log
    "log_area",
    "log_rooms",
    "log_floors",

    # ratio
    "area_per_room",
    "area_per_floor",
    "rooms_per_area",

    # dimension
    "width",
    "length",
    "frontage_ratio",

    # district
    "district_freq",
    "district_area_median",
    "district_strength",
    "district_score",
    "district_centroid_score",
    "area_vs_district",

    # cluster
    "cluster_freq",

    # advanced
    "complex_score",
    "density_score",
    "luxury_score",

    "shape_score",

    # flags
    "is_large_area",
    "is_small_area",

    # type
    "is_land",
    "is_hem",
    "is_mat_tien",
    "is_villa",
    "is_apartment",

    # legal
    "is_so_hong",

    # missing flags
    "area_missing",
    "bedroom_missing",
    "toilet_missing",
    "floors_missing"
]

# =========================================================
# VALIDATOR
# =========================================================
class DataValidator(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        if isinstance(X, pd.DataFrame):

            X = X.replace(
                [np.inf, -np.inf],
                np.nan
            )

            X = X.reset_index(drop=True)

        return X


# =========================================================
# ENSURE COLUMNS
# =========================================================
class EnsureColumns(BaseEstimator, TransformerMixin):

    def __init__(self, columns: List[str]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        for col in self.columns:

            if col not in X.columns:
                X[col] = 0.0

        # preserve order
        X = X[self.columns]

        return X


# =========================================================
# CLUSTER INJECTOR
# =========================================================
class ClusterInjector(BaseEstimator, TransformerMixin):

    """
    SAFE cluster frequency feature
    """

    def fit(self, X, y=None):

        if "Location" not in X.columns:
            self.cluster_freq_map_ = {"unknown":1.0}
            self.default_freq_ = 1.0
            return self

        cluster = create_cluster_key(X)

        self.cluster_freq_map_ = (
            cluster.value_counts(normalize=True)
            .to_dict()
        )

        self.default_freq_ = np.mean(
            list(self.cluster_freq_map_.values())
        )

        return self

    def transform(self, X):

        X = X.copy()

        cluster = create_cluster_key(X)

        X["cluster_freq"] = (
            cluster
            .map(self.cluster_freq_map_)
            .fillna(self.default_freq_)
            .astype(np.float32)
        )

        return X


# =========================================================
# SIMPLE PREPROCESSOR
# =========================================================
class SimplePreprocessor(BaseEstimator, TransformerMixin):

    """
    Stable imputer:
    - removes all-NaN columns during fit
    - restores missing columns during transform
    - avoids sklearn shape mismatch bug
    """

    def __init__(self):

        self.imputer = SimpleImputer(
            strategy="median"
        )

        self.valid_columns_ = []
        self.all_columns_ = []

    # =====================================================
    # FIT
    # =====================================================
    def fit(self, X, y=None):

        X = X.copy()

        self.all_columns_ = list(X.columns)

        self.valid_columns_ = [

            c for c in X.columns

            if not X[c].isna().all()
        ]

        if len(self.valid_columns_) > 0:

            self.imputer.fit(
                X[self.valid_columns_]
            )

        return self

    # =====================================================
    # TRANSFORM
    # =====================================================
    def transform(self, X):

        X = X.copy()

        # no valid cols
        if len(self.valid_columns_) == 0:

            return pd.DataFrame(
                0.0,
                index=X.index,
                columns=self.all_columns_
            )

        # transform valid cols
        X_valid = X[self.valid_columns_]

        X_imp = self.imputer.transform(X_valid)

        X_imp = pd.DataFrame(
            X_imp,
            columns=self.valid_columns_,
            index=X.index
        )

        # restore dropped cols
        for col in self.all_columns_:

            if col not in X_imp.columns:
                X_imp[col] = 0.0

        # preserve exact order
        X_imp = X_imp[self.all_columns_]

        # final cleanup
        X_imp = X_imp.replace(
            [np.inf, -np.inf],
            0
        ).fillna(0)

        return X_imp.astype(np.float32)


# =========================================================
# MODEL FACTORY
# =========================================================
def build_base_model(
    model_type: str,
    params: Dict
):

    params = params or {}

    # =====================================================
    # RIDGE
    # =====================================================
    if model_type == "ridge":

        return Ridge(

            alpha=params.get(
                "alpha",
                2.0
            )
        )

    # =====================================================
    # RANDOM FOREST
    # =====================================================
    if model_type == "rf":

        return RandomForestRegressor(

            n_estimators=params.get(
                "n_estimators",
                1200
            ),

            max_depth=params.get(
                "max_depth",
                22
            ),

            min_samples_leaf=2,

            min_samples_split=5,

            max_features="sqrt",

            bootstrap=True,

            n_jobs=-1,

            random_state=GLOBAL_RANDOM_STATE
        )

    # =====================================================
    # EXTRA TREES
    # =====================================================
    if model_type in ["et", "etr"]:

        return ExtraTreesRegressor(

            n_estimators=params.get(
                "n_estimators",
                1200
            ),

            max_depth=params.get(
                "max_depth",
                28
            ),

            min_samples_leaf=2,

            min_samples_split=4,

            max_features="sqrt",

            bootstrap=False,

            n_jobs=-1,

            random_state=GLOBAL_RANDOM_STATE
        )

    # =====================================================
    # HIST GRADIENT BOOSTING
    # =====================================================
    if model_type in ["gbr", "hgb"]:

        return HistGradientBoostingRegressor(
            learning_rate=0.02,
            max_depth=10,
            max_iter=2000,
            min_samples_leaf=8,
            l2_regularization=1,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=GLOBAL_RANDOM_STATE
        )
    # =====================================================
    # FALLBACK
    # =====================================================
    logger.warning(
        f"Unknown model_type={model_type}"
    )

    return RandomForestRegressor(

        n_estimators=600,

        max_depth=18,

        n_jobs=-1,

        random_state=GLOBAL_RANDOM_STATE
    )


# =========================================================
# BUILD PIPELINE
# =========================================================
def build_pipeline(
    params: Dict = None,
    debug: bool = False
):

    params = params or {}

    model_type = params.get(
        "model_type",
        "rf"
    )

    logger.info(
        f"🚀 Production pipeline (LEVEL 5 STABLE): {model_type}"
    )

    pipe = Pipeline([
        ("cluster", ClusterInjector()),
        ("feature_builder", FeatureBuilder()),
        ("validator", DataValidator()),
        ("ensure", EnsureColumns(NUMERIC_FEATURES)),
        ("imputer", SimplePreprocessor()),
        ("model", build_base_model(model_type, params))
    ])

    return pipe
import logging
from typing import Dict

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    ExtraTreesRegressor
)

from app.ml.model.feature_builder import FeatureBuilder

logger = logging.getLogger(__name__)

GLOBAL_RANDOM_STATE = 42


# =========================================================
# FEATURE SET = SINGLE SOURCE OF TRUTH
# =========================================================
NUMERIC_FEATURES = FeatureBuilder().feature_columns_


# =========================================================
# VALIDATOR
# =========================================================
class DataValidator(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        return X.replace([np.inf, -np.inf], np.nan)


# =========================================================
# STRICT ALIGNER (SAFE + NO LEAK)
# =========================================================
class EnsureColumns(BaseEstimator, TransformerMixin):

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        for c in self.columns:
            if c not in X.columns:
                X[c] = np.nan

        X = X[self.columns]

        return X.astype(np.float32)


# =========================================================
# SAFE IMPUTER (ONLY ONE, FINAL FIX)
# =========================================================
class SafeImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.columns = X.columns
        self.medians = X.median(numeric_only=True)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        X = X.reindex(columns=self.columns)

        for c in self.columns:
            if c in self.medians:
                X[c] = X[c].fillna(self.medians[c])
            else:
                X[c] = X[c].fillna(0)

        return X.astype(np.float32)


# =========================================================
# MODEL FACTORY
# =========================================================
def build_base_model(model_type: str, params: Dict):

    if model_type == "rf":
        return RandomForestRegressor(
            n_estimators=800,
            max_depth=18,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=GLOBAL_RANDOM_STATE
        )

    if model_type == "et":
        return ExtraTreesRegressor(
            n_estimators=1000,
            max_depth=22,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=GLOBAL_RANDOM_STATE
        )

    if model_type == "hgb":
        return HistGradientBoostingRegressor(
            learning_rate=0.03,
            max_depth=10,
            max_iter=900,
            min_samples_leaf=10,
            l2_regularization=1.5,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=GLOBAL_RANDOM_STATE
        )

    return Ridge(alpha=2.5)


# =========================================================
# PIPELINE BUILDER (PRO MAX CLEAN VERSION)
# =========================================================
def build_pipeline(params: Dict = None, debug: bool = False):

    params = params or {}
    model_type = params.get("model_type", "rf")

    logger.info(f"🚀 PRO MAX CLEAN PIPELINE: {model_type}")

    return Pipeline([

        # 1. FEATURE ENGINEERING
        ("feature_builder", FeatureBuilder(debug=debug)),

        # 2. VALIDATION
        ("validator", DataValidator()),

        # 3. ALIGN (STRICT)
        ("ensure", EnsureColumns(NUMERIC_FEATURES)),

        # 4. CLEAN IMPUTATION (NO sklearn warnings)
        ("imputer", SafeImputer()),

        # 5. MODEL
        ("model", build_base_model(model_type, params))
    ])
import os
import random
import logging

import joblib
import optuna
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (
    KFold,
    GroupKFold
)

from app.ml.model.pipeline import build_pipeline
from app.ml.model.config import GLOBAL_CONFIG

logger = logging.getLogger(__name__)

SEED = GLOBAL_CONFIG.get(
    "random_state",
    42
)

ARTIFACT_DIR = "artifacts"


# =========================================================
# SEED
# =========================================================
def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)


# =========================================================
# SAFE RMSE
# =========================================================
def safe_rmse(y_true, y_pred):

    y_true = np.asarray(
        y_true,
        dtype=np.float64
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64
    )

    mask = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return np.nan

    return float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        )
    )


# =========================================================
# BASELINE
# =========================================================
def baseline_rmse(y):

    y = np.asarray(
        y,
        dtype=np.float64
    )

    mask = np.isfinite(y)

    y = y[mask]

    if len(y) == 0:
        return np.nan

    median = np.nanmedian(y)

    pred = np.full(
        shape=len(y),
        fill_value=median,
        dtype=np.float64
    )

    return safe_rmse(y, pred)


# =========================================================
# LEAKAGE CHECK
# =========================================================
def leakage_check(y_true, preds):

    y_true = np.asarray(
        y_true,
        dtype=np.float64
    )

    preds = np.asarray(
        preds,
        dtype=np.float64
    )

    mask = (
        np.isfinite(y_true)
        & np.isfinite(preds)
    )

    y_true = y_true[mask]
    preds = preds[mask]

    if len(y_true) < 10:
        return False

    try:

        if (
            np.std(y_true) > 0
            and np.std(preds) > 0
        ):

            corr = np.corrcoef(
                y_true,
                preds
            )[0, 1]

            if (
                np.isfinite(corr)
                and corr > 0.9999
            ):
                return True

    except Exception:
        pass

    residuals = y_true - preds

    res_std = np.std(residuals)

    if (
        np.isfinite(res_std)
        and res_std < 1e-6
    ):
        return True

    return False


# =========================================================
# METRIC SANITY
# =========================================================
def metric_sanity(y_true, rmse):

    y_true = np.asarray(
        y_true,
        dtype=np.float64
    )

    y_true = y_true[
        np.isfinite(y_true)
    ]

    if len(y_true) == 0:
        return False

    y_range = (
        np.max(y_true)
        - np.min(y_true)
    )

    if y_range <= 1e-12:
        return False

    ratio = rmse / (
        y_range + 1e-8
    )

    return ratio >= 0.002


# =========================================================
# NOISE TEST
# =========================================================
def noise_test(model, X_val):

    try:

        if not isinstance(
            X_val,
            pd.DataFrame
        ):
            return True

        X_noise = X_val.copy()

        numeric_cols = (
            X_noise.select_dtypes(
                include=[np.number]
            ).columns
        )

        if len(numeric_cols) == 0:
            return True

        noise = np.random.normal(
            loc=0,
            scale=1e-4,
            size=(
                len(X_noise),
                len(numeric_cols)
            )
        )

        X_noise.loc[:, numeric_cols] = (
            X_noise[numeric_cols].values
            + noise
        )

        preds1 = model.predict(X_val)
        preds2 = model.predict(X_noise)

        preds1 = np.asarray(
            preds1,
            dtype=np.float64
        )

        preds2 = np.asarray(
            preds2,
            dtype=np.float64
        )

        diff = np.nanmean(
            np.abs(preds1 - preds2)
        )

        if not np.isfinite(diff):
            return False

        return diff >= 1e-8

    except Exception as e:

        logger.warning(
            f"Noise test failed: {e}"
        )

        return True


# =========================================================
# STABILITY SCORE
# =========================================================
def stability_score(scores):

    scores = np.asarray(
        scores,
        dtype=np.float64
    )

    scores = scores[
        np.isfinite(scores)
    ]

    if len(scores) == 0:
        return np.inf

    mean = np.mean(scores)

    if mean <= 1e-12:
        return np.inf

    return float(
        np.std(scores) / mean
    )


# =========================================================
# OBJECTIVE
# =========================================================
def objective(
    trial,
    X,
    y,
    group_col=None
):

    set_seed(SEED)

    # =====================================================
    # RESET INDEX
    # =====================================================
    X = X.reset_index(drop=True)

    y = pd.Series(
        y,
        dtype=np.float64
    ).reset_index(drop=True)

    # =====================================================
    # REMOVE INVALID TARGET
    # =====================================================
    mask = np.isfinite(y)

    X = X.loc[mask].reset_index(drop=True)

    y = y.loc[mask].reset_index(drop=True)

    # =====================================================
    # MODEL TYPE
    # =====================================================
    model_type = trial.suggest_categorical(
        "model_type",
        ["ridge", "rf", "gbr"]
    )

    params = {
        "model_type": model_type
    }

    # =====================================================
    # SEARCH SPACE
    # =====================================================
    if model_type == "ridge":

        params["alpha"] = (
            trial.suggest_float(
                "alpha",
                0.01,
                50.0,
                log=True
            )
        )

    elif model_type == "rf":

        params.update({

            "n_estimators":
                trial.suggest_int(
                    "n_estimators",
                    200,
                    700
                ),

            "max_depth":
                trial.suggest_int(
                    "max_depth",
                    5,
                    20
                ),

            "min_samples_leaf":
                trial.suggest_int(
                    "min_samples_leaf",
                    1,
                    8
                ),

            "min_samples_split":
                trial.suggest_int(
                    "min_samples_split",
                    2,
                    15
                ),

            "max_features":
                trial.suggest_categorical(
                    "max_features",
                    ["sqrt", "log2"]
                )
        })

    else:

        # HistGradientBoostingRegressor
        params.update({

            "learning_rate":
                trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.10
                ),

            "max_depth":
                trial.suggest_int(
                    "max_depth",
                    2,
                    10
                ),

            "max_iter":
                trial.suggest_int(
                    "max_iter",
                    200,
                    800
                ),

            "min_samples_leaf":
                trial.suggest_int(
                    "min_samples_leaf",
                    10,
                    50
                ),

            "l2_regularization":
                trial.suggest_float(
                    "l2_regularization",
                    0.0,
                    5.0
                )
        })

    # =====================================================
    # CV
    # =====================================================
    if (
        group_col is not None
        and group_col in X.columns
    ):

        logger.info(
            "📌 GroupKFold"
        )

        groups = (
            X[group_col]
            .fillna("unknown")
            .astype(str)
        )

        cv = GroupKFold(
            n_splits=5
        )

        splits = cv.split(
            X,
            groups=groups
        )

    else:

        logger.info(
            "📌 KFold"
        )

        cv = KFold(
            n_splits=5,
            shuffle=True,
            random_state=SEED
        )

        splits = cv.split(X)

    baseline = baseline_rmse(y)

    fold_scores = []

    # =====================================================
    # CV LOOP
    # =====================================================
    for fold, (tr, va) in enumerate(
        splits,
        start=1
    ):

        model = build_pipeline(params)

        # =================================================
        # SPLIT
        # =================================================
        X_train = (
            X.iloc[tr]
            .copy()
            .reset_index(drop=True)
        )

        X_valid = (
            X.iloc[va]
            .copy()
            .reset_index(drop=True)
        )

        y_train = (
            y.iloc[tr]
            .copy()
            .reset_index(drop=True)
        )

        y_valid = (
            y.iloc[va]
            .copy()
            .reset_index(drop=True)
        )

        # =================================================
        # FIT
        # =================================================
        try:

            model.fit(
                X_train,
                y_train
            )

        except Exception as e:

            logger.warning(
                f"Fit failed: {e}"
            )

            return 999999.0

        # =================================================
        # PREDICT
        # =================================================
        try:

            train_pred = model.predict(
                X_train
            )

            valid_pred = model.predict(
                X_valid
            )

        except Exception as e:

            logger.warning(
                f"Predict failed: {e}"
            )

            return 999999.0

        # =================================================
        # SAFE PRED
        # =================================================
        fill_value = np.nanmedian(
            y_train
        )

        train_pred = np.nan_to_num(
            train_pred,
            nan=fill_value,
            posinf=fill_value,
            neginf=fill_value
        )

        valid_pred = np.nan_to_num(
            valid_pred,
            nan=fill_value,
            posinf=fill_value,
            neginf=fill_value
        )

        # =================================================
        # METRICS
        # =================================================
        train_rmse = safe_rmse(
            y_train,
            train_pred
        )

        valid_rmse = safe_rmse(
            y_valid,
            valid_pred
        )

        if not np.isfinite(valid_rmse):
            return 999999.0

        score = float(valid_rmse)

        # =================================================
        # SOFT PENALTIES
        # =================================================
        if (
            np.isfinite(train_rmse)
            and train_rmse
            < valid_rmse * 0.5
        ):
            score *= 1.10

        if (
            np.isfinite(baseline)
            and valid_rmse
            > baseline * 1.5
        ):
            score *= 1.15

        if leakage_check(
            y_valid,
            valid_pred
        ):
            score *= 1.20

        if not metric_sanity(
            y_valid,
            valid_rmse
        ):
            score *= 1.10

        if not noise_test(
            model,
            X_valid
        ):
            score *= 1.10

        fold_scores.append(score)

        # =================================================
        # PRUNING
        # =================================================
        trial.report(
            np.mean(fold_scores),
            step=fold
        )

        if trial.should_prune():

            raise optuna.TrialPruned()

    # =====================================================
    # FINAL SCORE
    # =====================================================
    mean_score = float(
        np.mean(fold_scores)
    )

    stability = stability_score(
        fold_scores
    )

    final_score = (
        mean_score
        + stability
        * mean_score
        * 0.3
    )

    # =====================================================
    # UNSTABLE PENALTY
    # =====================================================
    if stability > 0.8:
        final_score *= 1.5

    return float(final_score)


# =========================================================
# CALLBACK
# =========================================================
def save_best_callback(
    study,
    trial
):

    os.makedirs(
        ARTIFACT_DIR,
        exist_ok=True
    )

    if (
        study.best_trial.number
        == trial.number
    ):

        path = (
            f"{ARTIFACT_DIR}/"
            f"optuna_study.pkl"
        )

        joblib.dump(
            study,
            path
        )

        logger.info(
            f"💾 Saved: {path}"
        )


# =========================================================
# MAIN TUNE
# =========================================================
def tune(
    X,
    y,
    n_trials=50,
    timeout=3600,
    n_jobs=1,
    group_col=None
):

    logger.info("=" * 70)
    logger.info("🔍 OPTUNA TUNING")
    logger.info("=" * 70)

    set_seed(SEED)

    os.makedirs(
        ARTIFACT_DIR,
        exist_ok=True
    )

    # =====================================================
    # CLEAN INPUT
    # =====================================================
    X = X.reset_index(drop=True)

    y = pd.Series(
        y,
        dtype=np.float64
    ).reset_index(drop=True)

    mask = np.isfinite(y)

    X = X.loc[mask].reset_index(drop=True)

    y = y.loc[mask].reset_index(drop=True)

    logger.info(
        f"📊 TUNING SHAPE: {X.shape}"
    )

    # =====================================================
    # STUDY
    # =====================================================
    study = optuna.create_study(

        direction="minimize",

        sampler=optuna.samplers.TPESampler(
            seed=SEED
        ),

        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=2
        )
    )

    # =====================================================
    # OPTIMIZE
    # =====================================================
    study.optimize(

        lambda trial:
        objective(
            trial,
            X,
            y,
            group_col
        ),

        n_trials=n_trials,

        timeout=timeout,

        n_jobs=n_jobs,

        callbacks=[
            save_best_callback
        ],

        show_progress_bar=True
    )

    logger.info("\n🏆 BEST RESULT")

    logger.info(
        f"RMSE: "
        f"{study.best_value:.5f}"
    )

    logger.info(
        f"PARAMS: "
        f"{study.best_params}"
    )

    # =====================================================
    # FINAL MODEL
    # =====================================================
    best_model = build_pipeline(
        study.best_params
    )

    best_model.fit(X, y)

    model_path = (
        f"{ARTIFACT_DIR}/"
        f"best_model.pkl"
    )

    study_path = (
        f"{ARTIFACT_DIR}/"
        f"optuna_study.pkl"
    )

    joblib.dump(
        best_model,
        model_path
    )

    joblib.dump(
        study,
        study_path
    )

    logger.info(
        f"💾 Saved model: "
        f"{model_path}"
    )

    return {

        **study.best_params,

        "best_rmse":
            float(
                study.best_value
            ),

        "n_trials":
            int(
                len(study.trials)
            )
    }
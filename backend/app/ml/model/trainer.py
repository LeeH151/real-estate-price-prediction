import os
import time
import json
import logging
import random

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (
    KFold,
    GroupKFold
)

from app.ml.model.evaluate import evaluate
from app.ml.model.config import GLOBAL_CONFIG

logger = logging.getLogger(__name__)

ARTIFACT_DIR = "artifacts"

SEED = GLOBAL_CONFIG.get(
    "random_state",
    42
)


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
# LEAKAGE SCORE
# =========================================================
def leakage_score(y_true, preds):

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

    if len(y_true) == 0:
        return 0

    residual = y_true - preds

    score = 0

    # =====================================================
    # CORRELATION
    # =====================================================
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
                score += 1

    except Exception:
        pass

    # =====================================================
    # RMSE TOO SMALL
    # =====================================================
    rmse = safe_rmse(
        y_true,
        preds
    )

    target_std = np.std(y_true)

    if (
        np.isfinite(target_std)
        and target_std > 0
        and np.isfinite(rmse)
    ):

        ratio = rmse / target_std

        if ratio < 0.01:
            score += 1

    # =====================================================
    # RESIDUAL STD
    # =====================================================
    res_std = np.std(residual)

    if (
        np.isfinite(res_std)
        and res_std < 1e-6
    ):
        score += 1

    return int(score)


# =========================================================
# STABILITY
# =========================================================
def fold_stability(rmses):

    rmses = np.asarray(
        rmses,
        dtype=np.float64
    )

    rmses = rmses[np.isfinite(rmses)]

    if len(rmses) == 0:
        return "unknown"

    std = np.std(rmses)
    mean = np.mean(rmses)

    if mean <= 1e-12:
        return "invalid"

    cv = std / mean

    if cv < 0.01:
        return "too_stable"

    if cv > 0.30:
        return "unstable"

    return "ok"


# =========================================================
# TRAINER
# =========================================================
class Trainer:

    def __init__(
        self,
        pipeline,
        *,
        n_splits=None,
        save_best=True,
        experiment_name="regression_exp",
        use_group=False,
        group_col=None
    ):

        self.pipeline = pipeline

        self.n_splits = (
            n_splits
            or GLOBAL_CONFIG.get(
                "n_splits",
                5
            )
        )

        self.save_best = save_best

        self.experiment_name = experiment_name

        self.use_group = use_group

        self.group_col = group_col

        self.fold_history = []

        self.best_score = np.inf

        self.final_model = None

        self.oof_preds = None

    # =====================================================
    # CV
    # =====================================================
    def _get_cv(self, X):

        if (
            self.use_group
            and self.group_col is not None
            and self.group_col in X.columns
        ):

            logger.info(
                "📌 Using GroupKFold"
            )

            groups = (
                X[self.group_col]
                .fillna("unknown")
                .astype(str)
            )

            return GroupKFold(
                n_splits=self.n_splits
            ).split(
                X,
                groups=groups
            )

        logger.info("📌 Using KFold")

        return KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=SEED
        ).split(X)

    # =====================================================
    # TRAIN
    # =====================================================
    def train(self, X, y):

        logger.info("=" * 70)
        logger.info(
            f"🚀 TRAINING: "
            f"{self.experiment_name}"
        )
        logger.info("=" * 70)

        set_seed(SEED)

        # =================================================
        # RESET INDEX
        # =================================================
        X = X.reset_index(drop=True)

        if isinstance(y, pd.Series):
            y = y.reset_index(drop=True)

        y = pd.Series(
            y,
            dtype=np.float64
        )

        # =================================================
        # REMOVE INVALID TARGET
        # =================================================
        mask = np.isfinite(y)

        X = X.loc[mask].reset_index(drop=True)
        y = y.loc[mask].reset_index(drop=True)

        logger.info(
            f"📊 TRAIN SHAPE: {X.shape}"
        )

        # =================================================
        # CV
        # =================================================
        splits = self._get_cv(X)

        # =================================================
        # OOF
        # =================================================
        oof = np.full(
            len(X),
            np.nan,
            dtype=np.float64
        )

        fold_rmses = []

        baseline = baseline_rmse(y)

        logger.info(
            f"📉 Baseline RMSE: "
            f"{baseline:.5f}"
        )

        start = time.time()

        # =================================================
        # FOLD LOOP
        # =================================================
        for fold, (tr, va) in enumerate(
            splits,
            start=1
        ):

            logger.info(
                "\n" + "=" * 60
            )

            logger.info(
                f"📂 Fold "
                f"{fold}/{self.n_splits}"
            )

            logger.info("=" * 60)

            model = clone(self.pipeline)

            # =============================================
            # SPLIT
            # =============================================
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

            # =============================================
            # FIT
            # =============================================
            model.fit(
                X_train,
                y_train
            )

            # =============================================
            # PREDICT
            # =============================================
            train_pred = model.predict(
                X_train
            )

            valid_pred = model.predict(
                X_valid
            )

            # =============================================
            # SAFE PRED
            # =============================================
            train_pred = np.nan_to_num(
                train_pred,
                nan=np.nanmedian(y_train),
                posinf=np.nanmedian(y_train),
                neginf=np.nanmedian(y_train)
            )

            valid_pred = np.nan_to_num(
                valid_pred,
                nan=np.nanmedian(y_train),
                posinf=np.nanmedian(y_train),
                neginf=np.nanmedian(y_train)
            )

            # =============================================
            # SAVE OOF
            # =============================================
            oof[va] = valid_pred

            # =============================================
            # METRICS
            # =============================================
            train_rmse = safe_rmse(
                y_train,
                train_pred
            )

            valid_rmse = safe_rmse(
                y_valid,
                valid_pred
            )

            fold_rmses.append(valid_rmse)

            # =============================================
            # LEAK CHECK
            # =============================================
            leak = leakage_score(
                y_valid,
                valid_pred
            )

            # =============================================
            # WARNINGS
            # =============================================
            if (
                np.isfinite(train_rmse)
                and np.isfinite(valid_rmse)
            ):

                if (
                    train_rmse
                    < valid_rmse * 0.5
                ):
                    logger.warning(
                        "⚠️ Potential overfit"
                    )

            if (
                np.isfinite(valid_rmse)
                and np.isfinite(baseline)
            ):

                if (
                    valid_rmse
                    > baseline * 1.5
                ):
                    logger.warning(
                        "⚠️ Worse than baseline"
                    )

            if leak >= 2:

                logger.error(
                    "🚨 Leakage suspected"
                )

            # =============================================
            # SAVE FOLD
            # =============================================
            self.fold_history.append({

                "fold": int(fold),

                "train_rmse": float(
                    train_rmse
                ),

                "valid_rmse": float(
                    valid_rmse
                ),

                "leak_score": int(leak)
            })

            logger.info(
                f"RMSE "
                f"train={train_rmse:.5f} | "
                f"valid={valid_rmse:.5f}"
            )

        # =================================================
        # OOF EVALUATION
        # =================================================
        mask = np.isfinite(oof)

        oof_rmse = safe_rmse(
            y[mask],
            oof[mask]
        )

        stability = fold_stability(
            fold_rmses
        )

        leak_oof = leakage_score(
            y[mask],
            oof[mask]
        )

        logger.info(
            "\n" + "=" * 70
        )

        logger.info(
            "📊 FINAL SUMMARY"
        )

        logger.info("=" * 70)

        logger.info(
            f"OOF RMSE: "
            f"{oof_rmse:.5f}"
        )

        logger.info(
            f"Leak Score: "
            f"{leak_oof}"
        )

        logger.info(
            f"Stability: "
            f"{stability}"
        )

        # =================================================
        # DETAILED EVALUATION
        # =================================================
        try:

            evaluate(
                y[mask],
                oof[mask],
                verbose=True
            )

        except Exception as e:

            logger.warning(
                f"Evaluate failed: {e}"
            )

        # =================================================
        # FINAL MODEL
        # =================================================
        final_model = clone(
            self.pipeline
        )

        final_model.fit(X, y)

        self.final_model = final_model

        self.best_score = oof_rmse

        self.oof_preds = oof

        # =================================================
        # SAVE
        # =================================================
        os.makedirs(
            ARTIFACT_DIR,
            exist_ok=True
        )

        version = int(time.time())

        model_path = (
            f"{ARTIFACT_DIR}/"
            f"model_{version}.pkl"
        )

        latest_path = (
            f"{ARTIFACT_DIR}/"
            f"model_latest.pkl"
        )

        metadata = {

            "experiment_name":
                self.experiment_name,

            "version":
                version,

            "oof_rmse":
                float(oof_rmse),

            "baseline_rmse":
                float(baseline),

            "leak_score":
                int(leak_oof),

            "stability":
                stability,

            "n_samples":
                int(len(X)),

            "n_splits":
                int(self.n_splits),

            "training_time_sec":
                round(
                    time.time() - start,
                    2
                )
        }

        # =================================================
        # SAVE HISTORY
        # =================================================
        with open(
            f"{ARTIFACT_DIR}/train_history.json",
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                self.fold_history,
                f,
                indent=4,
                ensure_ascii=False
            )

        # =================================================
        # SAVE METADATA
        # =================================================
        with open(
            f"{ARTIFACT_DIR}/metadata.json",
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                metadata,
                f,
                indent=4,
                ensure_ascii=False
            )

        # =================================================
        # SAVE MODEL
        # =================================================
        if self.save_best:

            joblib.dump(
                final_model,
                model_path
            )

            joblib.dump(
                final_model,
                latest_path
            )

            logger.info(
                "\n💾 MODEL SAVED"
            )

            logger.info(
                f"📦 {model_path}"
            )

        logger.info(
            "\n✅ TRAINING COMPLETE"
        )

        return final_model
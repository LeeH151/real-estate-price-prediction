import logging
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

logger = logging.getLogger(__name__)


# =========================================================
# SAFE CLEAN
# =========================================================
def clean_arrays(y_true, y_pred=None):

    y_true = np.asarray(
        y_true,
        dtype=np.float64
    )

    if y_pred is None:

        mask = np.isfinite(y_true)

        return y_true[mask]

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64
    )

    mask = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    return (
        y_true[mask],
        y_pred[mask]
    )


# =========================================================
# SAFE METRICS
# =========================================================
def safe_mae(y_true, y_pred):

    try:

        y_true, y_pred = clean_arrays(
            y_true,
            y_pred
        )

        if len(y_true) == 0:
            return np.nan

        return float(
            mean_absolute_error(
                y_true,
                y_pred
            )
        )

    except Exception as e:

        logger.warning(
            f"MAE failed: {e}"
        )

        return np.nan


def safe_rmse(y_true, y_pred):

    try:

        y_true, y_pred = clean_arrays(
            y_true,
            y_pred
        )

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

    except Exception as e:

        logger.warning(
            f"RMSE failed: {e}"
        )

        return np.nan


def safe_r2(y_true, y_pred):

    try:

        y_true, y_pred = clean_arrays(
            y_true,
            y_pred
        )

        if len(y_true) < 2:
            return np.nan

        score = float(
            r2_score(
                y_true,
                y_pred
            )
        )

        # =============================================
        # SAFETY
        # =============================================
        if not np.isfinite(score):
            return np.nan

        # prevent absurd values
        score = np.clip(score, -10, 1)

        return score

    except Exception as e:

        logger.warning(
            f"R2 failed: {e}"
        )

        return np.nan


# =========================================================
# BASELINE
# =========================================================
def baseline_rmse(y_true):

    y_true = clean_arrays(y_true)

    if len(y_true) == 0:
        return np.nan

    baseline = np.full(
        shape=len(y_true),
        fill_value=np.mean(y_true),
        dtype=np.float64
    )

    return safe_rmse(
        y_true,
        baseline
    )


# =========================================================
# LOG -> REAL SCALE
# TARGET UNIT = BILLION VND
# =========================================================
def to_real_scale(y_log):

    y_log = np.asarray(
        y_log,
        dtype=np.float64
    )

    # =============================================
    # ANTI OVERFLOW
    # =============================================
    y_log = np.clip(
        y_log,
        -20,
        20
    )

    result = np.expm1(y_log)

    result = np.nan_to_num(
        result,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return result


# =========================================================
# REAL SCALE METRICS
# =========================================================
def evaluate_real_scale(
    y_true_log,
    y_pred_log,
    clip_prediction=True
):

    y_true = to_real_scale(y_true_log)

    y_pred = to_real_scale(y_pred_log)

    # =============================================
    # CLIP PREDICTIONS
    # =============================================
    if (
        clip_prediction
        and len(y_true) > 0
    ):

        cap = np.percentile(
            y_true,
            99
        )

        if not np.isfinite(cap):
            cap = np.max(y_true)

        if not np.isfinite(cap):
            cap = 100.0

        cap = max(cap, 1.0)

        y_pred = np.clip(
            y_pred,
            0,
            cap * 3
        )

    return {

        "mae_real": safe_mae(
            y_true,
            y_pred
        ),

        "rmse_real": safe_rmse(
            y_true,
            y_pred
        ),

        "r2_real": safe_r2(
            y_true,
            y_pred
        ),

        "unit": "BILLION_VND"
    }


# =========================================================
# LEAKAGE / SUSPICIOUS DETECTOR
# =========================================================
def detect_suspicious(
    y_true,
    y_pred
):

    warnings = []

    y_true, y_pred = clean_arrays(
        y_true,
        y_pred
    )

    if len(y_true) < 5:
        return warnings

    residual = y_true - y_pred

    mae = np.mean(
        np.abs(residual)
    )

    # =============================================
    # NEAR PERFECT
    # =============================================
    if mae < 1e-6:

        warnings.append(
            "🚨 Near-perfect prediction "
            "(possible leakage)"
        )

    # =============================================
    # ZERO VARIANCE
    # =============================================
    if np.std(residual) < 1e-6:

        warnings.append(
            "🚨 Residual variance near zero"
        )

    # =============================================
    # CORRELATION
    # =============================================
    try:

        if (
            np.std(y_true) > 0
            and np.std(y_pred) > 0
        ):

            corr = np.corrcoef(
                y_true,
                y_pred
            )[0, 1]

            if (
                np.isfinite(corr)
                and corr > 0.999
            ):

                warnings.append(
                    "🚨 Extremely high correlation"
                )

    except Exception:
        pass

    return warnings


# =========================================================
# METRIC CONSISTENCY
# =========================================================
def check_metric_consistency(
    y_true,
    rmse,
    mae
):

    warnings = []

    y_true = clean_arrays(y_true)

    if len(y_true) == 0:
        return warnings

    y_range = (
        np.max(y_true)
        - np.min(y_true)
    )

    if y_range <= 1e-12:

        warnings.append(
            "⚠️ Constant target"
        )

        return warnings

    # =============================================
    # SUSPICIOUSLY LOW ERROR
    # =============================================
    if (
        rmse is not None
        and np.isfinite(rmse)
        and rmse / y_range < 0.01
    ):

        warnings.append(
            "🚨 RMSE suspiciously small"
        )

    if (
        mae is not None
        and np.isfinite(mae)
        and mae / y_range < 0.01
    ):

        warnings.append(
            "🚨 MAE suspiciously small"
        )

    return warnings


# =========================================================
# BASELINE GAP
# =========================================================
def check_baseline_gap(
    rmse,
    baseline
):

    if (
        rmse is None
        or baseline is None
        or not np.isfinite(rmse)
        or not np.isfinite(baseline)
        or baseline <= 0
    ):

        return []

    improvement = (
        baseline - rmse
    ) / baseline

    if improvement > 0.90:

        return [
            "🚨 Improvement > 90% vs baseline"
        ]

    return []


# =========================================================
# DISTRIBUTION SHIFT
# =========================================================
def check_distribution_shift(
    y_train,
    y_test
):

    y_train = clean_arrays(y_train)

    y_test = clean_arrays(y_test)

    if (
        len(y_train) == 0
        or len(y_test) == 0
    ):

        return None

    shift = abs(
        np.mean(y_train)
        - np.mean(y_test)
    )

    std = np.std(y_train)

    if std <= 1e-12:

        return {
            "warning": "constant train target"
        }

    if shift > 0.5 * std:

        logger.warning(
            "⚠️ Distribution shift detected"
        )

    return {

        "train_mean": float(
            np.mean(y_train)
        ),

        "test_mean": float(
            np.mean(y_test)
        ),

        "shift": float(shift),
    }


# =========================================================
# DUPLICATE CHECK
# =========================================================
def check_duplicates(X):

    if X is None:
        return

    try:

        if isinstance(X, np.ndarray):

            df = pd.DataFrame(X)

        else:

            df = X.copy()

        # =============================================
        # EMPTY
        # =============================================
        if len(df) == 0:
            return

        unique_ratio = (
            len(df.drop_duplicates())
            / len(df)
        )

        if unique_ratio < 0.7:

            logger.warning(
                f"⚠️ High duplicate ratio: "
                f"{(1 - unique_ratio):.2%}"
            )

    except Exception as e:

        logger.warning(
            f"Duplicate check failed: {e}"
        )


# =========================================================
# MAIN EVALUATION
# =========================================================
def evaluate(
    y_true,
    y_pred,
    *,
    y_train=None,
    y_train_pred=None,
    X=None,
    return_dict=True,
    verbose=True,
    real_scale=True
):

    # =====================================================
    # CLEAN
    # =====================================================
    y_true, y_pred = clean_arrays(
        y_true,
        y_pred
    )

    if len(y_true) == 0:

        raise ValueError(
            "Empty evaluation set"
        )

    # =====================================================
    # METRICS
    # =====================================================
    mae = safe_mae(
        y_true,
        y_pred
    )

    rmse = safe_rmse(
        y_true,
        y_pred
    )

    r2 = safe_r2(
        y_true,
        y_pred
    )

    base_rmse = baseline_rmse(
        y_true
    )

    residuals = y_true - y_pred

    # =====================================================
    # ERROR STATS
    # =====================================================
    error_stats = {

        "mean_error": float(
            np.mean(residuals)
        ),

        "std_error": float(
            np.std(residuals)
        ),

        "median_error": float(
            np.median(residuals)
        ),

        "max_abs_error": float(
            np.max(np.abs(residuals))
        ),

        "p95_abs_error": float(
            np.percentile(
                np.abs(residuals),
                95
            )
        ),

        "p99_abs_error": float(
            np.percentile(
                np.abs(residuals),
                99
            )
        ),
    }

    # =====================================================
    # TARGET STATS
    # =====================================================
    target_stats = {

        "mean": float(
            np.mean(y_true)
        ),

        "std": float(
            np.std(y_true)
        ),

        "min": float(
            np.min(y_true)
        ),

        "max": float(
            np.max(y_true)
        ),
    }

    # =====================================================
    # WARNINGS
    # =====================================================
    warnings = []

    warnings += detect_suspicious(
        y_true,
        y_pred
    )

    warnings += check_metric_consistency(
        y_true,
        rmse,
        mae
    )

    warnings += check_baseline_gap(
        rmse,
        base_rmse
    )

    # =====================================================
    # TRAIN CHECK
    # =====================================================
    train_rmse = None

    if (
        y_train is not None
        and y_train_pred is not None
    ):

        train_rmse = safe_rmse(
            y_train,
            y_train_pred
        )

        # =============================================
        # OVERFIT CHECK
        # =============================================
        if (
            train_rmse is not None
            and np.isfinite(train_rmse)
            and rmse is not None
            and np.isfinite(rmse)
            and train_rmse > 0
        ):

            overfit_ratio = rmse / train_rmse

            if overfit_ratio > 2.0:

                warnings.append(
                    "⚠️ Possible overfitting detected"
                )

    # =====================================================
    # DISTRIBUTION SHIFT
    # =====================================================
    dist_shift = None

    if y_train is not None:

        dist_shift = check_distribution_shift(
            y_train,
            y_true
        )

    # =====================================================
    # DUPLICATE CHECK
    # =====================================================
    check_duplicates(X)

    # =====================================================
    # REAL SCALE
    # =====================================================
    real_metrics = None

    if real_scale:

        real_metrics = evaluate_real_scale(
            y_true,
            y_pred
        )

    # =====================================================
    # FINAL RESULT
    # =====================================================
    result = {

        # =============================================
        # LOG SCALE
        # =============================================
        "mae_log": mae,
        "rmse_log": rmse,
        "r2_log": r2,

        # =============================================
        # BASELINE
        # =============================================
        "baseline_rmse_log": base_rmse,

        # =============================================
        # STATS
        # =============================================
        "error_stats": error_stats,

        "target_stats": target_stats,

        # =============================================
        # TRAIN
        # =============================================
        "train_rmse_log": train_rmse,

        # =============================================
        # SHIFT
        # =============================================
        "distribution_shift": dist_shift,

        # =============================================
        # META
        # =============================================
        "n_samples": int(len(y_true)),

        "warnings": warnings,

        # =============================================
        # REAL SCALE
        # =============================================
        "real_metrics": real_metrics
    }

    # =====================================================
    # LOGGING
    # =====================================================
    if verbose:

        logger.info("=" * 60)

        logger.info("📊 EVALUATION")

        logger.info(
            f"RMSE(log): {rmse:.5f}"
        )

        logger.info(
            f"MAE(log): {mae:.5f}"
        )

        logger.info(
            f"R2(log): {r2:.5f}"
        )

        logger.info(
            f"Baseline(log): "
            f"{base_rmse:.5f}"
        )

        if real_metrics:

            logger.info(
                f"RMSE(real): "
                f"{real_metrics['rmse_real']:.3f} tỷ"
            )

            logger.info(
                f"MAE(real): "
                f"{real_metrics['mae_real']:.3f} tỷ"
            )

        if warnings:

            logger.warning(
                f"⚠️ Warnings "
                f"({len(warnings)}):"
            )

            for w in warnings:

                logger.warning(w)

        logger.info("=" * 60)

    return (
        result
        if return_dict
        else rmse
    )
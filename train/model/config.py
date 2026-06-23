"""
PRODUCTION CONFIG FOR REAL ESTATE REGRESSION
Stable | Anti-overfit | Anti-leakage
"""

# =========================================================
# GLOBAL CONFIG
# =========================================================
GLOBAL_CONFIG = {

    # reproducibility
    "random_state": 42,

    # split
    "test_size": 0.2,
    "n_splits": 5,
    "shuffle": True,

    # system
    "n_jobs": -1,
    "verbose": 1,

    # reproducibility
    "deterministic": True,

    # =====================================================
    # SPLIT STRATEGY
    # =====================================================
    # IMPORTANT:
    # duplicate-heavy dataset
    # do not use naive random split forever
    # =====================================================
    "group_split": False,
    "time_split": False,
    "stratify_bins": True,

    # =====================================================
    # STABILITY CONTROL
    # =====================================================
    "min_fold_std": 1e-3,
    "max_fold_std_ratio": 0.3,

    # =====================================================
    # DATASET
    # =====================================================
    "dataset_min_rows": 1000,
    "dataset_warning_duplicate_ratio": 0.85
}


# =========================================================
# TARGET CONFIG
# =========================================================
TARGET_CONFIG = {

    "type": "regression",

    # IMPORTANT:
    # price distribution extremely skewed
    "log_transform": True,

    # =====================================================
    # TARGET CLIP
    # =====================================================
    "clip_target_outliers": True,

    "lower_quantile": 0.005,
    "upper_quantile": 0.995,

    # =====================================================
    # TARGET VALIDATION
    # =====================================================
    "max_zero_ratio": 0.6,
    "min_target_variance": 1e-6,

    # =====================================================
    # TARGET RANGE
    # UNIT = TỶ
    # =====================================================
    "target_min": 0.01,
    "target_max": 200
}


# =========================================================
# DATA CLEANING CONFIG
# =========================================================
DATA_CONFIG = {

    # =====================================================
    # AREA
    # =====================================================
    "area_min": 5,
    "area_max": 2000,

    # =====================================================
    # ROOMS
    # =====================================================
    "max_bedrooms": 20,
    "max_toilets": 20,
    "max_floors": 15,

    # =====================================================
    # MISSING
    # =====================================================
    "fillna_num": "median",
    "fillna_cat": "mode",

    # =====================================================
    # OUTLIERS
    # =====================================================
    "outlier_method": "quantile",
    "clip_outliers": True,

    "fit_stats_on_train_only": True,
    "drop_extreme_rows": False,

    # =====================================================
    # DUPLICATES
    # =====================================================
    # IMPORTANT:
    # dataset contains many repeated listings
    # do NOT hard-remove duplicates anymore
    # =====================================================
    "remove_duplicates": False,

    "max_duplicate_ratio": 0.98,

    "check_near_duplicate": True,

    # =====================================================
    # NOISE CONTROL
    # =====================================================
    "enable_noise_injection": True,
    "noise_rate": 0.03
}


# =========================================================
# FEATURE ENGINEERING
# =========================================================
FEATURE_CONFIG = {

    # =====================================================
    # SCALING
    # =====================================================
    "scale_numeric": True,

    # =====================================================
    # ENCODING
    # =====================================================
    "encoding": "none",

    # =====================================================
    # FEATURE SELECTION
    # =====================================================
    "feature_selection": False,
    "polynomial_features": False,

    "fit_on_train_only": True,

    # =====================================================
    # CORRELATION CONTROL
    # =====================================================
    "remove_high_corr": True,
    "corr_threshold": 0.995,

    # =====================================================
    # LEAKAGE CONTROL
    # =====================================================
    "max_target_corr": 0.995,

    # =====================================================
    # FEATURE VALUE CONTROL
    # =====================================================
    "max_feature_value": 1e6,
    "min_feature_variance": 1e-8,

    # =====================================================
    # DISTRICT FEATURES
    # =====================================================
    "enable_district_features": True,

    # =====================================================
    # HOUSE TYPE FEATURES
    # =====================================================
    "enable_house_type_features": True,

    # =====================================================
    # ADVANCED FEATURES
    # =====================================================
    "enable_ratio_features": True,
    "enable_interaction_features": True
}


# =========================================================
# LINEAR MODELS
# =========================================================
LINEAR_CONFIG = {

    "model": "ridge",

    "alpha": 2.0,

    "fit_intercept": True
}


# =========================================================
# RANDOM FOREST
# =========================================================
RF_CONFIG = {

    "n_estimators": 500,

    "max_depth": 18,

    "min_samples_split": 8,

    "min_samples_leaf": 3,

    "max_features": "sqrt",

    "bootstrap": True,

    "max_samples": 0.9,

    "oob_score": False,

    # anti-overfit
    "max_leaf_nodes": None
}


# =========================================================
# HIST GRADIENT BOOSTING
# =========================================================
GBR_CONFIG = {

    "learning_rate": 0.03,

    "max_depth": 8,

    "max_iter": 600,

    "min_samples_leaf": 20,

    "l2_regularization": 2.0,

    "early_stopping": True,

    "validation_fraction": 0.1,

    "n_iter_no_change": 30
}


# =========================================================
# MODEL SELECTION
# =========================================================
MODEL_CONFIG = {

    "allowed_models": [

        "ridge",
        "rf",
        "gbr",
        "hgb"
    ],

    "default_model": "rf"
}


# =========================================================
# TUNING CONFIG
# =========================================================
TUNING_CONFIG = {

    "enabled": False,

    "method": "random",

    "n_iter": 30,

    "cv": 5,

    "scoring": "neg_root_mean_squared_error",

    "shuffle": True,

    "random_state": 42,

    # =====================================================
    # OVERFIT CONTROL
    # =====================================================
    "reject_if_better_than_baseline_ratio": 0.9,

    "reject_if_overfit_ratio": 0.6,

    "reject_if_low_variance": True
}


# =========================================================
# EVALUATION CONFIG
# =========================================================
EVAL_CONFIG = {

    "metrics": [

        "mae",
        "rmse",
        "r2"
    ],

    "primary_metric": "rmse",

    # =====================================================
    # VALIDATION
    # =====================================================
    "cv": True,

    # =====================================================
    # PLOTS
    # =====================================================
    "plot_residuals": False,
    "plot_pred_vs_actual": False,

    # =====================================================
    # SAFETY
    # =====================================================
    "check_leakage": True,
    "check_overfit": True,

    # IMPORTANT:
    # R² too high on duplicate-heavy data
    # usually fake-good
    "max_r2": 0.98,

    "min_rmse_ratio": 0.01,

    "check_residual_variance": True,
    "check_prediction_stability": True
}


# =========================================================
# PIPELINE CONFIG
# =========================================================
PIPELINE_CONFIG = {

    # =====================================================
    # LEAKAGE CONTROL
    # =====================================================
    "split_before_preprocessing": True,

    "prevent_data_leakage": True,

    "fit_scaler_on_train_only": True,

    "fit_encoder_on_train_only": True,

    # =====================================================
    # REPRODUCIBILITY
    # =====================================================
    "deterministic": True,

    # =====================================================
    # DEBUG
    # =====================================================
    "debug_noise": False,

    # =====================================================
    # ROBUSTNESS TEST
    # =====================================================
    "enable_noise_test": True,

    "noise_level": 1e-4,

    # =====================================================
    # PREDICTION SAFETY
    # =====================================================
    "clip_predictions": True,

    "prediction_min": 0,

    # UNIT = TỶ
    "prediction_max": 500
}
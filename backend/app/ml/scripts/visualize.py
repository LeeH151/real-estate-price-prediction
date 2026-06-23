import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    auc,
    f1_score
)
from sklearn.calibration import calibration_curve

ARTIFACT_DIR = "artifacts"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ======================
# LOAD DATA
# ======================
def load_data():
    path = f"{ARTIFACT_DIR}/sample_predictions.csv"

    if not os.path.exists(path):
        logger.error("❌ sample_predictions.csv NOT FOUND")
        return None

    df = pd.read_csv(path)

    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    df["proba"] = pd.to_numeric(df["proba"], errors="coerce")

    df = df.dropna()
    df["y_true"] = df["y_true"].astype(int)
    df["y_pred"] = df["y_pred"].astype(int)
    df["proba"] = df["proba"].clip(0, 1)

    return df.reset_index(drop=True)


# ======================
# ROC
# ======================
def plot_roc(df):
    fpr, tpr, _ = roc_curve(df["y_true"], df["proba"])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(f"{ARTIFACT_DIR}/roc.png")
    plt.close()


# ======================
# PR
# ======================
def plot_pr(df):
    precision, recall, _ = precision_recall_curve(df["y_true"], df["proba"])

    plt.figure()
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.savefig(f"{ARTIFACT_DIR}/pr.png")
    plt.close()


# ======================
# CONFUSION MATRIX
# ======================
def plot_confusion(df):
    cm = confusion_matrix(df["y_true"], df["y_pred"])
    cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i,j]:.2f}", ha="center")

    plt.savefig(f"{ARTIFACT_DIR}/cm.png")
    plt.close()


# ======================
# CALIBRATION
# ======================
def plot_calibration(df):
    prob_true, prob_pred = calibration_curve(
        df["y_true"], df["proba"], n_bins=10
    )

    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], "--")
    plt.title("Calibration Curve")
    plt.savefig(f"{ARTIFACT_DIR}/calibration.png")
    plt.close()


# ======================
# THRESHOLD (🔥 ADD BEST POINT)
# ======================
def plot_threshold(df):
    thresholds = np.linspace(0, 1, 100)
    scores = []

    for t in thresholds:
        preds = (df["proba"] >= t).astype(int)
        scores.append(f1_score(df["y_true"], preds))

    best_idx = np.argmax(scores)

    plt.figure()
    plt.plot(thresholds, scores)
    plt.scatter(thresholds[best_idx], scores[best_idx])
    plt.title(f"Best F1={scores[best_idx]:.4f}")
    plt.savefig(f"{ARTIFACT_DIR}/threshold.png")
    plt.close()


# ======================
# KS (🔥 ADD MAX POINT)
# ======================
def plot_ks(df):
    df = df.sort_values("proba")

    y = df["y_true"].values

    cum_pos = np.cumsum(y) / y.sum()
    cum_neg = np.cumsum(1 - y) / (1 - y).sum()

    ks_values = np.abs(cum_pos - cum_neg)
    ks = np.max(ks_values)
    idx = np.argmax(ks_values)

    plt.figure()
    plt.plot(cum_pos, label="Positive")
    plt.plot(cum_neg, label="Negative")

    plt.axvline(idx, linestyle="--")
    plt.title(f"KS = {ks:.4f}")

    plt.legend()
    plt.savefig(f"{ARTIFACT_DIR}/ks.png")
    plt.close()


# ======================
# GAIN (🔥 FIX REAL)
# ======================
def plot_gain(df):
    df = df.sort_values("proba", ascending=False).reset_index(drop=True)

    cum_target = np.cumsum(df["y_true"])
    total = df["y_true"].sum()

    gain = cum_target / total
    x = np.arange(len(df)) / len(df)

    plt.figure()
    plt.plot(x, gain, label="Model")
    plt.plot([0, 1], [0, 1], "--", label="Random")

    plt.legend()
    plt.title("Cumulative Gain")
    plt.savefig(f"{ARTIFACT_DIR}/gain.png")
    plt.close()


# ======================
# LIFT (DECILE)
# ======================
def plot_lift(df):
    df = df.sort_values("proba", ascending=False)
    df["bucket"] = pd.qcut(df.index, 10, labels=False)

    lift = df.groupby("bucket")["y_true"].mean()
    lift = lift / df["y_true"].mean()

    plt.figure()
    plt.plot(range(1, 11), lift.values, marker="o")
    plt.title("Lift Curve")
    plt.savefig(f"{ARTIFACT_DIR}/lift.png")
    plt.close()


# ======================
# 🔥 TOP-K ANALYSIS (NEW)
# ======================
def plot_topk(df):
    df = df.sort_values("proba", ascending=False)

    total_pos = df["y_true"].sum()

    ks = [0.1, 0.2, 0.3, 0.5]

    capture = []

    for k in ks:
        n = int(len(df) * k)
        captured = df.iloc[:n]["y_true"].sum()
        capture.append(captured / total_pos)

    plt.figure()
    plt.bar([str(int(k*100))+"%" for k in ks], capture)
    plt.title("Top-K Capture Rate")
    plt.savefig(f"{ARTIFACT_DIR}/topk.png")
    plt.close()


# ======================
# FEATURE IMPORTANCE
# ======================
def plot_feature_importance():
    path = f"{ARTIFACT_DIR}/model_latest.pkl"

    if not os.path.exists(path):
        return

    artifact = joblib.load(path)
    model = artifact["model"]

    try:
        est = model.named_steps["model"]
        pre = model.named_steps["preprocess"]

        features = pre.get_feature_names_out()

        if hasattr(est, "feature_importances_"):
            imp = est.feature_importances_
        elif hasattr(est, "coef_"):
            imp = np.abs(est.coef_[0])
        else:
            return

        idx = np.argsort(imp)[-20:]

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(idx)), imp[idx])
        plt.yticks(range(len(idx)), np.array(features)[idx])
        plt.title("Feature Importance")
        plt.savefig(f"{ARTIFACT_DIR}/feature_importance.png")
        plt.close()

    except Exception as e:
        logger.warning(e)


# ======================
# 🔥 SUMMARY JSON
# ======================
def save_summary(df):
    summary = {
        "accuracy": float((df["y_true"] == df["y_pred"]).mean()),
        "positive_rate": float(df["y_true"].mean())
    }

    with open(f"{ARTIFACT_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=4)


# ======================
# MAIN
# ======================
def main():
    df = load_data()

    if df is None:
        return

    plot_roc(df)
    plot_pr(df)
    plot_confusion(df)
    plot_calibration(df)
    plot_threshold(df)

    plot_ks(df)
    plot_gain(df)
    plot_lift(df)
    plot_topk(df)

    plot_feature_importance()
    save_summary(df)

    logger.info("🎉 FULL ML VISUALIZATION READY")


if __name__ == "__main__":
    main()
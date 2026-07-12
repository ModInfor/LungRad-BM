
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import (
    roc_auc_score, accuracy_score, recall_score, f1_score, roc_curve, auc
)

from config import OUTPUT_DIR, TIMESTAMP, N_BOOTSTRAP, SEED


# ============================================================
#  COLOR PALETTE
# ============================================================
COLORS = {
    "LungRad-BM":  "#d62728",
    "SVM":         "#1f77b4",
    "Logistic Regression": "#ff7f0e",
    "Random Forest": "#2ca02c",
    "KNN":         "#8c564b",
    "DeepFM":      "#9467bd",
    "TabResNet":   "#17becf",
}


# ============================================================
#  1. COMPUTE METRICS FOR A SINGLE MODEL
# ============================================================
def compute_metrics(y_true, y_score, threshold):
    """Compute AUC, ACC, Sn, Sp, F1 for one model.
    threshold: classification cutoff (e.g. Youden's J optimal threshold).
    """
    y_pred = (y_score >= threshold).astype(int)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return {
        "AUC": roc_auc_score(y_true, y_score),
        "ACC": accuracy_score(y_true, y_pred),
        "Sn": recall_score(y_true, y_pred, zero_division=0),
        "Sp": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }


# ============================================================
#  2. CONFUSION MATRIX
# ============================================================
def confusion_matrix(y_true, y_pred, labels=(0, 1)):
    """
    Compute confusion matrix as a dict {TN, FP, FN, TP}.
    Returns: (cm_dict, cm_array).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tn = int(np.sum((y_true == labels[0]) & (y_pred == labels[0])))
    fp = int(np.sum((y_true == labels[0]) & (y_pred == labels[1])))
    fn = int(np.sum((y_true == labels[1]) & (y_pred == labels[0])))
    tp = int(np.sum((y_true == labels[1]) & (y_pred == labels[1])))
    cm_dict = {"TN": tn, "FP": fp, "FN": fn, "TP": tp}
    cm_array = np.array([[tn, fp], [fn, tp]])
    return cm_dict, cm_array


def plot_confusion_matrix(y_true, y_pred, labels=("Negative", "Positive"),
                          title="Confusion Matrix", cmap="Blues"):
    """
    Plot a styled confusion matrix heatmap.
    Returns the saved file path.
    """
    _, cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, cmap=cmap, aspect="auto")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                    fontsize=22, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    ax.set_xticks([0, 1]); ax.set_xticklabels([f"Pred {labels[0]}", f"Pred {labels[1]}"])
    ax.set_yticks([0, 1]); ax.set_yticklabels([f"True {labels[0]}", f"True {labels[1]}"])
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label", fontsize=13)
    plt.title(title, fontsize=14)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{TIMESTAMP}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Confusion Matrix: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")
    print(f"  Saved: {path}")
    return path


# ============================================================
#  3. SUMMARY TABLE
# ============================================================
def build_summary_table(model_metrics_list):
    """
    Build a summary comparison table from multiple models.
    Returns pd.DataFrame sorted by AUC descending.
    """
    df = pd.DataFrame(model_metrics_list)
    cols = ["Model", "AUC", "ACC", "Sn", "Sp", "F1"]
    df = df[[c for c in cols if c in df.columns]]
    df = df.sort_values("AUC", ascending=False).reset_index(drop=True)
    for col in ["AUC", "ACC", "Sn", "Sp", "F1"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.4f}")
    return df


def save_summary_table(model_metrics_list, cohort="ITR"):
    """
    Build, print, and save the summary comparison table.
    Returns raw numeric pd.DataFrame.
    """
    raw_df = pd.DataFrame(model_metrics_list)
    cols = ["Model", "AUC", "ACC", "Sn", "Sp", "F1"]
    raw_df = raw_df[[c for c in cols if c in raw_df.columns]]
    raw_df = raw_df.sort_values("AUC", ascending=False).reset_index(drop=True)

    csv_path = os.path.join(OUTPUT_DIR, f"summary_{cohort}_{TIMESTAMP}.csv")
    raw_df.to_csv(csv_path, index=False)

    fmt_df = raw_df.copy()
    for col in ["AUC", "ACC", "Sn", "Sp", "F1"]:
        if col in fmt_df.columns:
            fmt_df[col] = fmt_df[col].apply(lambda x: f"{x:.4f}")

    print(f"\n{'='*70}")
    print(f"  {cohort} Model Comparison (sorted by AUC)")
    print(f"{'='*70}")
    print(fmt_df.to_string(index=False))
    print(f"{'='*70}")
    print(f"  Saved: {csv_path}")
    return raw_df


# ============================================================
#  4. ROC CURVES
# ============================================================
def plot_roc(fpr, tpr, auc_val, name="Model", show=False):
    """Single ROC curve."""
    fig = plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"ROC (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], "navy", lw=2, linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC - {name}")
    plt.legend(loc="lower right"); plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"roc_{name}_{TIMESTAMP}.png")
    fig.savefig(path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()
    return path


def _bootstrap_roc_ci(y_true, y_score, n_boot=N_BOOTSTRAP, seed=SEED):
    """Compute ROC curve with 95% bootstrap CI band."""
    grid = np.linspace(0, 1, 1001)
    fpr_raw, tpr_raw, _ = roc_curve(y_true, y_score)
    order = np.argsort(fpr_raw)
    mean_tpr = np.interp(grid, fpr_raw[order], tpr_raw[order])

    rng = np.random.RandomState(seed)
    n = len(y_true)
    tprs, aucs = [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        fpr_b, tpr_b, _ = roc_curve(y_true[idx], y_score[idx])
        tprs.append(np.interp(grid, fpr_b[np.argsort(fpr_b)],
                              tpr_b[np.argsort(fpr_b)]))
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))

    if tprs:
        tprs = np.vstack(tprs)
        lo_tpr = np.percentile(tprs, 2.5, axis=0)
        hi_tpr = np.percentile(tprs, 97.5, axis=0)
        auc_lo = np.percentile(aucs, 2.5)
        auc_hi = np.percentile(aucs, 97.5)
    else:
        lo_tpr = hi_tpr = mean_tpr
        auc_lo = auc_hi = np.mean(aucs) if aucs else 0.5

    auc_mean = roc_auc_score(y_true, y_score)
    return grid, mean_tpr, lo_tpr, hi_tpr, auc_mean, auc_lo, auc_hi


def plot_roc_comparison(roc_entries, cohort="ITR"):
    """
    Plot combined ROC curves with 95% CI.
    roc_entries: list of {"name": ..., "y_true": ..., "y_score": ...}
    Returns saved file path.
    """
    fig = plt.figure(figsize=(8, 8), dpi=300)

    for entry in roc_entries:
        name = entry["name"]
        c = COLORS.get(name, "#333333")
        grid_x, mt, lo, hi, auc_m, auc_l, auc_h = _bootstrap_roc_ci(
            entry["y_true"], entry["y_score"])
        plt.fill_between(grid_x, lo, hi, color=c, alpha=0.2, edgecolor="none")
        plt.plot(grid_x, mt, color=c, lw=2.5,
                 label=f"{name} AUC={auc_m:.3f} ({auc_l:.3f}–{auc_h:.3f})")

    plt.plot([0, 1], [0, 1], "gray", lw=1.4, linestyle="--")
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.xlabel("False Positive Rate", fontsize=13)
    plt.ylabel("True Positive Rate", fontsize=13)
    plt.title(f"ROC Comparison — {cohort}", fontsize=14)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, f"Fig_4_ROC_{cohort}_{TIMESTAMP}.png")
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  ROC comparison saved: {save_path}")
    return save_path


# ============================================================
#  5. FEATURE IMPORTANCE
# ============================================================
def plot_feature_importance(importance, feature_names, name="TabNet"):
    """Horizontal bar chart of feature importance. Returns saved file path."""
    imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importance}
                         ).sort_values("Importance", ascending=True)

    fig = plt.figure(figsize=(10, 10))
    plt.barh(imp_df["Feature"], imp_df["Importance"], color="skyblue")
    plt.xlabel("Importance"); plt.title(f"Feature Importance - {name}")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"importance_{name}_{TIMESTAMP}.png")
    fig.savefig(path, dpi=200)
    plt.close()

    imp_df.iloc[::-1].to_csv(
        os.path.join(OUTPUT_DIR, f"importance_{name}_{TIMESTAMP}.csv"), index=False)
    print(f"  Feature importance saved: {path}")

    print("  Top 10:")
    for i, (_, row) in enumerate(imp_df.iloc[::-1][:10].iterrows()):
        print(f"    {i+1}. {row['Feature']:30s} {row['Importance']:.6f}")
    return path


# ============================================================
#  6. TRAINING LOSS CURVE
# ============================================================
def plot_loss(history, name="TabNet"):
    """Training loss curve. Returns saved file path."""
    fig = plt.figure(figsize=(8, 8))
    plt.plot(history)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"Training Loss - {name}")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"loss_{name}_{TIMESTAMP}.png")
    fig.savefig(path, dpi=200)
    plt.close()
    return path

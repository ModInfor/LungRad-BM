"""
LungRad-BM-ITR: Individual Treatment Response (ITR) analysis pipeline.

Pipeline:
  1. LASSO feature selection (5-fold CV + λ=0.1)
  2. TabNet training (7:3 train/test) + evaluation metrics + plots
  3. 5-fold cross-validation (quantitative metrics only)
  4. SHAP interpretability analysis (heatmap + force plots for all samples)
  5. Baseline comparison (SVM, LR, RF, KNN, DeepFM, TabResNet vs TabNet)
  6. Evaluation: summary table + combined ROC with 95% CI

Usage:
  python Main/LungRad-BM-ITR.py
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from data_loader import load_data, preprocess_split
from Model.Feature_selection_LASSO import lasso_cv_analysis
from Model.Training import train_tabnet, cross_validate_tabnet
from Model.SHAP import shap_analyze
from Model.Comparison import ML_MODELS, DL_MODELS, train_eval_ml, train_eval_dl
from Model.Evaluation import (
    compute_metrics, save_summary_table, plot_roc_comparison,
    plot_roc, plot_feature_importance, plot_loss,
)


# ============================================================
#  CONFIGURATION
# ============================================================
COHORT = "ITR"
DATA_PATHS = config.DATA_PATHS_ITR
DATA_KEY = "CT"  # primary data source: "CT", "PET", or "PETCT"

OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, COHORT)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Override config OUTPUT_DIR for this run
config.OUTPUT_DIR = OUTPUT_DIR

# ============================================================
#  1. DATA LOADING
# ============================================================
file_path = DATA_PATHS[DATA_KEY]
print(f"\n{'='*60}")
print(f"[{COHORT}] Loading: {file_path}")
print(f"{'='*60}")

labels, features, feature_names = load_data(file_path)
X_train, X_test, y_train, y_test, scaler = preprocess_split(
    features, labels, seed=config.SEED)

# ============================================================
#  2. LASSO FEATURE SELECTION 
# ============================================================
print(f"\n{'='*60}")
print(f"[{COHORT}] LASSO Feature Selection")
print(f"{'='*60}")

selected_features, cv_alpha = lasso_cv_analysis(file_path, final_alpha=0.1)
print(f"  CV optimal α: {cv_alpha:.4f} | Final λ: 0.1 | Selected: {len(selected_features)} features")

# ============================================================
#  3. TABNET TRAINING (7:3 SPLIT)
# ============================================================
print(f"\n{'='*60}")
print(f"[{COHORT}] TabNet Training (7:3 Split)")
print(f"{'='*60}")

preds_proba, preds, clf, fpr, tpr, threshold = train_tabnet(
    X_train, y_train, X_test, y_test)

# TabNet metrics via Evaluation module
tabnet_metrics = compute_metrics(y_test, preds_proba, threshold=threshold)
print(f"\nTabNet 7:3 Metrics:")
for k, v in tabnet_metrics.items():
    print(f"  {k}: {v:.4f}")

pd.DataFrame([tabnet_metrics]).to_csv(
    os.path.join(OUTPUT_DIR, f"metrics_TabNet_{COHORT}_{config.TIMESTAMP}.csv"), index=False)

# Plots
plot_roc(fpr, tpr, tabnet_metrics["AUC"], f"TabNet_{COHORT}", show=False)
plot_feature_importance(clf.feature_importances_, feature_names, f"TabNet_{COHORT}")
plot_loss(clf.history["loss"], f"TabNet_{COHORT}")

# ============================================================
#  4. 5-FOLD CROSS-VALIDATION (metrics only)
# ============================================================
print(f"\n{'='*60}")
print(f"[{COHORT}] 5-Fold Cross-Validation")
print(f"{'='*60}")

cv_folds_df, cv_summary_df = cross_validate_tabnet(features, labels)

# ============================================================
#  5. SHAP INTERPRETABILITY ANALYSIS
# ============================================================
print(f"\n{'='*60}")
print(f"[{COHORT}] SHAP Analysis")
print(f"{'='*60}")

shap_values, top_shap_names = shap_analyze(
    clf.predict, X_test, y_test, feature_names)

# ============================================================
#  6. BASELINE COMPARISON (ML + DL)
# ============================================================
print(f"\n{'='*60}")
print(f"[{COHORT}] Baseline Comparison (ML + DL)")
print(f"{'='*60}")

all_model_metrics = []
roc_entries = []

# ---- Classic ML models ----
for name, model in ML_MODELS.items():
    print(f"\n--- {name} ---")
    y_score, y_pred, base_met = train_eval_ml(
        model, name, X_train, y_train, X_test, y_test)
    all_model_metrics.append(base_met)
    roc_entries.append({"name": name, "y_true": y_test, "y_score": y_score})
    for k, v in base_met.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

# ---- Deep Learning models ----
for name, model_class in DL_MODELS.items():
    print(f"\n--- {name} ---")
    y_score, y_pred, base_met, _, _, _ = train_eval_dl(
        model_class, name, X_train, y_train, X_test, y_test)
    all_model_metrics.append(base_met)
    roc_entries.append({"name": name, "y_true": y_test, "y_score": y_score})
    for k, v in base_met.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

# ---- Add TabNet (LungRad-BM) ----
tabnet_entry = {"Model": "LungRad-BM", **tabnet_metrics}
all_model_metrics.append(tabnet_entry)
roc_entries.append({"name": "LungRad-BM", "y_true": y_test, "y_score": preds_proba})

# ============================================================
#  7. EVALUATION
# ============================================================
print(f"\n{'='*60}")
print(f"[{COHORT}] Evaluation Summary")
print(f"{'='*60}")

# Summary comparison table
save_summary_table(all_model_metrics, cohort=COHORT)

plot_roc_comparison(roc_entries, cohort=COHORT)

print(f"\n{'='*60}")
print(f"[{COHORT}] All results saved to: {OUTPUT_DIR}")
print(f"{'='*60}")

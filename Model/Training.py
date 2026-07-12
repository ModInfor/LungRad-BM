
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.optim as optim

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    recall_score, f1_score, roc_curve
)

from pytorch_tabnet.tab_model import TabNetClassifier

from config import (SEED, TABNET_PARAMS, TABNET_FIT_ARGS,
                     N_FOLDS, CV_SEED, OUTPUT_DIR, TIMESTAMP)


# ============================================================
#  TABNET TRAINING (7:3 split)
# ============================================================
def train_tabnet(X_train, y_train, X_test, y_test):
    """Train TabNet classifier with paper-specified hyperparameters.

    Hyperparams (per paper): epochs=100, batch_size=8, Adam, lr=0.02, BCE.
    Threshold: optimal via Youden's J statistic on ROC curve.

    Returns
    -------
    preds_proba : array — predicted probability of positive class
    preds : array — binary predictions (0/1) at optimal threshold
    clf : TabNetClassifier — trained model
    fpr, tpr : array — ROC curve coordinates
    best_threshold : float — Youden's J optimal threshold
    """
    clf = TabNetClassifier(
        **{k: v for k, v in TABNET_PARAMS.items() if k != "optimizer_fn"},
        optimizer_fn=optim.Adam,
    )
    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_name=["test"],
        eval_metric=["accuracy"],
        **TABNET_FIT_ARGS,
    )
    preds_proba = clf.predict_proba(X_test)[:, 1]

    # Optimal threshold via Youden's J
    fpr, tpr, thresholds = roc_curve(y_test, preds_proba)
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    preds = (preds_proba >= best_threshold).astype(int)

    return preds_proba, preds, clf, fpr, tpr, best_threshold


# ============================================================
#  5-FOLD CROSS-VALIDATION (metrics only)
# ============================================================
def cross_validate_tabnet(features, labels, n_folds=N_FOLDS, seed=CV_SEED):
    """
    5-fold stratified CV for TabNet — quantitative metrics only.
    Trains a fresh model per fold, computes AUC/ACC/Sn/Sp/F1.
    Returns (per_fold_df, summary_df).
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(features, labels), 1):
        X_tr, X_val = features[train_idx], features[val_idx]
        y_tr, y_val = labels[train_idx], labels[val_idx]

        clf = TabNetClassifier(
            **{k: v for k, v in TABNET_PARAMS.items() if k != "optimizer_fn"},
            optimizer_fn=optim.Adam, verbose=0,
        )
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_name=["val"],
            eval_metric=["accuracy"],
            **TABNET_FIT_ARGS,
        )
        preds_proba = clf.predict_proba(X_val)[:, 1]

        fpr, tpr, thresholds = roc_curve(y_val, preds_proba)
        best_th = thresholds[np.argmax(tpr - fpr)]
        preds = (preds_proba >= best_th).astype(int)

        tn = np.sum((y_val == 0) & (preds == 0))
        fp = np.sum((y_val == 0) & (preds == 1))

        fold_metrics.append({
            "Fold": fold_idx,
            "AUC": roc_auc_score(y_val, preds_proba),
            "ACC": accuracy_score(y_val, preds),
            "Sn":  recall_score(y_val, preds, zero_division=0),
            "Sp":  tn / (tn + fp) if (tn + fp) > 0 else 0,
            "F1":  f1_score(y_val, preds, zero_division=0),
        })
        print(f"  Fold {fold_idx}: AUC={fold_metrics[-1]['AUC']:.4f}  "
              f"ACC={fold_metrics[-1]['ACC']:.4f}  "
              f"Sn={fold_metrics[-1]['Sn']:.4f}  "
              f"Sp={fold_metrics[-1]['Sp']:.4f}")

    per_fold_df = pd.DataFrame(fold_metrics)
    means = per_fold_df.drop(columns="Fold").mean()
    stds  = per_fold_df.drop(columns="Fold").std()
    summary_df = pd.DataFrame({"Mean": means, "Std": stds})

    print(f"\n5-Fold CV Summary (mean ± std):")
    for col in means.index:
        print(f"  {col}: {means[col]:.4f} ± {stds[col]:.4f}")

    # Save
    per_fold_df.to_csv(
        os.path.join(OUTPUT_DIR, f"cv5_folds_{TIMESTAMP}.csv"), index=False)
    summary_df.to_csv(
        os.path.join(OUTPUT_DIR, f"cv5_summary_{TIMESTAMP}.csv"))

    return per_fold_df, summary_df

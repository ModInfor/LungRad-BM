
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV

from config import OUTPUT_DIR, TIMESTAMP


def lasso_cv_analysis(file_path, final_alpha=0.1):
    """
    LASSO feature selection with 5-fold CV diagnostic plots.

    Steps (per paper):
      1. Standardize features
      2. Run LassoCV with 5-fold CV → find optimal alpha, generate
         diagnostic plots (coefficient path + MSE curve)
      3. Select features at final_alpha (λ=0.1 per paper)

    Parameters
    ----------
    file_path : str
        Path to Excel data (col 0 = label, cols 1+ = features).
    final_alpha : float
        The penalization parameter used for final feature selection (default 0.1).

    Returns
    -------
    selected_names : list[str]
        Names of features retained by LASSO.
    cv_best_alpha : float
        Optimal alpha found by 5-fold CV (for reference).
    """
    # ==================== 1. LOAD & STANDARDIZE ====================
    data = pd.read_excel(file_path)
    labels = data.iloc[:, 0].values
    features = data.iloc[:, 1:]
    feature_names = features.columns.tolist()

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    print(f"  Samples: {len(labels)} | Features: {len(feature_names)}")

    # ==================== 2. 5-FOLD CV (diagnostic) ====================
    alphas = np.logspace(-4, 1, 100)
    lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=10000)
    lasso_cv.fit(features_scaled, labels)
    cv_best_alpha = lasso_cv.alpha_

    print(f"  5-fold CV optimal alpha: {cv_best_alpha:.4f}")

    # ---- Coefficient path (per alpha) ----
    coefs = []
    for a in alphas:
        lasso = Lasso(alpha=a, max_iter=10000)
        lasso.fit(features_scaled, labels)
        coefs.append(lasso.coef_)
    coefs = np.array(coefs)

    # ==================== 3. DIAGNOSTIC PLOTS ====================
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # ---- Left: Coefficient Path ----
    for j in range(coefs.shape[1]):
        axes[0].plot(alphas, coefs[:, j], alpha=0.6)
    axes[0].axvline(x=cv_best_alpha, color="red", linestyle="--", lw=1.5,
                    label=f"CV optimal α = {cv_best_alpha:.4f}")
    axes[0].axvline(x=final_alpha, color="black", linestyle=":", lw=1.5,
                    label=f"Selected λ = {final_alpha}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Alpha (λ)")
    axes[0].set_ylabel("Coefficient Value")
    axes[0].set_title("LASSO Coefficient Path")
    axes[0].legend(fontsize=9)

    # ---- Right: 5-Fold CV MSE Curve ----
    m_log = -np.log10(lasso_cv.alphas_)
    axes[1].errorbar(m_log, lasso_cv.mse_path_.mean(axis=-1),
                     yerr=lasso_cv.mse_path_.std(axis=-1),
                     fmt="o", ms=3, capsize=2, label="Mean MSE ± Std")
    axes[1].axvline(-np.log10(cv_best_alpha), color="red", linestyle="--", lw=1.5,
                    label=f"CV optimal α = {cv_best_alpha:.4f}")
    axes[1].axvline(-np.log10(final_alpha), color="black", linestyle=":", lw=1.5,
                    label=f"Selected λ = {final_alpha}")
    axes[1].set_xlabel("-log10(λ)")
    axes[1].set_ylabel("Mean Squared Error (MSE)")
    axes[1].set_title("5-Fold Cross-Validation MSE")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f"lasso_cv_{TIMESTAMP}.png")
    fig.savefig(plot_path, dpi=300)
    plt.close()
    print(f"  Diagnostic plots saved: {plot_path}")

    # ==================== 4. FINAL SELECTION (λ=0.1) ====================
    lasso_final = Lasso(alpha=final_alpha, max_iter=10000)
    lasso_final.fit(features_scaled, labels)

    selected_idx = [i for i, coef in enumerate(lasso_final.coef_) if coef != 0]
    selected_names = [feature_names[i] for i in selected_idx]

    print(f"  LASSO (λ={final_alpha}): {len(selected_names)} / {len(feature_names)} features retained")

    # ---- Save selected features ----
    out_df = features.iloc[:, selected_idx].copy()
    out_df.insert(0, "Label", labels)
    out_path = os.path.join(OUTPUT_DIR, "lasso_selected.xlsx")
    out_df.to_excel(out_path, index=False)
    print(f"  Selected features saved: {out_path}")

    return selected_names, cv_best_alpha

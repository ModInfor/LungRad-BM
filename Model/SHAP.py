import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shap

from config import OUTPUT_DIR, TIMESTAMP, SHAP_MAX_DISPLAY, SHAP_NUM_SAMPLES


def shap_analyze(clf_predict_fn, X_test, y_test, feature_names):
    """
    Run SHAP analysis: heatmap + force plots for all test samples.

    Returns: shap_values object, top feature names.
    """
    print("Running SHAP analysis (this may take a few minutes)...")
    explainer = shap.Explainer(clf_predict_fn, X_test, feature_names=feature_names)
    shap_values = explainer(X_test)
    shap_arr = shap_values.values

    # ---- Rank features by mean |SHAP| ----
    shap_abs = np.mean(np.abs(shap_arr), axis=0)
    top_idx = np.argsort(-shap_abs)[:SHAP_MAX_DISPLAY]
    top_names = [feature_names[i] for i in top_idx]

    print("Top SHAP features (by mean |SHAP|):")
    for rank, (i, name) in enumerate(zip(top_idx, top_names), 1):
        print(f"  {rank:2d}. {name:35s} |SHAP|={shap_abs[i]:.6f}")

    # ---- Heatmap ----
    fig = plt.figure(figsize=(10, 10))
    shap.plots.heatmap(shap_values[:SHAP_NUM_SAMPLES], max_display=SHAP_MAX_DISPLAY,
                       show=False)
    plt.gca().set_yticklabels([])
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"shap_heatmap_{TIMESTAMP}.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  SHAP heatmap saved")

    # ---- Force plots ----
    force_dir = os.path.join(OUTPUT_DIR, "shap_force")
    os.makedirs(force_dir, exist_ok=True)

    n_samples = len(y_test)
    print(f"  Generating force plots for all {n_samples} test samples...")
    for i in range(n_samples):
        p = shap.plots.force(shap_values[i], matplotlib=False)
        lbl = int(y_test[i])
        path = os.path.join(force_dir, f"force_sample{i:03d}_label{lbl}_{TIMESTAMP}.html")
        shap.save_html(path, p)
    print(f"  {n_samples} force plots saved to: {force_dir}")

    return shap_values, top_names

"""
Central configuration for the LungRad-BM project.
Update the paths and hyperparameters here before running.
"""
import os
from datetime import datetime

# ======================== PATHS ========================
DATA_DIR = "./data"
OUTPUT_DIR = "./output"

# Input data files per cohort (Excel: col 0 = label, cols 1+ = features)
# ITR: Individual Treatment Response; PTR: Population Treatment Response
DATA_PATHS_ITR = {
    "CT":     os.path.join(DATA_DIR, "ITR_CT.xlsx"),
    "PET":    os.path.join(DATA_DIR, "ITR_PET.xlsx"),
    "PETCT":  os.path.join(DATA_DIR, "ITR_PETCT.xlsx"),
}

DATA_PATHS_PTR = {
    "CT":     os.path.join(DATA_DIR, "PTR_CT.xlsx"),
    "PET":    os.path.join(DATA_DIR, "PTR_PET.xlsx"),
    "PETCT":  os.path.join(DATA_DIR, "PTR_PETCT.xlsx"),
}

# Timestamp for run versioning
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ======================== SEEDS ========================
SEED = 42

# ======================== TABNET PARAMS ========================
TABNET_PARAMS = dict(
    seed=42,
    optimizer_fn=None,  # set at runtime (torch.optim.Adam)
    optimizer_params=dict(lr=0.02),
)
TABNET_FIT_ARGS = dict(
    max_epochs=100,
    batch_size=8,
    patience=0,
    virtual_batch_size=8,
    weights=1,
    drop_last=True,
)

# ======================== SHAP PARAMS ========================
SHAP_MAX_DISPLAY = 10
SHAP_NUM_SAMPLES = 50

# ======================== BOOTSTRAP ========================
N_BOOTSTRAP = 1000

# ======================== DL TRAINING ========================
DL_EPOCHS = 100
DL_LR = 0.02

# ======================== CROSS-VALIDATION ========================
N_FOLDS = 5
CV_SEED = 42

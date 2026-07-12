# LungRad-BM
### Non-invasive diagnosis of Brain Metastasis (BM) in Non-small Cell Lung Cancer (NSCLC) by integrating multiregional and multimodal PET/CT radiomic features

## Overview
Brain metastasis (BM) poses a major threat to patients with non-small cell lung cancer (NSCLC), making early risk evaluation a pressing need. To address this challenge, we developed a novel framework, namely **LungRad-BM**, to achieve non-invasive diagnosis of BM in NSCLC patients, by integrating multiregional and multimodal features derived from 18F-fluorodeoxyglucose positron emission tomography/computed tomography (18F-FDG PET/CT) radiomics.

Using a combined strategy of feature selection via the Least Absolute Shrinkage and Selection Operator (LASSO) followed by modeling with the TabNet deep learning architecture, we individually developed two **LungRad-BM** models for **intra-tumoral region (ITR)** and **peri-tumoral region (PTR)**, by integrating multimodal features from PET and CT.

## Model Performance
| Model | Test AUC | ACC | Sn | Sp | F1 |
|:------------------|:------------:|:------------:|:------------:|:------------:|:------------:|
| ITR | 0.9167 | 0.8947 | 0.7143 | 1.000 | 0.8333|
| PTR | 0.9048 | 0.6842 | 0.1429 | 1.0000 | 0.2500|

## Project Structure
```
├── Main/    # Main entry points for ITR and PTR pipelines
│  ├── LungRad-BM-ITR.py    # Full ITR pipeline (LASSO → TabNet → CV → SHAP → Comparison)
│  └── LungRad-BM-PTR.py    # Full PTR pipeline (LASSO → TabNet → CV → SHAP → Comparison)
├── Model/    # Core modules for training, evaluation, and analysis
│  ├── __init__.py
│  ├── Feature_selection_LASSO.py    # LASSO feature selection with 5-fold CV diagnostic plots
│  ├── Training.py    # TabNet training (7:3 split) and 5-fold cross-validation
│  ├── Evaluation.py    # Metrics, confusion matrix, ROC curves, calibration, feature importance
│  ├── Validation-5-fold-CV.py    # Standalone 5-fold CV implementation
│  ├── Comparison.py    # Classic ML (SVM/LR/RF/KNN) and DL (DeepFM/TabResNet) baselines
│  └── SHAP.py    # SHAP interpretability analysis (beeswarm, waterfall, heatmap, force)
├── config.py    # Central configuration: paths, hyperparameters, seeds
├── data_loader.py    # Data loading, standardization, and train/test splitting
├── data/    # Input Excel data files (not included; see Data Preparation)
├── output/    # Generated results: metrics, plots, CSV files
└── README.md
```

## Quick Start

### Software Requirements
Python 3.9, PyTorch 2.0, and CUDA 11.8

```bash
pip install pytorch-tabnet shap numpy pandas scikit-learn matplotlib torch openpyxl scipy
```

### Data Preparation
Place Excel files in `./data/` with the following naming convention:
- `ITR_CT.xlsx`, `ITR_PET.xlsx`, `ITR_PETCT.xlsx` — Intra-tumoral region data
- `PTR_CT.xlsx`, `PTR_PET.xlsx`, `PTR_PETCT.xlsx` — Peri-tumoral region data

Each Excel file should have: column 0 = label (0/1), columns 1+ = radiomic features.

### Usage

Run the full pipeline for ITR or PTR:
```bash
python Main/LungRad-BM-ITR.py
python Main/LungRad-BM-PTR.py
```

Each pipeline executes:
1. LASSO feature selection (5-fold CV + λ=0.1)
2. TabNet training (7:3 train/test split) + evaluation metrics + plots
3. 5-fold cross-validation
4. SHAP interpretability analysis
5. Baseline comparison (SVM, LR, RF, KNN, DeepFM, TabResNet)
6. Summary table + combined ROC curve

### Configuration

Edit `config.py` to adjust:
- `DATA_PATHS_ITR` / `DATA_PATHS_PTR` — input file paths
- `TABNET_PARAMS` / `TABNET_FIT_ARGS` — TabNet hyperparameters
- `N_FOLDS` / `CV_SEED` — cross-validation settings
- `SEED` — random seed for reproducibility

## Outputs

| Output | Description |
|---|---|
| `metrics_*.csv` | AUC, ACC, Sn, Sp, F1 for each model |
| `cv5_folds_*.csv` | Per-fold metrics (5-fold CV) |
| `cv5_summary_*.csv` | Mean ± std across folds |
| `summary_*.csv` | Model comparison table |
| `roc_*.png` | Single ROC curves |
| `Fig_4_ROC_*.png` | Combined ROC comparison with 95% CI |
| `importance_*.csv/png` | TabNet feature importance |
| `loss_*.png` | Training loss curves |
| `shap_*.png/html` | Beeswarm, waterfall, heatmap, force plots |
| `confusion_matrix_*.png` | Confusion matrix heatmap |
| `lasso_cv_*.png` | LASSO coefficient path + CV MSE |

## Modules

| Module | Key Functions |
|---|---|
| `config.py` | `DATA_PATHS_ITR`, `DATA_PATHS_PTR`, `TABNET_PARAMS`, `TABNET_FIT_ARGS`, `N_FOLDS`, `SEED` |
| `data_loader.py` | `load_data()`, `preprocess_split()` |
| `Model/Feature_selection_LASSO.py` | `lasso_cv_analysis()` |
| `Model/Training.py` | `train_tabnet()`, `cross_validate_tabnet()` |
| `Model/Evaluation.py` | `compute_metrics()`, `confusion_matrix()`, `plot_confusion_matrix()`, `plot_roc()`, `plot_roc_comparison()`, `plot_feature_importance()`, `plot_loss()`, `save_summary_table()` |
| `Model/Comparison.py` | `ML_MODELS`, `DL_MODELS`, `train_eval_ml()`, `train_eval_dl()`, `DeepFM`, `TabResNet` |
| `Model/SHAP.py` | `shap_analyze()` |

## Contact
Shaofeng Lin: linshaofeng@fjmu.edu.cn
Zengbei Yuan: ahmuyuanzengbei@163.com

## License

Research code — for academic use.

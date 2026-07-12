import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from scipy.special import expit

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    recall_score, f1_score, roc_curve
)

from config import SEED, DL_EPOCHS, DL_LR, OUTPUT_DIR, TIMESTAMP


# ============================================================
#  CLASSIC ML MODELS
# ============================================================
ML_MODELS = {
    "SVM":                 SVC(kernel="rbf", probability=True, random_state=SEED),
    "Logistic Regression": LogisticRegression(random_state=SEED),
    "Random Forest":       RandomForestClassifier(random_state=SEED),
    "KNN":                 KNeighborsClassifier(),
}


def train_eval_ml(model, name, X_train, y_train, X_test, y_test):
    """Train a classic ML model and return predictions + metrics."""
    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)

    metrics = {
        "Model": name,
        "AUC": roc_auc_score(y_test, y_score),
        "ACC": accuracy_score(y_test, y_pred),
        "Sn":  recall_score(y_test, y_pred, zero_division=0),
        "Sp":  (np.sum((y_test == 0) & (y_pred == 0))
                / max(np.sum(y_test == 0), 1)),
        "F1":  f1_score(y_test, y_pred, zero_division=0),
    }
    pd.DataFrame([metrics]).to_csv(
        os.path.join(OUTPUT_DIR, f"metrics_{name.replace(' ', '_')}_{TIMESTAMP}.csv"),
        index=False)
    return y_score, y_pred, metrics


# ============================================================
#  DEEP LEARNING MODELS
# ============================================================

class ResBlock(nn.Module):
    """Residual block: two FC layers with skip connection.
    (Paper: two 128-dimensional FC layers with skip connections)"""
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim),
        )
    def forward(self, x): return torch.relu(self.block(x) + x)


class DeepFM(nn.Module):
    """DeepFM: factorization machines + deep neural network.
    (Paper: embedding size=10, three-layer MLP with ReLU)
    Three parallel components summed for final prediction:
      (i)   Linear layer — first-order feature interactions
      (ii)  FM layer — second-order interactions via latent vectors (k=10)
      (iii) DNN — three-layer MLP (input→64→32→1) with ReLU
    """
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.V = nn.Parameter(torch.randn(input_dim, 10))
        self.dnn = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        linear_part = self.linear(x)
        inter_part = 0.5 * torch.sum(
            torch.pow(torch.matmul(x, self.V), 2)
            - torch.matmul(x ** 2, self.V ** 2), dim=1, keepdim=True)
        return linear_part + inter_part + self.dnn(x)


class TabResNet(nn.Module):
    """TabResNet: residual network for tabular radiomics data.
    (Paper: initial FC 128 units + ReLU, two residual blocks
     with skip connections, final linear output)
    """
    def __init__(self, input_dim):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU())
        self.res1 = ResBlock(128)
        self.res2 = ResBlock(128)
        self.output = nn.Linear(128, 1)
    def forward(self, x):
        x = self.layer1(x)
        x = self.res1(x)
        x = self.res2(x)
        return self.output(x)


# Available DL models
DL_MODELS = {
    "DeepFM":      DeepFM,
    "TabResNet":   TabResNet,
}


def train_eval_dl(model_class, name, X_train, y_train, X_test, y_test):
    """Train a PyTorch model and return predictions + metrics.
    Hyperparams per paper: epochs=100, batch_size=8, Adam, lr=0.02, BCE loss.
    Note: batch_size is not enforced here (full-batch by default); set via config.
    """
    input_dim = X_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(input_dim).to(device)

    criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy (paper)
    optimizer = optim.Adam(model.parameters(), lr=DL_LR)

    Xt = torch.tensor(X_train, dtype=torch.float32).to(device)
    yt = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    Xv = torch.tensor(X_test, dtype=torch.float32).to(device)

    for _ in range(DL_EPOCHS):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(Xt), yt)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(Xv).cpu().numpy().ravel()
    y_score = expit(logits)

    y_pred = (y_score >= 0.5).astype(int)

    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc_val = roc_auc_score(y_test, y_score)

    metrics = {
        "Model": name,
        "AUC": auc_val,
        "ACC": accuracy_score(y_test, y_pred),
        "Sn":  recall_score(y_test, y_pred, zero_division=0),
        "Sp":  (np.sum((y_test == 0) & (y_pred == 0))
                / max(np.sum(y_test == 0), 1)),
        "F1":  f1_score(y_test, y_pred, zero_division=0),
    }
    pd.DataFrame([metrics]).to_csv(
        os.path.join(OUTPUT_DIR, f"metrics_{name}_{TIMESTAMP}.csv"), index=False)

    return y_score, y_pred, metrics, fpr, tpr, auc_val

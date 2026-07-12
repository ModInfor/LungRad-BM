"""
Data loading, standardization, and train/test splitting.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file_path: str):
    """
    Load Excel data.
    Returns: labels (1D), features (2D), feature_names (list)
    """
    data = pd.read_excel(file_path)
    labels = data.iloc[:, 0].values
    features = data.iloc[:, 1:].values
    feature_names = data.columns[1:].tolist()
    print(f"Loaded: {data.shape[0]} samples x {data.shape[1] - 1} features "
          f"(pos={labels.sum()}, neg={len(labels) - labels.sum()})")
    return labels, features, feature_names


def preprocess_split(features, labels, test_size=0.3, seed=42):
    """
    Standardize features, then stratified train/test split.
    Returns: X_train, X_test, y_train, y_test, scaler
    """
    scaler = StandardScaler()
    features_std = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(
        features_std, labels, test_size=test_size,
        stratify=labels, random_state=seed
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, scaler

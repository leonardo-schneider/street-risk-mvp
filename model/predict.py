"""
model/predict.py — feature matrix reconstruction and SHAP utilities.

Used by api/main.py at startup to compute SHAP values for all hexagons.
"""

import numpy as np
import pandas as pd
import shap


def build_feature_matrix(df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    """
    Reconstruct the model feature matrix from a Gold table DataFrame.

    feat_cols lists every column the model was trained on (from feature_columns.json).
    Columns starting with "road_" are dummy-encoded from road_type_primary.
    All other columns are taken directly from df.

    Returns a DataFrame with columns == feat_cols, index == df.index.
    """
    base_cols  = [c for c in feat_cols if not c.startswith("road_")]
    dummy_cols = [c for c in feat_cols if c.startswith("road_")]

    X_base = df[base_cols].reset_index(drop=True)

    if dummy_cols:
        road_dummies = pd.get_dummies(df["road_type_primary"], prefix="road", drop_first=True)
        road_dummies = road_dummies.reindex(columns=dummy_cols, fill_value=0).reset_index(drop=True)
        X = pd.concat([X_base, road_dummies], axis=1)
    else:
        X = X_base

    X.index = df.index
    return X[feat_cols]


def compute_shap_values(model, X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute SHAP values for every row in X using TreeExplainer.

    Returns a DataFrame with same shape, columns, and index as X.
    Each value is the SHAP contribution of that feature for that row.
    """
    explainer = shap.TreeExplainer(model)
    values    = explainer.shap_values(X)
    return pd.DataFrame(values, columns=X.columns, index=X.index)

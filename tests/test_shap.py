"""Tests for SHAP computation in model/predict.py."""

import numpy as np
import pandas as pd
import pytest
import lightgbm as lgb


@pytest.fixture
def tiny_model_and_X():
    """Train a minimal LightGBM on synthetic data."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((50, 3)), columns=["f1", "f2", "f3"])
    y = X["f1"] * 2 + rng.standard_normal(50) * 0.1
    model = lgb.LGBMRegressor(n_estimators=10, verbose=-1)
    model.fit(X, y)
    return model, X


def test_build_feature_matrix_columns_match_feat_cols(tiny_model_and_X):
    """build_feature_matrix() returns a DataFrame whose columns match feat_cols."""
    from model.predict import build_feature_matrix

    _, X = tiny_model_and_X
    df = X.copy()
    df["h3_index"] = [f"hex_{i}" for i in range(len(df))]
    df["road_type_primary"] = "residential"
    feat_cols = ["f1", "f2", "f3"]

    result = build_feature_matrix(df, feat_cols)

    assert list(result.columns) == feat_cols
    assert len(result) == len(df)


def test_compute_shap_values_shape(tiny_model_and_X):
    """compute_shap_values() returns a DataFrame with same shape as X."""
    from model.predict import compute_shap_values

    model, X = tiny_model_and_X
    shap_df = compute_shap_values(model, X)

    assert shap_df.shape == X.shape
    assert list(shap_df.columns) == list(X.columns)


def test_compute_shap_values_index_preserved(tiny_model_and_X):
    """compute_shap_values() preserves the index of X."""
    from model.predict import compute_shap_values

    model, X = tiny_model_and_X
    X.index = [f"hex_{i}" for i in range(len(X))]
    shap_df = compute_shap_values(model, X)

    assert list(shap_df.index) == list(X.index)

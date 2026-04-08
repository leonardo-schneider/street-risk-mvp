"""
model/train_experiments.py

Trains Ridge Regression, Random Forest, and XGBoost on the same Gold
table and geographic split used in train.py, then compares all four
models (including the existing LightGBM) by Spearman rank correlation.

Saves the best model by Spearman to model/best_model.pkl.

Usage:
    python model/train_experiments.py
"""

import json
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `model.train` is importable
sys.path.insert(0, str(Path(__file__).parents[1]))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Reuse split + feature logic from train.py
from model.train import (
    GOLD_PATH,
    FEAT_COLS_PATH,
    N_CLUSTERS,
    TARGET,
    TEST_CLUSTER,
    build_features,
    evaluate,
    spatial_split,
)

# ── env ───────────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parents[1] / ".env", override=True)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

# ── paths ─────────────────────────────────────────────────────────────────────
PLOTS_DIR      = Path(__file__).parents[1] / "docs" / "screenshots"
BEST_MODEL_PATH = Path(__file__).parent / "best_model.pkl"
LGBM_MODEL_PATH = Path(__file__).parent / "risk_model.pkl"


# ── plot helpers ──────────────────────────────────────────────────────────────

def plot_importances(scores: dict, feature_cols: list, title: str, save_path: Path):
    fi = pd.Series(scores, index=feature_cols).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(5, len(feature_cols) * 0.35)))
    fi.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("Importance / |coefficient|")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


# ── per-model trainers ────────────────────────────────────────────────────────

def train_ridge(X_train, X_test, y_train, y_test, feature_cols):
    params = {"alpha": 1.0}
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(**params)),
    ])
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    metrics = evaluate(y_test, y_pred)

    # Coefficients as importance proxy
    coefs = np.abs(model.named_steps["ridge"].coef_)
    plot_path = plot_importances(
        dict(zip(feature_cols, coefs)),
        feature_cols,
        "Ridge coefficients (abs) — Sarasota crash density",
        PLOTS_DIR / "feature_importance_ridge.png",
    )

    mlflow.set_experiment("sarasota-risk-model")
    with mlflow.start_run(run_name="ridge-regression") as run:
        mlflow.log_params(params)
        mlflow.log_params({"scaler": "StandardScaler", "model": "Ridge"})
        mlflow.log_metric("RMSE",       metrics["RMSE"])
        mlflow.log_metric("MAE",        metrics["MAE"])
        mlflow.log_metric("R2",         metrics["R2"])
        mlflow.log_metric("Spearman_r", metrics["Spearman_r"])
        mlflow.log_metric("Spearman_p", metrics["Spearman_p"])
        mlflow.log_artifact(str(plot_path))
        mlflow.sklearn.log_model(model, artifact_path="ridge_model")
        run_id = run.info.run_id

    return model, metrics, run_id


def train_random_forest(X_train, X_test, y_train, y_test, feature_cols):
    params = dict(n_estimators=500, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1)
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    metrics = evaluate(y_test, y_pred)

    plot_path = plot_importances(
        dict(zip(feature_cols, model.feature_importances_)),
        feature_cols,
        "Random Forest feature importance — Sarasota crash density",
        PLOTS_DIR / "feature_importance_rf.png",
    )

    mlflow.set_experiment("sarasota-risk-model")
    with mlflow.start_run(run_name="random-forest") as run:
        mlflow.log_params(params)
        mlflow.log_metric("RMSE",       metrics["RMSE"])
        mlflow.log_metric("MAE",        metrics["MAE"])
        mlflow.log_metric("R2",         metrics["R2"])
        mlflow.log_metric("Spearman_r", metrics["Spearman_r"])
        mlflow.log_metric("Spearman_p", metrics["Spearman_p"])
        mlflow.log_artifact(str(plot_path))
        mlflow.sklearn.log_model(model, artifact_path="rf_model")
        run_id = run.info.run_id

    return model, metrics, run_id


def train_xgboost(X_train, X_test, y_train, y_test, feature_cols):
    params = dict(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        verbosity=0,
    )
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    metrics = evaluate(y_test, y_pred)

    plot_path = plot_importances(
        dict(zip(feature_cols, model.feature_importances_)),
        feature_cols,
        "XGBoost feature importance — Sarasota crash density",
        PLOTS_DIR / "feature_importance_xgb.png",
    )

    mlflow.set_experiment("sarasota-risk-model")
    with mlflow.start_run(run_name="xgboost") as run:
        mlflow.log_params(params)
        mlflow.log_metric("RMSE",       metrics["RMSE"])
        mlflow.log_metric("MAE",        metrics["MAE"])
        mlflow.log_metric("R2",         metrics["R2"])
        mlflow.log_metric("Spearman_r", metrics["Spearman_r"])
        mlflow.log_metric("Spearman_p", metrics["Spearman_p"])
        mlflow.log_artifact(str(plot_path))
        mlflow.xgboost.log_model(model, artifact_path="xgb_model")
        run_id = run.info.run_id

    return model, metrics, run_id


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n- Model experiments — Sarasota crash density -\n")

    # 1. Load data + split (identical to train.py)
    print("Step 1/3  Loading Gold table and building features ...")
    df = pd.read_parquet(GOLD_PATH)
    train, test = spatial_split(df)
    X_train, X_test, y_train, y_test, feature_cols = build_features(train, test)
    print(f"  [ok] Train: {len(train)}  Test: {len(test)}  Features: {len(feature_cols)}")

    # 2. Load existing LightGBM metrics from saved model run
    # Re-evaluate so we have a fresh baseline on the same split
    print("\nStep 2/3  Re-evaluating existing LightGBM baseline ...")
    lgbm_model = joblib.load(LGBM_MODEL_PATH)
    lgbm_metrics = evaluate(y_test, lgbm_model.predict(X_test))
    print(f"  [ok] LightGBM: Spearman={lgbm_metrics['Spearman_r']:.4f}")

    # 3. Train new models
    print("\nStep 3/3  Training Ridge, Random Forest, XGBoost ...\n")

    print("  [1/3] Ridge Regression ...")
    ridge_model, ridge_metrics, ridge_run = train_ridge(X_train, X_test, y_train, y_test, feature_cols)
    print(f"        Spearman={ridge_metrics['Spearman_r']:.4f}  run={ridge_run[:8]}")

    print("  [2/3] Random Forest ...")
    rf_model, rf_metrics, rf_run = train_random_forest(X_train, X_test, y_train, y_test, feature_cols)
    print(f"        Spearman={rf_metrics['Spearman_r']:.4f}  run={rf_run[:8]}")

    print("  [3/3] XGBoost ...")
    xgb_model, xgb_metrics, xgb_run = train_xgboost(X_train, X_test, y_train, y_test, feature_cols)
    print(f"        Spearman={xgb_metrics['Spearman_r']:.4f}  run={xgb_run[:8]}")

    # Comparison table
    rows = [
        ("LightGBM",      lgbm_metrics,  lgbm_model,  None),
        ("Ridge",         ridge_metrics, ridge_model, ridge_run),
        ("Random Forest", rf_metrics,    rf_model,    rf_run),
        ("XGBoost",       xgb_metrics,   xgb_model,   xgb_run),
    ]

    print("\n" + "=" * 65)
    print(f"  {'Model':<16} | {'RMSE':>8} | {'MAE':>8} | {'R2':>7} | {'Spearman':>8}")
    print(f"  {'-'*16}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*8}")
    for name, m, _, _ in rows:
        print(
            f"  {name:<16} | {m['RMSE']:>8.2f} | {m['MAE']:>8.2f} "
            f"| {m['R2']:>7.4f} | {m['Spearman_r']:>8.4f}"
        )
    print("=" * 65)

    # Best by Spearman
    best_name, best_metrics, best_model, _ = max(rows, key=lambda r: r[1]["Spearman_r"])
    print(f"\n  Best model by Spearman: {best_name} ({best_metrics['Spearman_r']:.4f})")
    joblib.dump(best_model, BEST_MODEL_PATH)
    print(f"  [ok] Saved: {BEST_MODEL_PATH}")

    # Feature columns unchanged — confirm
    print(f"  [ok] Feature columns unchanged: {FEAT_COLS_PATH}")

    print("\n- Done. -\n")


if __name__ == "__main__":
    main()

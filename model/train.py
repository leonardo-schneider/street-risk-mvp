"""
model/train.py

Trains a LightGBM regressor on the Gold training table to predict
crash_density (crashes per km2) per H3 hexagon.

Uses spatial train/test split (KMeans on hex centroids) to avoid
geographic leakage. Tracks everything with MLflow.

Usage:
    python model/train.py
"""

import json
import os
from pathlib import Path

import h3
import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ── env ───────────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parents[1] / ".env", override=True)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

# ── paths ─────────────────────────────────────────────────────────────────────
GOLD_PATH      = Path(__file__).parents[1] / "data" / "gold" / "training_table" / "sarasota_gold.parquet"
MODEL_PATH     = Path(__file__).parent / "risk_model.pkl"
FEAT_COLS_PATH = Path(__file__).parent / "feature_columns.json"
FI_PLOT_PATH   = Path(__file__).parents[1] / "docs" / "screenshots" / "feature_importance.png"

# ── feature config ────────────────────────────────────────────────────────────
CLIP_FEATURES = [
    "clip_heavy_traffic", "clip_poor_lighting", "clip_no_sidewalks",
    "clip_damaged_road",  "clip_clear_road",    "clip_no_signals",
    "clip_parked_cars",
]
NUMERIC_FEATURES = [
    "speed_limit_mean", "lanes_mean", "dist_to_intersection_mean", "point_count",
]
TARGET = "crash_density"

LGBM_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    min_child_samples=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

N_CLUSTERS  = 5
TEST_CLUSTER = 0


# ── spatial split ─────────────────────────────────────────────────────────────

def get_hex_centroids(df: pd.DataFrame) -> np.ndarray:
    """Return (N, 2) array of [lat, lon] for each row's h3_index."""
    coords = np.array([h3.cell_to_latlng(idx) for idx in df["h3_index"]])
    return coords


def spatial_split(df: pd.DataFrame):
    coords = get_hex_centroids(df)
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    clusters = km.fit_predict(coords)
    df = df.copy()
    df["_cluster"] = clusters

    print(f"\n  Spatial clusters (KMeans n={N_CLUSTERS} on hex centroids):")
    for c in range(N_CLUSTERS):
        n = (clusters == c).sum()
        role = "TEST " if c == TEST_CLUSTER else "train"
        print(f"    Cluster {c}: {n:>4} hexagons  [{role}]")

    train = df[df["_cluster"] != TEST_CLUSTER].drop(columns="_cluster")
    test  = df[df["_cluster"] == TEST_CLUSTER].drop(columns="_cluster")
    return train, test


# ── feature engineering ───────────────────────────────────────────────────────

def build_features(train: pd.DataFrame, test: pd.DataFrame):
    road_dummies_train = pd.get_dummies(train["road_type_primary"], prefix="road", drop_first=True)
    road_dummies_test  = pd.get_dummies(test["road_type_primary"],  prefix="road", drop_first=True)
    road_dummies_test  = road_dummies_test.reindex(columns=road_dummies_train.columns, fill_value=0)

    # Auto-detect whether to use probe feature or zero-shot CLIP features
    if "clip_risk_prob" in train.columns:
        clip_feats = ["clip_risk_prob"]
        print("  [probe mode] Using clip_risk_prob feature.")
    else:
        clip_feats = CLIP_FEATURES
        print("  [zero-shot mode] Using 7 CLIP concept features.")

    base_cols    = clip_feats + NUMERIC_FEATURES
    dummy_cols   = list(road_dummies_train.columns)
    feature_cols = base_cols + dummy_cols

    X_train = pd.concat([train[base_cols].reset_index(drop=True),
                         road_dummies_train.reset_index(drop=True)], axis=1)
    X_test  = pd.concat([test[base_cols].reset_index(drop=True),
                         road_dummies_test.reset_index(drop=True)], axis=1)

    y_train = train[TARGET].values
    y_test  = test[TARGET].values

    return X_train, X_test, y_train, y_test, feature_cols


# ── feature importance plot ───────────────────────────────────────────────────

def plot_feature_importance(model, feature_cols: list, save_path: Path):
    fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(5, len(feature_cols) * 0.35)))
    fi.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Feature importance (gain)")
    ax.set_title("LightGBM feature importance — Sarasota crash density")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [ok] Feature importance plot saved: {save_path}")


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(y_true, y_pred):
    rmse    = np.sqrt(mean_squared_error(y_true, y_pred))
    mae     = mean_absolute_error(y_true, y_pred)
    r2      = r2_score(y_true, y_pred)
    spear_r, spear_p = spearmanr(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "Spearman_r": spear_r, "Spearman_p": spear_p}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n- Model training — Sarasota crash density -\n")

    # 1. Load
    print("Step 1/6  Loading Gold table ...")
    df = pd.read_parquet(GOLD_PATH)
    print(f"  [ok] {len(df):,} hexagons loaded")

    # 2. Spatial split
    print("\nStep 2/6  Spatial train/test split ...")
    train, test = spatial_split(df)
    print(f"\n  Train: {len(train)} hexagons  |  Test: {len(test)} hexagons")

    # 3. Features
    print("\nStep 3/6  Building feature matrix ...")
    X_train, X_test, y_train, y_test, feature_cols = build_features(train, test)
    print(f"  [ok] {len(feature_cols)} features: {feature_cols}")

    # 4. Train
    print("\nStep 4/6  Training LightGBM ...")
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(X_train, y_train)
    print(f"  [ok] Training complete")

    # 5. Evaluate
    print("\nStep 5/6  Evaluating on test set ...")
    y_pred  = model.predict(X_test)
    metrics = evaluate(y_test, y_pred)

    plot_feature_importance(model, feature_cols, FI_PLOT_PATH)

    # 6. MLflow
    print("\nStep 6/6  Logging to MLflow ...")
    mlflow.set_experiment("sarasota-risk-model")

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # Params
        mlflow.log_params(LGBM_PARAMS)
        mlflow.log_params({
            "n_train":       len(train),
            "n_test":        len(test),
            "n_features":    len(feature_cols),
            "test_cluster":  TEST_CLUSTER,
            "n_clusters":    N_CLUSTERS,
            "target":        TARGET,
        })

        # Metrics
        mlflow.log_metric("RMSE",       metrics["RMSE"])
        mlflow.log_metric("MAE",        metrics["MAE"])
        mlflow.log_metric("R2",         metrics["R2"])
        mlflow.log_metric("Spearman_r", metrics["Spearman_r"])
        mlflow.log_metric("Spearman_p", metrics["Spearman_p"])

        # Artifacts
        mlflow.log_artifact(str(FI_PLOT_PATH))
        mlflow.lightgbm.log_model(model, artifact_path="lgbm_model")

        print(f"  [ok] MLflow run logged: {run_id}")

    # 7. Save model + feature columns
    joblib.dump(model, MODEL_PATH)
    print(f"  [ok] Model saved: {MODEL_PATH}")

    with open(FEAT_COLS_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"  [ok] Feature columns saved: {FEAT_COLS_PATH}")

    # Summary
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Train size   : {len(train)} hexagons")
    print(f"  Test size    : {len(test)} hexagons (cluster {TEST_CLUSTER})")
    print(f"\n  Metrics on test set:")
    print(f"    RMSE             : {metrics['RMSE']:.2f} crashes/km2")
    print(f"    MAE              : {metrics['MAE']:.2f} crashes/km2")
    print(f"    R2               : {metrics['R2']:.4f}")
    print(f"    Spearman r       : {metrics['Spearman_r']:.4f}  (p={metrics['Spearman_p']:.4f})")
    print(f"\n  MLflow run ID    : {run_id}")
    print(f"  MLflow tracking  : {os.getenv('MLFLOW_TRACKING_URI')}")
    print(f"\n  Feature columns  : {feature_cols}")
    print(f"\n  Feature importance (top 10):")
    fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    for feat, score in fi.head(10).items():
        print(f"    {feat:<32} {score:>6.0f}")
    print("=" * 60)

    print(f"\n  Prediction sample (test set, first 10):")
    sample = pd.DataFrame({
        "h3_index":    test["h3_index"].values[:10],
        "actual":      y_test[:10],
        "predicted":   y_pred[:10],
        "risk_tier":   test["risk_tier"].values[:10],
    })
    print(sample.to_string(index=False))

    print("\n- Done. -\n")
    return model, metrics, run_id


if __name__ == "__main__":
    main()

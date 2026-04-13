"""
model/train_multicity.py

Trains LightGBM on the multi-city Gold table (Sarasota + Tampa) with
three evaluation scenarios:

  1. Sarasota-only (baseline)             — spatial cluster split
  2. Multicity same-city eval             — spatial cluster split on full table
  3. Cross-city: train Sarasota → test Tampa  — true generalization

Usage:
    python model/train_multicity.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import h3
import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error

load_dotenv(dotenv_path=Path(__file__).parents[1] / ".env", override=True)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

ROOT = Path(__file__).parents[1]

MULTICITY_GOLD  = ROOT / "data" / "gold" / "training_table" / "multicity_gold.parquet"
SARASOTA_GOLD   = ROOT / "data" / "gold" / "training_table" / "sarasota_gold.parquet"

CLIP_FEATURES = [
    "clip_heavy_traffic", "clip_poor_lighting", "clip_no_sidewalks",
    "clip_damaged_road",  "clip_clear_road",    "clip_no_signals",
    "clip_parked_cars",
]
NUMERIC_FEATURES = [
    "speed_limit_mean", "lanes_mean", "dist_to_intersection_mean", "point_count",
]
POI_FEATURES = [
    "bars_count", "schools_count", "hospitals_count",
    "gas_stations_count", "fast_food_count", "traffic_signals_count",
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


# ── helpers ───────────────────────────────────────────────────────────────────

def evaluate(y_true, y_pred) -> dict:
    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae   = float(mean_absolute_error(y_true, y_pred))
    spear = float(spearmanr(y_true, y_pred)[0])
    return {"Spearman_r": spear, "RMSE": rmse, "MAE": mae}


def spatial_split_clusters(df: pd.DataFrame, n_clusters: int = 8):
    """KMeans spatial split. Returns (train_df, test_df, test_cluster)."""
    coords   = np.array([h3.cell_to_latlng(idx) for idx in df["h3_index"]])
    km       = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = km.fit_predict(coords)
    df       = df.copy()
    df["_cluster"] = clusters

    # Pick the smallest cluster as test (gives consistent results)
    sizes       = pd.Series(clusters).value_counts().sort_values()
    test_cluster = int(sizes.index[0])

    print(f"\n  Spatial clusters (KMeans n={n_clusters}):")
    for c in sorted(range(n_clusters)):
        n    = (clusters == c).sum()
        role = "TEST " if c == test_cluster else "train"
        print(f"    Cluster {c}: {n:>5} hexagons  [{role}]")

    train = df[df["_cluster"] != test_cluster].drop(columns="_cluster")
    test  = df[df["_cluster"] == test_cluster].drop(columns="_cluster")
    return train, test


def build_features(train: pd.DataFrame, test: pd.DataFrame, include_poi: bool = True):
    road_tr = pd.get_dummies(train["road_type_primary"], prefix="road", drop_first=True)
    road_te = pd.get_dummies(test["road_type_primary"],  prefix="road", drop_first=True)
    road_te = road_te.reindex(columns=road_tr.columns, fill_value=0)

    base = CLIP_FEATURES + NUMERIC_FEATURES
    if include_poi:
        poi_avail = [c for c in POI_FEATURES if c in train.columns]
        base = base + poi_avail
    dummy_cols   = list(road_tr.columns)
    feature_cols = base + dummy_cols

    X_tr = pd.concat([train[base].reset_index(drop=True),
                      road_tr.reset_index(drop=True)], axis=1)
    X_te = pd.concat([test[base].reset_index(drop=True),
                      road_te.reset_index(drop=True)], axis=1)
    return X_tr, X_te, train[TARGET].values, test[TARGET].values, feature_cols


def train_and_eval(X_tr, y_tr, X_te, y_te) -> tuple:
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(X_tr, y_tr)
    metrics = evaluate(y_te, model.predict(X_te))
    return model, metrics


# ── scenarios ─────────────────────────────────────────────────────────────────

def scenario_sarasota_only():
    """Scenario 1: Sarasota spatial split (baseline, no POI)."""
    print("\n[Scenario 1] Sarasota-only — spatial cluster split")
    df = pd.read_parquet(SARASOTA_GOLD)
    print(f"  {len(df)} hexagons loaded")
    train, test = spatial_split_clusters(df, n_clusters=5)
    X_tr, X_te, y_tr, y_te, feat_cols = build_features(train, test, include_poi=False)
    model, m = train_and_eval(X_tr, y_tr, X_te, y_te)
    print(f"  => Spearman={m['Spearman_r']:.4f}  RMSE={m['RMSE']:.2f}  MAE={m['MAE']:.2f}")
    return m, model, feat_cols


def scenario_multicity_same_city(gold: pd.DataFrame):
    """Scenario 2: Train+test on multicity table, spatial split (includes POI)."""
    print("\n[Scenario 2] Multicity — spatial split (same-city eval)")
    print(f"  {len(gold)} hexagons  ({gold['city'].value_counts().to_dict()})")
    train, test = spatial_split_clusters(gold, n_clusters=8)
    X_tr, X_te, y_tr, y_te, feat_cols = build_features(train, test, include_poi=True)
    model, m = train_and_eval(X_tr, y_tr, X_te, y_te)
    print(f"  => Spearman={m['Spearman_r']:.4f}  RMSE={m['RMSE']:.2f}  MAE={m['MAE']:.2f}")
    return m, model, feat_cols


def scenario_cross_city(gold: pd.DataFrame):
    """
    Scenario 3: Train on Sarasota only → test on Tampa only.
    True cross-city generalization.
    """
    print("\n[Scenario 3] Cross-city — train Sarasota, test Tampa")
    train = gold[gold["city"] == "sarasota"].copy()
    test  = gold[gold["city"] == "tampa"].copy()
    print(f"  Train (sarasota): {len(train)}  |  Test (tampa): {len(test)}")
    X_tr, X_te, y_tr, y_te, feat_cols = build_features(train, test, include_poi=True)
    model, m = train_and_eval(X_tr, y_tr, X_te, y_te)
    print(f"  => Spearman={m['Spearman_r']:.4f}  RMSE={m['RMSE']:.2f}  MAE={m['MAE']:.2f}")
    return m, model, feat_cols


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n- Multi-city LightGBM training -\n")

    if not MULTICITY_GOLD.exists():
        raise FileNotFoundError(
            f"Multi-city gold table not found: {MULTICITY_GOLD}\n"
            "Run `python pipeline/gold/build_gold_table.py --multicity` first."
        )

    gold = pd.read_parquet(MULTICITY_GOLD)
    print(f"  Multicity gold loaded: {len(gold):,} hexagons  "
          f"cities={gold['city'].value_counts().to_dict()}")
    print(f"  Features available: {list(gold.columns)}")

    # Run all 3 scenarios
    m1, model1, feat1 = scenario_sarasota_only()
    m2, model2, feat2 = scenario_multicity_same_city(gold)
    m3, model3, feat3 = scenario_cross_city(gold)

    # Log all to MLflow
    mlflow.set_experiment("sarasota-risk-model")

    for run_name, m, feat_cols in [
        ("multicity-sarasota-only",    m1, feat1),
        ("multicity-same-city-split",  m2, feat2),
        ("multicity-cross-city",       m3, feat3),
    ]:
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(LGBM_PARAMS)
            mlflow.log_metrics({k.lower(): v for k, v in m.items()})

    # Save best model (cross-city — most meaningful)
    best_path = ROOT / "model" / "multicity_model.pkl"
    feat_path = ROOT / "model" / "multicity_feature_columns.json"
    joblib.dump(model3, best_path)
    with open(feat_path, "w") as f:
        json.dump(feat3, f, indent=2)
    print(f"\n  [ok] Cross-city model saved: {best_path}")

    # Final comparison table
    BASELINE = 0.6664
    print("\n" + "=" * 68)
    print("  COMPARISON TABLE")
    print("=" * 68)
    print(f"  {'Model':<42} | {'Spearman':>8} | {'RMSE':>7}")
    print(f"  {'-'*42}-+-{'-'*8}-+-{'-'*7}")
    print(f"  {'LightGBM (Sarasota only, original)':<42} | {BASELINE:>8.4f} | {'80.28':>7}")
    print(f"  {'LightGBM (Sarasota only, rerun)':<42} | {m1['Spearman_r']:>8.4f} | {m1['RMSE']:>7.2f}")
    print(f"  {'LightGBM (multicity, same-city split)':<42} | {m2['Spearman_r']:>8.4f} | {m2['RMSE']:>7.2f}")
    print(f"  {'LightGBM (train Sarasota -> test Tampa)':<42} | {m3['Spearman_r']:>8.4f} | {m3['RMSE']:>7.2f}")
    print("=" * 68)

    # Feature importance for cross-city model
    fi = pd.Series(model3.feature_importances_, index=feat3).sort_values(ascending=False)
    print(f"\n  Top 12 features (cross-city model):")
    for feat, score in fi.head(12).items():
        print(f"    {feat:<36} {score:>6.0f}")

    print("\n- Done. -\n")
    return m1, m2, m3


if __name__ == "__main__":
    main()

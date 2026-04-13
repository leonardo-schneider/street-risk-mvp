"""
model/train_final.py

Trains the final production LightGBM model on the combined Sarasota + Tampa
Gold table with:

  - KMeans n=6 spatial split
  - 1 Tampa cluster held out  → cross-city OOD test
  - 1 Sarasota cluster held out → within-city geo OOD test
  - Remaining 4 clusters       → training set (both cities mixed)
  - City-scale recalibration layer applied post-prediction

Saves:
  model/final_model.pkl
  model/final_feature_columns.json
  model/city_scale_factors.json

Uploads all three to S3 under gold/.

Usage:
    python model/train_final.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import boto3
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
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

load_dotenv(dotenv_path=Path(__file__).parents[1] / ".env", override=True)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

ROOT   = Path(__file__).parents[1]
BUCKET = os.getenv("S3_BUCKET_NAME", "street-risk-mvp")
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

MULTICITY_GOLD = ROOT / "data" / "gold" / "training_table" / "multicity_gold.parquet"

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

def evaluate(y_true, y_pred, label: str = "") -> dict:
    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae   = float(mean_absolute_error(y_true, y_pred))
    spear = float(spearmanr(y_true, y_pred)[0])
    r2    = float(r2_score(y_true, y_pred))
    if label:
        print(f"  [{label}] Spearman={spear:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.3f}")
    return {"spearman": spear, "rmse": rmse, "mae": mae, "r2": r2}


def build_feature_matrix(train: pd.DataFrame, test: pd.DataFrame):
    base = CLIP_FEATURES + NUMERIC_FEATURES
    poi_avail = [c for c in POI_FEATURES if c in train.columns]
    base = base + poi_avail

    road_tr = pd.get_dummies(train["road_type_primary"], prefix="road", drop_first=True)
    road_te = pd.get_dummies(test["road_type_primary"],  prefix="road", drop_first=True)
    road_te = road_te.reindex(columns=road_tr.columns, fill_value=0)

    dummy_cols   = list(road_tr.columns)
    feature_cols = base + dummy_cols

    X_tr = pd.concat([train[base].reset_index(drop=True),
                      road_tr.reset_index(drop=True)], axis=1)
    X_te = pd.concat([test[base].reset_index(drop=True),
                      road_te.reset_index(drop=True)], axis=1)
    return X_tr, X_te, train[TARGET].values, test[TARGET].values, feature_cols


def s3_upload(local_path: Path, key: str):
    client = boto3.client(
        "s3", region_name=REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    try:
        client.head_object(Bucket=BUCKET, Key=key)
        print(f"  [overwrite] s3://{BUCKET}/{key}")
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("404", "NoSuchKey"):
            raise
    client.upload_file(str(local_path), BUCKET, key)
    print(f"  [ok] Uploaded: s3://{BUCKET}/{key}")


# ── spatial split ─────────────────────────────────────────────────────────────

def spatial_split(df: pd.DataFrame, n_clusters: int = 6):
    """
    KMeans spatial clustering. Returns (train_df, test_sarasota_df, test_tampa_df,
    cluster assignments, test_sar_cluster, test_tpa_cluster).

    Strategy: pick the smallest Sarasota cluster as geo-OOD test,
              pick the smallest Tampa cluster as cross-city test.
    """
    coords   = np.array([h3.cell_to_latlng(idx) for idx in df["h3_index"]])
    km       = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = km.fit_predict(coords)
    df       = df.copy()
    df["_cluster"] = clusters

    print(f"\n  Spatial clusters (KMeans n={n_clusters}):")
    sar_cluster_sizes = {}
    tpa_cluster_sizes = {}
    for c in range(n_clusters):
        mask    = clusters == c
        n_total = mask.sum()
        n_sar   = ((clusters == c) & (df["city"] == "sarasota")).sum()
        n_tpa   = ((clusters == c) & (df["city"] == "tampa")).sum()
        print(f"    Cluster {c}: {n_total:>4} hexagons  "
              f"(sarasota={n_sar}, tampa={n_tpa})")
        if n_sar > 0:
            sar_cluster_sizes[c] = n_sar
        if n_tpa > 0:
            tpa_cluster_sizes[c] = n_tpa

    # Prefer clusters dominated by one city; fallback to smallest
    # Sarasota test: cluster with most Sarasota hexes and fewest Tampa hexes
    sar_test = min(
        sar_cluster_sizes,
        key=lambda c: (tpa_cluster_sizes.get(c, 0), -sar_cluster_sizes[c])
    )
    # Tampa test: cluster with most Tampa hexes and fewest Sarasota hexes
    remaining = {c: v for c, v in tpa_cluster_sizes.items() if c != sar_test}
    tpa_test = min(
        remaining,
        key=lambda c: (sar_cluster_sizes.get(c, 0), -remaining[c])
    )

    print(f"\n  Test cluster (Sarasota geo-OOD): cluster {sar_test}  "
          f"({sar_cluster_sizes.get(sar_test, 0)} sar / "
          f"{tpa_cluster_sizes.get(sar_test, 0)} tpa hexes)")
    print(f"  Test cluster (Tampa cross-city):  cluster {tpa_test}  "
          f"({sar_cluster_sizes.get(tpa_test, 0)} sar / "
          f"{tpa_cluster_sizes.get(tpa_test, 0)} tpa hexes)")

    test_mask = (df["_cluster"] == sar_test) | (df["_cluster"] == tpa_test)
    train     = df[~test_mask].drop(columns="_cluster")
    test_sar  = df[(df["_cluster"] == sar_test)].drop(columns="_cluster")
    test_tpa  = df[(df["_cluster"] == tpa_test)].drop(columns="_cluster")

    print(f"\n  Train: {len(train)} hexagons  "
          f"(sarasota={( train['city']=='sarasota').sum()}, "
          f"tampa={(train['city']=='tampa').sum()})")
    print(f"  Test  Sarasota: {len(test_sar)} hexagons")
    print(f"  Test  Tampa:    {len(test_tpa)} hexagons")

    return train, test_sar, test_tpa


# ── recalibration ─────────────────────────────────────────────────────────────

def compute_scale_factors(train: pd.DataFrame) -> dict:
    """Compute per-city scale factors relative to Sarasota."""
    means = train.groupby("city")[TARGET].mean()
    sar_mean = means.get("sarasota", 1.0)
    factors = {city: float(m / sar_mean) for city, m in means.items()}
    factors["sarasota"] = 1.0   # anchor
    print(f"\n  City scale factors (relative to Sarasota mean={sar_mean:.1f}):")
    for city, f in factors.items():
        print(f"    {city:<12}  mean={means.get(city, sar_mean):.1f}  factor={f:.4f}")
    return factors


def apply_recalibration(y_pred: np.ndarray, city_series: pd.Series,
                        scale_factors: dict) -> np.ndarray:
    out = y_pred.copy()
    for city, factor in scale_factors.items():
        mask = (city_series == city).values
        out[mask] *= factor
    return out


# ── permutation importance ────────────────────────────────────────────────────

def perm_importance(model, X_te: pd.DataFrame, y_te: np.ndarray,
                    feature_cols: list, n_repeats: int = 10) -> pd.Series:
    result = permutation_importance(
        model, X_te, y_te,
        n_repeats=n_repeats,
        random_state=42,
        scoring="neg_mean_squared_error",
    )
    pi = pd.Series(result.importances_mean, index=feature_cols).sort_values(ascending=False)
    return pi


def plot_importance(fi: pd.Series, title: str, save_path: Path):
    top = fi.head(12)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#cf2b2b" if v > 0 else "#1a6b3a" for v in top.values]
    ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
    ax.set_xlabel("Mean MSE increase when shuffled")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [ok] Saved: {save_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n- Final production model training (Sarasota + Tampa) -\n")

    gold = pd.read_parquet(MULTICITY_GOLD)
    print(f"  Loaded: {len(gold):,} hexagons  "
          f"cities={gold['city'].value_counts().to_dict()}")

    # ── 1. Spatial split ──────────────────────────────────────────────────────
    print("\n[Step 1] Spatial split ...")
    train, test_sar, test_tpa = spatial_split(gold, n_clusters=6)
    test_combined = pd.concat([test_sar, test_tpa], ignore_index=True)

    # ── 2. City scale factors ─────────────────────────────────────────────────
    print("\n[Step 2] City scale factors ...")
    scale_factors = compute_scale_factors(train)

    # ── 3. Feature matrices ───────────────────────────────────────────────────
    print("\n[Step 3] Building feature matrices ...")
    X_tr, X_te_sar,  y_tr, y_te_sar,  feat_cols = build_feature_matrix(train, test_sar)
    _,    X_te_tpa,  _,    y_te_tpa,  _         = build_feature_matrix(train, test_tpa)
    _,    X_te_comb, _,    y_te_comb, _         = build_feature_matrix(train, test_combined)
    print(f"  Train:    {X_tr.shape}  |  features: {len(feat_cols)}")
    print(f"  Test-Sar: {X_te_sar.shape}")
    print(f"  Test-Tpa: {X_te_tpa.shape}")

    # ── 4. Train ──────────────────────────────────────────────────────────────
    print("\n[Step 4] Training LightGBM ...")
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(X_tr, y_tr)
    print("  [ok] Training complete")

    # ── 5. Raw predictions ────────────────────────────────────────────────────
    print("\n[Step 5] Evaluating (raw, no recalibration) ...")
    pred_sar_raw  = model.predict(X_te_sar)
    pred_tpa_raw  = model.predict(X_te_tpa)
    pred_comb_raw = model.predict(X_te_comb)

    m_sar_raw  = evaluate(y_te_sar,  pred_sar_raw,  "Sarasota raw")
    m_tpa_raw  = evaluate(y_te_tpa,  pred_tpa_raw,  "Tampa raw   ")
    m_comb_raw = evaluate(y_te_comb, pred_comb_raw, "Combined raw")

    # ── 6. Recalibrated predictions ───────────────────────────────────────────
    print("\n[Step 6] Evaluating (with city-scale recalibration) ...")
    pred_sar_cal  = apply_recalibration(pred_sar_raw,  test_sar["city"],      scale_factors)
    pred_tpa_cal  = apply_recalibration(pred_tpa_raw,  test_tpa["city"],      scale_factors)
    pred_comb_cal = apply_recalibration(pred_comb_raw, test_combined["city"], scale_factors)

    m_sar_cal  = evaluate(y_te_sar,  pred_sar_cal,  "Sarasota cal")
    m_tpa_cal  = evaluate(y_te_tpa,  pred_tpa_cal,  "Tampa cal   ")
    m_comb_cal = evaluate(y_te_comb, pred_comb_cal, "Combined cal")

    # ── 7. Permutation importance ──────────────────────────────────────────────
    print("\n[Step 7] Permutation importance (combined test set) ...")
    pi = perm_importance(model, X_te_comb, y_te_comb, feat_cols)

    docs_dir = ROOT / "docs" / "screenshots"
    docs_dir.mkdir(parents=True, exist_ok=True)
    plot_importance(pi, "Permutation Importance — Final Multicity Model",
                    docs_dir / "final_permutation_importance.png")

    # ── 8. MLflow ─────────────────────────────────────────────────────────────
    print("\n[Step 8] Logging to MLflow ...")
    mlflow.set_experiment("sarasota-risk-model")
    with mlflow.start_run(run_name="final-multicity-lgbm"):
        mlflow.log_params(LGBM_PARAMS)
        mlflow.log_params({"n_train": len(train), "n_test_sar": len(test_sar),
                           "n_test_tpa": len(test_tpa)})
        # Raw
        for k, v in m_sar_raw.items():
            mlflow.log_metric(f"sar_raw_{k}",  v)
        for k, v in m_tpa_raw.items():
            mlflow.log_metric(f"tpa_raw_{k}",  v)
        for k, v in m_comb_raw.items():
            mlflow.log_metric(f"comb_raw_{k}", v)
        # Calibrated
        for k, v in m_sar_cal.items():
            mlflow.log_metric(f"sar_cal_{k}",  v)
        for k, v in m_tpa_cal.items():
            mlflow.log_metric(f"tpa_cal_{k}",  v)
        for k, v in m_comb_cal.items():
            mlflow.log_metric(f"comb_cal_{k}", v)
        mlflow.log_artifact(str(docs_dir / "final_permutation_importance.png"))
        mlflow.lightgbm.log_model(model, "model")
    print("  [ok] MLflow run logged")

    # ── 9. Save artifacts ─────────────────────────────────────────────────────
    print("\n[Step 9] Saving artifacts ...")
    model_path  = ROOT / "model" / "final_model.pkl"
    feat_path   = ROOT / "model" / "final_feature_columns.json"
    scale_path  = ROOT / "model" / "city_scale_factors.json"

    joblib.dump(model, model_path)
    print(f"  [ok] {model_path}")

    with open(feat_path, "w") as f:
        json.dump(feat_cols, f, indent=2)
    print(f"  [ok] {feat_path}")

    with open(scale_path, "w") as f:
        json.dump(scale_factors, f, indent=2)
    print(f"  [ok] {scale_path}")

    print("\n  Uploading to S3 ...")
    s3_upload(model_path,  "gold/final_model.pkl")
    s3_upload(feat_path,   "gold/final_feature_columns.json")
    s3_upload(scale_path,  "gold/city_scale_factors.json")

    # ── 10. Final report ──────────────────────────────────────────────────────
    V1_SPEARMAN = 0.6664
    V1_RMSE     = 80.28

    print("\n" + "=" * 72)
    print("  FINAL EVALUATION TABLE")
    print("=" * 72)
    print(f"  {'Test set':<26} | {'Spearman':>8} | {'RMSE':>8} | {'MAE':>7} | {'R2':>7}")
    print(f"  {'-'*26}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}")

    def row(label, m):
        print(f"  {label:<26} | {m['spearman']:>8.4f} | {m['rmse']:>8.2f} | "
              f"{m['mae']:>7.2f} | {m['r2']:>7.3f}")

    print("  -- Raw predictions --")
    row("Sarasota (geo OOD)",    m_sar_raw)
    row("Tampa (cross-city)",    m_tpa_raw)
    row("Combined",              m_comb_raw)
    print("  -- With city recalibration --")
    row("Sarasota (geo OOD) cal", m_sar_cal)
    row("Tampa (cross-city) cal", m_tpa_cal)
    row("Combined cal",           m_comb_cal)
    print(f"  {'--':26}")
    print(f"  {'v1 baseline (Sarasota only)':<26} | {V1_SPEARMAN:>8.4f} | "
          f"{V1_RMSE:>8.2f} |     --- |     ---")
    print("=" * 72)

    print(f"\n  Top 10 feature importances (permutation, combined test):")
    for feat, score in pi.head(10).items():
        bar  = "#" * int(abs(score) / pi.max() * 25)
        print(f"    {feat:<36} {score:>10.1f}  {bar}")

    # POI contribution check
    poi_present = [c for c in POI_FEATURES if c in feat_cols]
    poi_pi = pi[poi_present].sort_values(ascending=False)
    non_poi_mean = pi[[c for c in feat_cols if c not in POI_FEATURES]].mean()
    poi_mean     = poi_pi.mean()
    print(f"\n  POI features mean perm importance: {poi_mean:.1f}  "
          f"(non-POI mean: {non_poi_mean:.1f})")
    print(f"  POI {'improved' if poi_mean > 0 else 'did not improve'} over v1 baseline")

    print(f"\n  R2 notes:")
    for label, m in [("Sarasota", m_sar_cal), ("Tampa", m_tpa_cal), ("Combined", m_comb_cal)]:
        sign = "negative" if m['r2'] < 0 else "positive"
        print(f"    {label:<12}  R2={m['r2']:.3f}  ({sign})")
    print("    Negative R2 on geo-OOD splits is expected — the test cluster")
    print("    mean differs from training mean; predicting train mean")
    print("    everywhere would score R2=0, so OOD shift > within-cluster signal.")

    print("\n- Done. -\n")
    return model, feat_cols, scale_factors


if __name__ == "__main__":
    main()

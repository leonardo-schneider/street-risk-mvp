"""
model/train_final_v4.py

Trains the v4 production LightGBM model on multicity_gold_v2.parquet,
which extends v3 with three AADT features:
  aadt_mean, aadt_max, aadt_segment_count

Same training protocol as train_final.py:
  - KMeans n=6 spatial split
  - 1 Sarasota cluster held out  → within-city geo OOD test
  - 1 Tampa cluster held out     → cross-city OOD test
  - Remaining 4 clusters         → training set

MLflow run: "multicity-lgbm-aadt-v4"

Saves:
  model/final_model_v4.pkl
  model/final_feature_columns_v4.json

Uploads to S3:
  gold/final_model_v4.pkl
  gold/final_feature_columns_v4.json

Usage:
    python model/train_final_v4.py
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

GOLD_V2 = ROOT / "data" / "gold" / "training_table" / "multicity_gold_v2.parquet"

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
AADT_FEATURES = [
    "aadt_mean", "aadt_max", "aadt_segment_count",
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

# ── historical baselines ──────────────────────────────────────────────────────
BASELINES = [
    ("v1  Sarasota only",        0.6664, 80.28,  None,   -0.594),
    ("v2  cross-city zero-shot", 0.6920, 331.18, None,   None),
    ("v3  multicity + POI",      0.6320, None,   None,    0.540),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def evaluate(y_true, y_pred, label: str = "") -> dict:
    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae   = float(mean_absolute_error(y_true, y_pred))
    spear = float(spearmanr(y_true, y_pred)[0])
    r2    = float(r2_score(y_true, y_pred))
    if label:
        print(f"  [{label}] Spearman={spear:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.3f}")
    return {"spearman": spear, "rmse": rmse, "mae": mae, "r2": r2}


def build_feature_matrix(train: pd.DataFrame, test: pd.DataFrame):
    base = (CLIP_FEATURES + NUMERIC_FEATURES
            + [c for c in POI_FEATURES  if c in train.columns]
            + [c for c in AADT_FEATURES if c in train.columns])

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
    coords   = np.array([h3.cell_to_latlng(idx) for idx in df["h3_index"]])
    km       = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = km.fit_predict(coords)
    df       = df.copy()
    df["_cluster"] = clusters

    print(f"\n  Spatial clusters (KMeans n={n_clusters}):")
    sar_cluster_sizes = {}
    tpa_cluster_sizes = {}
    for c in range(n_clusters):
        n_sar = ((clusters == c) & (df["city"] == "sarasota")).sum()
        n_tpa = ((clusters == c) & (df["city"] == "tampa")).sum()
        print(f"    Cluster {c}: {(clusters==c).sum():>4}  (sar={n_sar}, tpa={n_tpa})")
        if n_sar > 0:
            sar_cluster_sizes[c] = n_sar
        if n_tpa > 0:
            tpa_cluster_sizes[c] = n_tpa

    sar_test = min(sar_cluster_sizes,
                   key=lambda c: (tpa_cluster_sizes.get(c, 0), -sar_cluster_sizes[c]))
    remaining = {c: v for c, v in tpa_cluster_sizes.items() if c != sar_test}
    tpa_test  = min(remaining,
                    key=lambda c: (sar_cluster_sizes.get(c, 0), -remaining[c]))

    print(f"\n  Test cluster (Sarasota geo-OOD): {sar_test}  "
          f"({sar_cluster_sizes.get(sar_test,0)} sar / {tpa_cluster_sizes.get(sar_test,0)} tpa)")
    print(f"  Test cluster (Tampa cross-city):  {tpa_test}  "
          f"({sar_cluster_sizes.get(tpa_test,0)} sar / {tpa_cluster_sizes.get(tpa_test,0)} tpa)")

    test_mask = (df["_cluster"] == sar_test) | (df["_cluster"] == tpa_test)
    train    = df[~test_mask].drop(columns="_cluster")
    test_sar = df[df["_cluster"] == sar_test].drop(columns="_cluster")
    test_tpa = df[df["_cluster"] == tpa_test].drop(columns="_cluster")

    print(f"\n  Train: {len(train)}  "
          f"(sar={( train['city']=='sarasota').sum()}, tpa={( train['city']=='tampa').sum()})")
    print(f"  Test Sarasota: {len(test_sar)}")
    print(f"  Test Tampa:    {len(test_tpa)}")
    return train, test_sar, test_tpa


# ── permutation importance ────────────────────────────────────────────────────

def perm_importance(model, X_te: pd.DataFrame, y_te: np.ndarray,
                    feature_cols: list, n_repeats: int = 10) -> pd.Series:
    result = permutation_importance(
        model, X_te, y_te,
        n_repeats=n_repeats,
        random_state=42,
        scoring="neg_mean_squared_error",
    )
    return pd.Series(result.importances_mean, index=feature_cols).sort_values(ascending=False)


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
    print("\n- v4 model training: multicity + POI + AADT -\n")

    if not GOLD_V2.exists():
        raise FileNotFoundError(
            f"Gold v2 not found: {GOLD_V2}\n"
            f"Run: python pipeline/gold/build_gold_table.py --multicity --aadt"
        )

    gold = pd.read_parquet(GOLD_V2)
    print(f"  Loaded: {len(gold):,} hexagons  "
          f"cities={gold['city'].value_counts().to_dict()}  "
          f"columns={len(gold.columns)}")

    aadt_in_gold = [c for c in AADT_FEATURES if c in gold.columns]
    print(f"  AADT features present: {aadt_in_gold}")

    # ── 1. Spatial split ──────────────────────────────────────────────────────
    print("\n[Step 1] Spatial split ...")
    train, test_sar, test_tpa = spatial_split(gold, n_clusters=6)
    test_combined = pd.concat([test_sar, test_tpa], ignore_index=True)

    # ── 2. Feature matrices ───────────────────────────────────────────────────
    print("\n[Step 2] Building feature matrices ...")
    X_tr, X_te_sar,  y_tr, y_te_sar,  feat_cols = build_feature_matrix(train, test_sar)
    _,    X_te_tpa,  _,    y_te_tpa,  _         = build_feature_matrix(train, test_tpa)
    _,    X_te_comb, _,    y_te_comb, _         = build_feature_matrix(train, test_combined)
    print(f"  Train:    {X_tr.shape}  |  features: {len(feat_cols)}")
    print(f"  Test-Sar: {X_te_sar.shape}")
    print(f"  Test-Tpa: {X_te_tpa.shape}")
    print(f"  AADT features in matrix: "
          f"{[c for c in feat_cols if c.startswith('aadt')]}")

    # ── 3. Train ──────────────────────────────────────────────────────────────
    print("\n[Step 3] Training LightGBM ...")
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(X_tr, y_tr)
    print("  [ok] Training complete")

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    print("\n[Step 4] Evaluating ...")
    pred_sar  = model.predict(X_te_sar)
    pred_tpa  = model.predict(X_te_tpa)
    pred_comb = model.predict(X_te_comb)

    m_sar  = evaluate(y_te_sar,  pred_sar,  "Sarasota (geo OOD)")
    m_tpa  = evaluate(y_te_tpa,  pred_tpa,  "Tampa (cross-city)")
    m_comb = evaluate(y_te_comb, pred_comb, "Combined           ")

    # ── 5. Permutation importance ──────────────────────────────────────────────
    print("\n[Step 5] Permutation importance (combined test set) ...")
    pi = perm_importance(model, X_te_comb, y_te_comb, feat_cols)

    docs_dir = ROOT / "docs" / "screenshots"
    docs_dir.mkdir(parents=True, exist_ok=True)
    plot_importance(pi, "Permutation Importance — v4 (AADT)", docs_dir / "v4_permutation_importance.png")

    # ── 6. MLflow ─────────────────────────────────────────────────────────────
    print("\n[Step 6] Logging to MLflow ...")
    mlflow.set_experiment("sarasota-risk-model")
    with mlflow.start_run(run_name="multicity-lgbm-aadt-v4"):
        mlflow.log_params(LGBM_PARAMS)
        mlflow.log_params({
            "n_train": len(train), "n_test_sar": len(test_sar),
            "n_test_tpa": len(test_tpa), "n_features": len(feat_cols),
        })
        for k, v in m_sar.items():
            mlflow.log_metric(f"sar_{k}", v)
        for k, v in m_tpa.items():
            mlflow.log_metric(f"tpa_{k}", v)
        for k, v in m_comb.items():
            mlflow.log_metric(f"comb_{k}", v)
        mlflow.log_artifact(str(docs_dir / "v4_permutation_importance.png"))
        mlflow.lightgbm.log_model(model, "model")
    print("  [ok] MLflow run logged")

    # ── 7. Save artifacts ─────────────────────────────────────────────────────
    print("\n[Step 7] Saving artifacts ...")
    model_path = ROOT / "model" / "final_model_v4.pkl"
    feat_path  = ROOT / "model" / "final_feature_columns_v4.json"

    joblib.dump(model, model_path)
    print(f"  [ok] {model_path}")

    with open(feat_path, "w") as f:
        json.dump(feat_cols, f, indent=2)
    print(f"  [ok] {feat_path}")

    print("\n  Uploading to S3 ...")
    s3_upload(model_path, "gold/final_model_v4.pkl")
    s3_upload(feat_path,  "gold/final_feature_columns_v4.json")

    # ── 8. Full comparison table ───────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  MODEL EVOLUTION TABLE")
    print("=" * 78)
    hdr = f"  {'Model':<28} | {'Test set':<18} | {'Spearman':>8} | {'RMSE':>8} | {'R2':>7} | Features"
    print(hdr)
    print(f"  {'-'*28}-+-{'-'*18}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+---------")

    def fmt_row(label, test_lbl, spear, rmse, r2, feat_note):
        rmse_s = f"{rmse:8.2f}" if rmse is not None else "       -"
        r2_s   = f"{r2:7.3f}" if r2   is not None else "      -"
        print(f"  {label:<28} | {test_lbl:<18} | {spear:8.4f} | {rmse_s} | {r2_s} | {feat_note}")

    for (label, spear, rmse, mae, r2) in BASELINES:
        test_lbl = "Tampa (0-shot)" if "zero-shot" in label else "geo-OOD split"
        feat_note = ("CLIP+road" if "v1" in label else
                     "CLIP+road"  if "v2" in label else
                     "CLIP+road+POI")
        fmt_row(label, test_lbl, spear, rmse, r2, feat_note)

    print(f"  {'-'*28}-+-{'-'*18}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+---------")
    fmt_row("v4  multicity+POI+AADT", "Sarasota geo-OOD", m_sar["spearman"],  m_sar["rmse"],  m_sar["r2"],  "+AADT")
    fmt_row("v4  multicity+POI+AADT", "Tampa cross-city", m_tpa["spearman"],  m_tpa["rmse"],  m_tpa["r2"],  "+AADT")
    fmt_row("v4  multicity+POI+AADT", "Combined",         m_comb["spearman"], m_comb["rmse"], m_comb["r2"], "+AADT")
    print("=" * 78)

    # ── 9. Top 10 permutation importance ──────────────────────────────────────
    print(f"\n  Top 10 feature importances (permutation, combined test set):")
    print(f"  {'Feature':<36} {'MSE increase':>12}  Bar")
    print(f"  {'-'*36}  {'-'*12}  ---")
    pi_max = pi.max() if pi.max() > 0 else 1
    for feat, score in pi.head(10).items():
        bar  = "#" * int(abs(score) / pi_max * 25)
        tag  = " [AADT]" if feat.startswith("aadt") else ""
        print(f"  {feat:<36} {score:>12,.1f}  {bar}{tag}")

    aadt_pi   = pi[[c for c in feat_cols if c.startswith("aadt")]].mean()
    poi_pi    = pi[[c for c in feat_cols if c in POI_FEATURES]].mean()
    other_pi  = pi[[c for c in feat_cols if not c.startswith("aadt") and c not in POI_FEATURES]].mean()
    print(f"\n  Mean permutation importance by group:")
    print(f"    AADT features    : {aadt_pi:>10,.1f}")
    print(f"    POI features     : {poi_pi:>10,.1f}")
    print(f"    CLIP + road      : {other_pi:>10,.1f}")

    print("\n- Done. -\n")
    return model, feat_cols


if __name__ == "__main__":
    main()

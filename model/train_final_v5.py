"""
model/train_final_v5.py

Trains the v5 model on Sarasota + Tampa, tests zero-shot on Orlando.

Gold table: multicity_gold_v3.parquet (1,220 hexagons)
Split:
  Train: all Sarasota + Tampa hexagons (806)
  Test:  all Orlando hexagons          (414) — never seen during training

MLflow run: "multicity-lgbm-orlando-v5"

Saves:
  model/final_model_v5.pkl
  model/final_feature_columns_v5.json
  gold/final_model_v5.pkl  (S3)
  gold/final_feature_columns_v5.json  (S3)
"""

import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import boto3, h3, joblib, lightgbm as lgb
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow, mlflow.lightgbm
import numpy as np, pandas as pd
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

load_dotenv(dotenv_path=Path(__file__).parents[1] / ".env", override=True)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

ROOT   = Path(__file__).parents[1]
BUCKET = os.getenv("S3_BUCKET_NAME", "street-risk-mvp")
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

GOLD_V3 = ROOT / "data" / "gold" / "training_table" / "multicity_gold_v3.parquet"

CLIP_FEATURES    = ["clip_heavy_traffic","clip_poor_lighting","clip_no_sidewalks",
                    "clip_damaged_road","clip_clear_road","clip_no_signals","clip_parked_cars"]
NUMERIC_FEATURES = ["speed_limit_mean","lanes_mean","dist_to_intersection_mean","point_count"]
POI_FEATURES     = ["bars_count","schools_count","hospitals_count",
                    "gas_stations_count","fast_food_count","traffic_signals_count"]
AADT_FEATURES    = ["aadt_mean","aadt_max","aadt_segment_count"]
TARGET           = "crash_density"

LGBM_PARAMS = dict(n_estimators=500, learning_rate=0.05, max_depth=6,
                   num_leaves=31, min_child_samples=5, subsample=0.8,
                   colsample_bytree=0.8, random_state=42, verbose=-1)

BASELINES = [
    ("v1  Sarasota only",       "Sara geo-OOD",    0.6664, 80.28,  -0.594),
    ("v2  cross-city 0-shot",   "Tampa (0-shot)",   0.6920, 331.18,  None),
    ("v3  multicity+POI",       "Sara+Tampa OOD",   0.6320,  None,   0.540),
    ("v4  multicity+POI+AADT",  "Tampa cross-city", 0.7920, 247.06,  0.628),
]


def evaluate(y_true, y_pred, label=""):
    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae   = float(mean_absolute_error(y_true, y_pred))
    spear = float(spearmanr(y_true, y_pred)[0])
    r2    = float(r2_score(y_true, y_pred))
    if label:
        print(f"  [{label}] Spearman={spear:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.3f}")
    return {"spearman": spear, "rmse": rmse, "mae": mae, "r2": r2}


def build_feature_matrix(train, test):
    base = (CLIP_FEATURES + NUMERIC_FEATURES
            + [c for c in POI_FEATURES  if c in train.columns]
            + [c for c in AADT_FEATURES if c in train.columns])
    road_tr = pd.get_dummies(train["road_type_primary"], prefix="road", drop_first=True)
    road_te = pd.get_dummies(test["road_type_primary"],  prefix="road", drop_first=True)
    road_te = road_te.reindex(columns=road_tr.columns, fill_value=0)
    feat_cols = base + list(road_tr.columns)
    X_tr = pd.concat([train[base].reset_index(drop=True), road_tr.reset_index(drop=True)], axis=1)
    X_te = pd.concat([test[base].reset_index(drop=True),  road_te.reset_index(drop=True)], axis=1)
    return X_tr, X_te, train[TARGET].values, test[TARGET].values, feat_cols


def s3_upload(local_path, key):
    client = boto3.client("s3", region_name=REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
    try:
        client.head_object(Bucket=BUCKET, Key=key)
        print(f"  [overwrite] s3://{BUCKET}/{key}")
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("404","NoSuchKey"):
            raise
    client.upload_file(str(local_path), BUCKET, key)
    print(f"  [ok] Uploaded: s3://{BUCKET}/{key}")


def perm_importance(model, X_te, y_te, feat_cols, n_repeats=10):
    result = permutation_importance(model, X_te, y_te, n_repeats=n_repeats,
                                    random_state=42, scoring="neg_mean_squared_error")
    return pd.Series(result.importances_mean, index=feat_cols).sort_values(ascending=False)


def plot_importance(fi, title, save_path):
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


def main():
    print("\n- v5 model: train Sarasota+Tampa, test Orlando zero-shot -\n")

    gold = pd.read_parquet(GOLD_V3)
    print(f"  Loaded: {len(gold):,} hexagons  "
          f"cities={gold['city'].value_counts().to_dict()}")

    # Hard city split — no spatial clustering needed
    train = gold[gold["city"].isin(["sarasota","tampa"])].copy()
    test  = gold[gold["city"] == "orlando"].copy()
    print(f"\n  Train: {len(train)} hexagons  "
          f"(sarasota={( train['city']=='sarasota').sum()}, "
          f"tampa={( train['city']=='tampa').sum()})")
    print(f"  Test (Orlando zero-shot): {len(test)} hexagons")

    # Feature matrices
    print("\n[Step 1] Building feature matrices ...")
    X_tr, X_te, y_tr, y_te, feat_cols = build_feature_matrix(train, test)
    print(f"  Train: {X_tr.shape}  |  Test: {X_te.shape}  |  Features: {len(feat_cols)}")
    print(f"  AADT features: {[c for c in feat_cols if c.startswith('aadt')]}")

    # Train
    print("\n[Step 2] Training LightGBM on Sarasota + Tampa ...")
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(X_tr, y_tr)
    print("  [ok] Training complete")

    # Evaluate on Orlando (zero-shot)
    print("\n[Step 3] Zero-shot evaluation on Orlando ...")
    pred = model.predict(X_te)
    m    = evaluate(y_te, pred, "Orlando zero-shot")

    # Also evaluate on training cities for reference
    X_tr_eval, _, y_tr_eval, _, _ = build_feature_matrix(train, train)
    pred_train = model.predict(X_tr_eval)
    m_train = evaluate(y_tr_eval, pred_train, "Train (Sara+Tampa)")

    # Per-city breakdown within training set
    for city in ["sarasota","tampa"]:
        mask = train["city"] == city
        city_df = train[mask].copy()
        _, X_city, _, y_city, _ = build_feature_matrix(train, city_df)
        pred_city = model.predict(X_city)
        evaluate(y_city, pred_city, f"  {city.title()} (train city)")

    # Permutation importance on Orlando test
    print("\n[Step 4] Permutation importance on Orlando test set ...")
    pi = perm_importance(model, X_te, y_te, feat_cols)
    docs_dir = ROOT / "docs" / "screenshots"
    docs_dir.mkdir(parents=True, exist_ok=True)
    plot_importance(pi, "Permutation Importance — v5 (Orlando zero-shot)",
                    docs_dir / "v5_permutation_importance.png")

    # MLflow
    print("\n[Step 5] Logging to MLflow ...")
    mlflow.set_experiment("sarasota-risk-model")
    with mlflow.start_run(run_name="multicity-lgbm-orlando-v5"):
        mlflow.log_params(LGBM_PARAMS)
        mlflow.log_params({"n_train": len(train), "n_test_orlando": len(test),
                           "n_features": len(feat_cols), "test_city": "orlando"})
        for k, v in m.items():
            mlflow.log_metric(f"orlando_{k}", v)
        mlflow.log_artifact(str(docs_dir / "v5_permutation_importance.png"))
        mlflow.lightgbm.log_model(model, "model")
    print("  [ok] MLflow run logged")

    # Save artifacts
    print("\n[Step 6] Saving artifacts ...")
    model_path = ROOT / "model" / "final_model_v5.pkl"
    feat_path  = ROOT / "model" / "final_feature_columns_v5.json"
    joblib.dump(model, model_path)
    print(f"  [ok] {model_path}")
    with open(feat_path, "w") as f:
        json.dump(feat_cols, f, indent=2)
    print(f"  [ok] {feat_path}")
    s3_upload(model_path, "gold/final_model_v5.pkl")
    s3_upload(feat_path,  "gold/final_feature_columns_v5.json")

    # Final comparison table
    print("\n" + "="*80)
    print("  MODEL EVOLUTION TABLE")
    print("="*80)
    hdr = f"  {'Model':<28} | {'Test':<18} | {'Spearman':>8} | {'RMSE':>8} | {'R2':>7} | Notes"
    print(hdr)
    print(f"  {'-'*28}-+-{'-'*18}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-------")

    def row(label, test_lbl, spear, rmse, r2, note=""):
        rmse_s = f"{rmse:8.2f}" if rmse  is not None else "       -"
        r2_s   = f"{r2:7.3f}"   if r2    is not None else "      -"
        print(f"  {label:<28} | {test_lbl:<18} | {spear:8.4f} | {rmse_s} | {r2_s} | {note}")

    for (label, test_lbl, spear, rmse, r2) in BASELINES:
        row(label, test_lbl, spear, rmse, r2)
    print(f"  {'-'*28}-+-{'-'*18}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-------")
    row("v5  Sara+Tampa -> Orlando", "Orlando (0-shot)",
        m["spearman"], m["rmse"], m["r2"], "NEW")
    print("="*80)

    # Top 10 features
    print(f"\n  Top 10 features (permutation importance, Orlando test set):")
    print(f"  {'Feature':<36} {'MSE increase':>12}  Bar")
    print(f"  {'-'*36}  {'-'*12}  ---")
    pi_max = pi.max() if pi.max() > 0 else 1
    for feat, score in pi.head(10).items():
        bar = "#" * int(abs(score) / pi_max * 25)
        tag = " [AADT]" if feat.startswith("aadt") else " [POI]" if feat in POI_FEATURES else ""
        print(f"  {feat:<36} {score:>12,.1f}  {bar}{tag}")

    aadt_mean_pi  = pi[[c for c in feat_cols if c.startswith("aadt")]].mean()
    poi_mean_pi   = pi[[c for c in feat_cols if c in POI_FEATURES]].mean()
    other_mean_pi = pi[[c for c in feat_cols if not c.startswith("aadt") and c not in POI_FEATURES]].mean()
    print(f"\n  Mean importance by group:")
    print(f"    AADT : {aadt_mean_pi:>10,.1f}")
    print(f"    POI  : {poi_mean_pi:>10,.1f}")
    print(f"    Other: {other_mean_pi:>10,.1f}")

    # Generalization verdict
    print(f"\n  Generalization verdict:")
    v4_spearman = 0.7920
    delta = m['spearman'] - v4_spearman
    print(f"    v4 Tampa zero-shot Spearman : {v4_spearman:.4f}")
    print(f"    v5 Orlando zero-shot Spearman: {m['spearman']:.4f}  ({delta:+.4f} vs v4)")
    if m['spearman'] >= 0.70:
        print("    RESULT: Strong generalization — model transfers to Orlando.")
    elif m['spearman'] >= 0.55:
        print("    RESULT: Moderate generalization — useful signal on unseen city.")
    else:
        print("    RESULT: Weak generalization — city-specific retraining needed.")

    print("\n- Done. -\n")
    return model, feat_cols, m


if __name__ == "__main__":
    main()

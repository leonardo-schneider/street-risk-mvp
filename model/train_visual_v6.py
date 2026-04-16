"""
model/train_visual_v6.py  — RESEARCH EXPERIMENT, NOT PRODUCTION

Visual-only mode ablation: how well does the model rank Orlando
risk when AADT features are completely removed?

Experiment design
-----------------
  Train : Sarasota + Tampa  (same as v5)
  Test  : Orlando zero-shot (same as v5)
  Target: crash_density     (same as v5)

Three feature sets:
  v6a  CLIP + OSM road features + POI   (no AADT)
  v6b  CLIP + OSM road features only    (no AADT, no POI)
  v6c  CLIP only                        (7 zero-shot scores)

Production model v5 (Spearman 0.878) is the baseline.
This script does NOT touch any production artifact.

Saves model/visual_model_v6.pkl ONLY if v6a Spearman > 0.60.

Run:
    python model/train_visual_v6.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

load_dotenv(dotenv_path=Path(__file__).parents[1] / ".env", override=True)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

ROOT    = Path(__file__).parents[1]
GOLD_V3 = ROOT / "data" / "gold" / "training_table" / "multicity_gold_v3.parquet"

# ── Feature groups ────────────────────────────────────────────────────────────

CLIP_FEATURES = [
    "clip_heavy_traffic", "clip_poor_lighting", "clip_no_sidewalks",
    "clip_damaged_road",  "clip_clear_road",    "clip_no_signals",
    "clip_parked_cars",
]
OSM_FEATURES = [
    "speed_limit_mean", "lanes_mean", "dist_to_intersection_mean", "point_count",
]
POI_FEATURES = [
    "bars_count", "schools_count", "hospitals_count",
    "gas_stations_count", "fast_food_count", "traffic_signals_count",
]
TARGET = "crash_density"

# v5 baseline (never retrained here — read-only reference)
V5_RESULTS = {"spearman": 0.878, "rmse": 323.9, "r2": 0.577}

LGBM_PARAMS = dict(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    num_leaves=31, min_child_samples=5, subsample=0.8,
    colsample_bytree=0.8, random_state=42, verbose=-1,
)

SAVE_THRESHOLD = 0.60


# ── Helpers ───────────────────────────────────────────────────────────────────

def evaluate(y_true, y_pred, label=""):
    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae   = float(mean_absolute_error(y_true, y_pred))
    spear = float(spearmanr(y_true, y_pred)[0])
    r2    = float(r2_score(y_true, y_pred))
    if label:
        print(f"  [{label}]  Spearman={spear:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.3f}")
    return {"spearman": spear, "rmse": rmse, "mae": mae, "r2": r2}


def build_X(df_train, df_test, base_cols):
    """
    Combine base numeric/CLIP/POI columns with road_type_primary one-hot dummies.
    Returns (X_train, X_test, feature_col_list).
    """
    road_tr = pd.get_dummies(df_train["road_type_primary"], prefix="road", drop_first=True)
    road_te = pd.get_dummies(df_test["road_type_primary"],  prefix="road", drop_first=True)
    road_te = road_te.reindex(columns=road_tr.columns, fill_value=0)

    feat_cols = base_cols + list(road_tr.columns)

    X_tr = pd.concat(
        [df_train[base_cols].reset_index(drop=True), road_tr.reset_index(drop=True)],
        axis=1,
    )
    X_te = pd.concat(
        [df_test[base_cols].reset_index(drop=True), road_te.reset_index(drop=True)],
        axis=1,
    )
    return X_tr, X_te, feat_cols


def build_X_clip_only(df_train, df_test):
    """CLIP-only: no road_type dummies needed."""
    return (
        df_train[CLIP_FEATURES].reset_index(drop=True),
        df_test[CLIP_FEATURES].reset_index(drop=True),
        CLIP_FEATURES,
    )


def plot_importance(model, feat_cols, title, save_path):
    imp = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
    top = imp.head(12)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top.index[::-1], top.values[::-1], color="#cf2b2b")
    ax.set_xlabel("Gain importance")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [ok] Saved: {save_path}")


# ── Sub-experiment runner ─────────────────────────────────────────────────────

def run_variant(name, run_name, X_tr, X_te, y_tr, y_te, feat_cols):
    """Train one LightGBM variant, log to MLflow, return metrics dict."""
    print(f"\n[{name}]  features={len(feat_cols)}  train={len(X_tr)}  test={len(X_te)}")

    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(X_tr, y_tr)

    pred = model.predict(X_te)
    m    = evaluate(y_te, pred, label=name)

    mlflow.set_experiment("sarasota-risk-model")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(LGBM_PARAMS)
        mlflow.log_params({
            "variant":        name,
            "n_features":     len(feat_cols),
            "n_train":        len(X_tr),
            "n_test_orlando": len(X_te),
            "feature_groups": run_name,
        })
        for k, v in m.items():
            mlflow.log_metric(f"orlando_{k}", v)
        mlflow.lightgbm.log_model(model, "model")
    print(f"  [ok] MLflow run '{run_name}' logged")

    return model, m


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("  v6 VISUAL-ONLY ABLATION  (research experiment — not production)")
    print("=" * 70)

    # Load gold table
    gold = pd.read_parquet(GOLD_V3)
    print(f"\n  Gold v3: {len(gold):,} hexagons  "
          f"cities={gold['city'].value_counts().to_dict()}")

    train = gold[gold["city"].isin(["sarasota", "tampa"])].copy()
    test  = gold[gold["city"] == "orlando"].copy()
    print(f"  Train: {len(train)} (Sarasota+Tampa)  |  Test: {len(test)} (Orlando zero-shot)")

    y_tr = train[TARGET].values
    y_te = test[TARGET].values

    docs_dir = ROOT / "docs" / "screenshots"
    docs_dir.mkdir(parents=True, exist_ok=True)

    # ── v6a: CLIP + OSM + POI (no AADT) ──────────────────────────────────────
    base_a = [c for c in CLIP_FEATURES + OSM_FEATURES + POI_FEATURES
              if c in train.columns]
    X_tr_a, X_te_a, feat_a = build_X(train, test, base_a)
    model_a, m_a = run_variant(
        "v6a  CLIP+OSM+POI", "visual-v6a", X_tr_a, X_te_a, y_tr, y_te, feat_a,
    )
    plot_importance(model_a, feat_a, "Feature Importance — v6a (CLIP+OSM+POI)",
                    docs_dir / "v6a_importance.png")

    # ── v6b: CLIP + OSM only (no POI, no AADT) ───────────────────────────────
    base_b = [c for c in CLIP_FEATURES + OSM_FEATURES if c in train.columns]
    X_tr_b, X_te_b, feat_b = build_X(train, test, base_b)
    _, m_b = run_variant(
        "v6b  CLIP+OSM", "visual-v6b", X_tr_b, X_te_b, y_tr, y_te, feat_b,
    )

    # ── v6c: CLIP only (7 zero-shot scores) ──────────────────────────────────
    X_tr_c, X_te_c, feat_c = build_X_clip_only(train, test)
    _, m_c = run_variant(
        "v6c  CLIP only", "visual-v6c", X_tr_c, X_te_c, y_tr, y_te, feat_c,
    )

    # ── Comparison table ──────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  VISUAL-ONLY ABLATION — Orlando zero-shot test")
    print("=" * 78)
    hdr = (f"  {'Model':<8} {'Features':<22} {'Spearman':>9} {'RMSE':>8} "
           f"{'R2':>7} {'vs v5':>8}")
    print(hdr)
    print(f"  {'-'*8}-+-{'-'*22}-+-{'-'*9}-+-{'-'*8}-+-{'-'*7}-+-{'-'*8}")

    def row(label, feat_desc, m, is_baseline=False):
        delta = "" if is_baseline else f"{m['spearman'] - V5_RESULTS['spearman']:+.4f}"
        tag   = "  (baseline)" if is_baseline else ""
        print(f"  {label:<8} {feat_desc:<22} {m['spearman']:>9.4f} "
              f"{m['rmse']:>8.1f} {m['r2']:>7.3f} {delta:>8}{tag}")

    row("v5",  "All + AADT",         V5_RESULTS, is_baseline=True)
    print(f"  {'-'*8}-+-{'-'*22}-+-{'-'*9}-+-{'-'*8}-+-{'-'*7}-+-{'-'*8}")
    row("v6a", "CLIP+OSM+POI",       m_a)
    row("v6b", "CLIP+OSM",           m_b)
    row("v6c", "CLIP only",          m_c)
    print("=" * 78)

    # ── AADT lift analysis ────────────────────────────────────────────────────
    lift_aadt = V5_RESULTS["spearman"] - m_a["spearman"]
    lift_poi  = m_a["spearman"] - m_b["spearman"]
    lift_osm  = m_b["spearman"] - m_c["spearman"]
    lift_clip = m_c["spearman"]

    print(f"\n  Spearman contribution breakdown (Orlando zero-shot):")
    print(f"    CLIP baseline          : {lift_clip:+.4f}")
    print(f"    OSM lift (v6c -> v6b)  : {lift_osm:+.4f}")
    print(f"    POI lift (v6b -> v6a)  : {lift_poi:+.4f}")
    print(f"    AADT lift (v6a -> v5)  : {lift_aadt:+.4f}")
    print(f"    --------------------------------")
    print(f"    Total (v5 Spearman)    :  {V5_RESULTS['spearman']:.4f}")

    # ── Visual-only sufficiency verdict ──────────────────────────────────────
    print(f"\n  Visual-only verdict (v6a, no AADT):")
    if m_a["spearman"] >= SAVE_THRESHOLD:
        verdict = (
            f"    Spearman {m_a['spearman']:.4f} >= {SAVE_THRESHOLD} threshold — "
            f"visual signal sufficient for cold-start use."
        )
    else:
        verdict = (
            f"    Spearman {m_a['spearman']:.4f} < {SAVE_THRESHOLD} threshold — "
            f"Visual signal insufficient for production use."
        )
    print(verdict)

    # ── Save visual model only if threshold met ───────────────────────────────
    if m_a["spearman"] >= SAVE_THRESHOLD:
        out_path = ROOT / "model" / "visual_model_v6.pkl"
        joblib.dump(model_a, out_path)
        feat_path = ROOT / "model" / "visual_feature_columns_v6.json"
        with open(feat_path, "w") as f:
            json.dump(feat_a, f, indent=2)
        print(f"\n  [saved] {out_path.name}  ({len(feat_a)} features, research only)")
        print(f"  [saved] {feat_path.name}")
        print("  NOTE: v5 remains the production model. This file is for research.")
    else:
        print("\n  [skipped] No model file saved (threshold not met).")

    print("\n  [done]\n")
    return {"v6a": m_a, "v6b": m_b, "v6c": m_c}


if __name__ == "__main__":
    main()

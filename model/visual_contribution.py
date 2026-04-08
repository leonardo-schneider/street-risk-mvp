"""
model/visual_contribution.py

Answers: "How much does the visual (CLIP) signal contribute vs
road structure?"

Trains three LightGBM variants on the same geographic split:
  A — CLIP features only
  B — Structural features only
  C — Full model (all features)

Also runs permutation importance on the full model to measure the
true per-feature contribution to Spearman rank correlation.

Usage:
    python model/visual_contribution.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from model.train import (
    GOLD_PATH,
    LGBM_PARAMS,
    TARGET,
    build_features,
    evaluate,
    spatial_split,
)

PLOTS_DIR = Path(__file__).parents[1] / "docs" / "screenshots"

CLIP_FEATURES = [
    "clip_heavy_traffic", "clip_poor_lighting", "clip_no_sidewalks",
    "clip_damaged_road",  "clip_clear_road",    "clip_no_signals",
    "clip_parked_cars",
]

STRUCTURAL_NUMERIC = [
    "speed_limit_mean", "lanes_mean", "dist_to_intersection_mean", "point_count",
]


# ── feature subsets ───────────────────────────────────────────────────────────

def build_clip_only(train, test):
    X_train = train[CLIP_FEATURES].reset_index(drop=True)
    X_test  = test[CLIP_FEATURES].reset_index(drop=True)
    y_train = train[TARGET].values
    y_test  = test[TARGET].values
    return X_train, X_test, y_train, y_test, CLIP_FEATURES


def build_structural_only(train, test):
    road_dummies_train = pd.get_dummies(train["road_type_primary"], prefix="road", drop_first=True)
    road_dummies_test  = pd.get_dummies(test["road_type_primary"],  prefix="road", drop_first=True)
    road_dummies_test  = road_dummies_test.reindex(columns=road_dummies_train.columns, fill_value=0)

    dummy_cols   = list(road_dummies_train.columns)
    feature_cols = STRUCTURAL_NUMERIC + dummy_cols

    X_train = pd.concat([train[STRUCTURAL_NUMERIC].reset_index(drop=True),
                         road_dummies_train.reset_index(drop=True)], axis=1)
    X_test  = pd.concat([test[STRUCTURAL_NUMERIC].reset_index(drop=True),
                         road_dummies_test.reset_index(drop=True)], axis=1)

    y_train = train[TARGET].values
    y_test  = test[TARGET].values
    return X_train, X_test, y_train, y_test, feature_cols


# ── permutation importance ────────────────────────────────────────────────────

def permutation_importance(model, X_test: pd.DataFrame, y_test: np.ndarray,
                           n_repeats: int = 10) -> pd.Series:
    """
    Shuffle each feature n_repeats times, measure mean Spearman drop.
    Positive drop = feature helps; negative = feature hurts (noise).
    """
    baseline_r, _ = spearmanr(y_test, model.predict(X_test))
    drops = {}

    for col in X_test.columns:
        col_drops = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            X_perm[col] = np.random.permutation(X_perm[col].values)
            r_perm, _ = spearmanr(y_test, model.predict(X_perm))
            col_drops.append(baseline_r - r_perm)
        drops[col] = float(np.mean(col_drops))

    return pd.Series(drops).sort_values(ascending=False)


def plot_permutation_importance(perm_imp: pd.Series, save_path: Path):
    fig, ax = plt.subplots(figsize=(9, max(5, len(perm_imp) * 0.38)))
    colors = ["#d73027" if v >= 0 else "#4575b4" for v in perm_imp.sort_values().values]
    perm_imp.sort_values().plot(kind="barh", ax=ax, color=colors)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Mean Spearman drop when feature is shuffled")
    ax.set_title("Permutation importance — Full model\n(higher = feature matters more)")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [ok] Plot saved: {save_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)
    print("\n- Visual vs Structural contribution analysis -\n")

    # Load + split
    print("Loading Gold table and splitting ...")
    df = pd.read_parquet(GOLD_PATH)
    train, test = spatial_split(df)
    print(f"  Train: {len(train)}  |  Test: {len(test)}\n")

    params = {k: v for k, v in LGBM_PARAMS.items()}  # copy

    # Model A — CLIP only
    print("Training Model A: CLIP only ...")
    Xa_tr, Xa_te, ya_tr, ya_te, cols_a = build_clip_only(train, test)
    model_a = lgb.LGBMRegressor(**params)
    model_a.fit(Xa_tr, ya_tr)
    metrics_a = evaluate(ya_te, model_a.predict(Xa_te))
    print(f"  Spearman={metrics_a['Spearman_r']:.4f}  RMSE={metrics_a['RMSE']:.2f}")

    # Model B — Structural only
    print("Training Model B: Structural only ...")
    Xb_tr, Xb_te, yb_tr, yb_te, cols_b = build_structural_only(train, test)
    model_b = lgb.LGBMRegressor(**params)
    model_b.fit(Xb_tr, yb_tr)
    metrics_b = evaluate(yb_te, model_b.predict(Xb_te))
    print(f"  Spearman={metrics_b['Spearman_r']:.4f}  RMSE={metrics_b['RMSE']:.2f}")

    # Model C — Full model (re-train for consistency)
    print("Training Model C: Full model ...")
    Xc_tr, Xc_te, yc_tr, yc_te, cols_c = build_features(train, test)
    model_c = lgb.LGBMRegressor(**params)
    model_c.fit(Xc_tr, yc_tr)
    metrics_c = evaluate(yc_te, model_c.predict(Xc_te))
    print(f"  Spearman={metrics_c['Spearman_r']:.4f}  RMSE={metrics_c['RMSE']:.2f}")

    # Comparison table
    print("\n" + "=" * 72)
    print(f"  {'Feature set':<18} | {'Spearman':>8} | {'RMSE':>8} | {'MAE':>8} | What this means")
    print(f"  {'-'*18}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+------------------")
    rows = [
        ("CLIP only",       metrics_a, "Pure visual signal (Street View)"),
        ("Structural only", metrics_b, "Pure road geometry + speed"),
        ("Full model",      metrics_c, "Combined — best ranking"),
    ]
    for name, m, meaning in rows:
        print(
            f"  {name:<18} | {m['Spearman_r']:>8.4f} | {m['RMSE']:>8.2f} "
            f"| {m['MAE']:>8.2f} | {meaning}"
        )
    print("=" * 72)

    # Quantify CLIP's marginal lift
    lift = metrics_c["Spearman_r"] - metrics_b["Spearman_r"]
    struct_pct = metrics_b["Spearman_r"] / metrics_c["Spearman_r"] * 100
    clip_pct   = metrics_a["Spearman_r"] / metrics_c["Spearman_r"] * 100
    print(f"\n  CLIP marginal lift over structural alone: +{lift:.4f} Spearman")
    print(f"  Structural signal explains ~{struct_pct:.1f}% of full-model Spearman")
    print(f"  CLIP-only signal is ~{clip_pct:.1f}% as predictive as the full model")

    # Permutation importance on full model
    print(f"\nRunning permutation importance on full model (10 repeats per feature) ...")
    perm_imp = permutation_importance(model_c, Xc_te, yc_te, n_repeats=10)

    print(f"\n  Permutation importance (Spearman drop when shuffled):")
    print(f"  {'Feature':<32} | {'Spearman drop':>13} | Signal type")
    print(f"  {'-'*32}-+-{'-'*13}-+-----------")
    for feat, drop in perm_imp.items():
        sig = "CLIP" if feat.startswith("clip_") else "Structural"
        print(f"  {feat:<32} | {drop:>+13.4f} | {sig}")

    # Aggregate by signal type
    clip_imp   = perm_imp[[c for c in perm_imp.index if c.startswith("clip_")]].sum()
    struct_imp = perm_imp[[c for c in perm_imp.index if not c.startswith("clip_")]].sum()
    total_imp  = clip_imp + struct_imp
    print(f"\n  Aggregate permutation importance:")
    print(f"    CLIP features      : {clip_imp:+.4f}  ({100*clip_imp/total_imp:.1f}% of total)")
    print(f"    Structural features: {struct_imp:+.4f}  ({100*struct_imp/total_imp:.1f}% of total)")

    # Plot
    plot_permutation_importance(perm_imp, PLOTS_DIR / "permutation_importance.png")

    print("\n- Done. -\n")


if __name__ == "__main__":
    main()

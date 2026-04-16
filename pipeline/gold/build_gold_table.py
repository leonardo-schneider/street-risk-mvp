"""
pipeline/gold/build_gold_table.py

Assembles Gold training tables by joining Silver layers.

Single-city mode (default / --city sarasota|tampa):
  Outputs: data/gold/training_table/{city}_gold.parquet

Multi-city mode (--multicity):
  Stacks Sarasota + Tampa, adds `city` column, includes POI features.
  Outputs: data/gold/training_table/multicity_gold.parquet

Usage:
    python pipeline/gold/build_gold_table.py                    # sarasota (default)
    python pipeline/gold/build_gold_table.py --city tampa
    python pipeline/gold/build_gold_table.py --multicity
    python pipeline/gold/build_gold_table.py --use-probe        # sarasota probe features
"""

import argparse
import os
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# ── env ───────────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parents[2] / ".env", override=True)

BUCKET = os.getenv("S3_BUCKET_NAME", "street-risk-mvp")
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

ROOT = Path(__file__).parents[2]

CLIP_COLS = [
    "clip_heavy_traffic", "clip_poor_lighting", "clip_no_sidewalks",
    "clip_damaged_road",  "clip_clear_road",    "clip_no_signals",
    "clip_parked_cars",
]

POI_COLS = [
    "bars_count", "schools_count", "hospitals_count",
    "gas_stations_count", "fast_food_count", "traffic_signals_count",
]

AADT_COLS = [
    "aadt_mean", "aadt_max", "aadt_segment_count",
]

BASE_COLS = [
    "h3_index",
    "crash_density", "crash_count", "injury_rate",
    *CLIP_COLS,
    "road_type_primary", "speed_limit_mean", "lanes_mean",
    "dist_to_intersection_mean", "point_count",
    "risk_tier",
]

MULTICITY_COLS = [
    "h3_index", "city",
    "crash_density", "crash_count", "injury_rate",
    *CLIP_COLS,
    "road_type_primary", "speed_limit_mean", "lanes_mean",
    "dist_to_intersection_mean", "point_count",
    *POI_COLS,
    "risk_tier",
]

MULTICITY_V2_COLS = [
    "h3_index", "city",
    "crash_density", "crash_count", "injury_rate",
    *CLIP_COLS,
    "road_type_primary", "speed_limit_mean", "lanes_mean",
    "dist_to_intersection_mean", "point_count",
    *POI_COLS,
    *AADT_COLS,
    "risk_tier",
]


# ── road aggregation ──────────────────────────────────────────────────────────

def aggregate_roads(roads: pd.DataFrame) -> pd.DataFrame:
    roads = roads.copy()
    roads["speed_limit"] = roads["speed_limit"].fillna(25.0)
    roads["lanes"]       = roads["lanes"].fillna(1.0)

    agg = roads.groupby("h3_index").agg(
        road_type_primary         = ("road_type",                lambda x: x.mode().iloc[0] if len(x) else "unclassified"),
        speed_limit_mean          = ("speed_limit",              "mean"),
        lanes_mean                = ("lanes",                    "mean"),
        dist_to_intersection_mean = ("dist_to_intersection_m",  "mean"),
        point_count               = ("h3_index",                 "count"),
    ).reset_index()
    return agg


# ── join ──────────────────────────────────────────────────────────────────────

def build_city_gold(city: str, use_poi: bool = False) -> pd.DataFrame:
    """Build the gold table for a single city."""
    data = ROOT / "data"

    clip_path  = data / "silver" / "image_features" / f"{city}_clip_hex.parquet"
    crash_path = data / "silver" / "crash_hex"       / f"{city}_crash_hex.parquet"
    road_path  = data / "bronze"                     / f"{city}_road_points.parquet"
    poi_path   = data / "silver" / "poi_features"    / f"{city}_poi_hex.parquet"

    if not clip_path.exists():
        raise FileNotFoundError(f"CLIP hex not found: {clip_path}")
    if not crash_path.exists():
        raise FileNotFoundError(f"Crash hex not found: {crash_path}")
    if not road_path.exists():
        raise FileNotFoundError(f"Road points not found: {road_path}")

    clip  = pd.read_parquet(clip_path)
    crash = pd.read_parquet(crash_path)
    roads = pd.read_parquet(road_path)

    roads_agg = aggregate_roads(roads)

    gold = clip.copy()
    gold = gold.merge(roads_agg, on="h3_index", how="left")
    gold = gold.merge(
        crash[["h3_index", "crash_count", "crash_density", "injury_rate"]],
        on="h3_index", how="left",
    )
    gold["crash_count"]   = gold["crash_count"].fillna(0.0)
    gold["crash_density"] = gold["crash_density"].fillna(0.0)
    gold["injury_rate"]   = gold["injury_rate"].fillna(0.0)

    if use_poi:
        if poi_path.exists():
            poi = pd.read_parquet(poi_path)
            gold = gold.merge(poi, on="h3_index", how="left")
            for col in POI_COLS:
                gold[col] = gold[col].fillna(0).astype(int)
        else:
            print(f"  [warn] POI file not found for {city}, filling zeros: {poi_path}")
            for col in POI_COLS:
                gold[col] = 0

    gold["city"] = city
    return gold


# ── risk tier ─────────────────────────────────────────────────────────────────

def add_risk_tier(df: pd.DataFrame) -> tuple:
    p33 = df["crash_density"].quantile(0.33)
    p66 = df["crash_density"].quantile(0.66)

    def tier(v):
        if v <= p33:   return "Low"
        elif v <= p66: return "Medium"
        return "High"

    df["risk_tier"] = df["crash_density"].apply(tier)
    return df, p33, p66


# ── S3 ────────────────────────────────────────────────────────────────────────

def s3_key_exists(client, bucket, key):
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def upload(local_path: Path, bucket: str, key: str):
    client = boto3.client(
        "s3", region_name=REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    if s3_key_exists(client, bucket, key):
        print(f"  [overwrite] s3://{bucket}/{key}")
    client.upload_file(str(local_path), bucket, key)
    print(f"  [ok] Uploaded: s3://{bucket}/{key}")


def save_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)
    print(f"  [ok] Saved locally: {path}")


# ── summary ───────────────────────────────────────────────────────────────────

def print_summary(gold: pd.DataFrame, p33: float, p66: float, final_cols: list, label: str = ""):
    n_total      = len(gold)
    n_with_crash = (gold["crash_count"] > 0).sum()

    print(f"\n{'='*60}")
    print(f"  GOLD TABLE SUMMARY{' — ' + label if label else ''}")
    print(f"{'='*60}")
    print(f"\n  Total hexagons          : {n_total}")
    print(f"  With crash data         : {n_with_crash}  ({100*n_with_crash/n_total:.1f}%)")
    print(f"\n  Risk tier thresholds:")
    print(f"    p33 : {p33:.2f} crashes/km2")
    print(f"    p66 : {p66:.2f} crashes/km2")
    print(f"\n  Risk tier distribution:")
    for tier, count in gold["risk_tier"].value_counts().sort_index().items():
        print(f"    {tier:<8} : {count:>5}  ({100*count/n_total:.1f}%)")

    if "city" in gold.columns:
        print(f"\n  By city:")
        for city, grp in gold.groupby("city"):
            print(f"    {city:<12}: {len(grp):>5} hexagons  "
                  f"density mean={grp['crash_density'].mean():.1f}  max={grp['crash_density'].max():.1f}")

    d = gold["crash_density"]
    print(f"\n  crash_density: min={d.min():.2f}  mean={d.mean():.2f}  median={d.median():.2f}  max={d.max():.2f}")

    print(f"\n  Feature correlations with crash_density:")
    num_cols = [c for c in gold.columns
                if c.startswith("clip_") or c in
                ["speed_limit_mean","lanes_mean","dist_to_intersection_mean","point_count",
                 *POI_COLS, *AADT_COLS]]
    num_cols = [c for c in num_cols if c in gold.columns]
    corrs = gold[num_cols + ["crash_density"]].corr()["crash_density"].drop("crash_density")
    for col, r in corrs.sort_values(key=abs, ascending=False).head(15).items():
        bar  = "#" * int(abs(r) * 30)
        sign = "+" if r >= 0 else "-"
        print(f"    {col:<32} {sign}{abs(r):.4f}  {bar}")

    print(f"\n  NaN counts per column:")
    nan_counts = gold[[c for c in final_cols if c in gold.columns]].isna().sum()
    any_nan = False
    for col, cnt in nan_counts.items():
        if cnt > 0:
            print(f"    {col:<32} {cnt}")
            any_nan = True
    if not any_nan:
        print("    (none)")

    print(f"\n{'='*60}")


# ── single-city main ──────────────────────────────────────────────────────────

def main_single(city: str = "sarasota", use_probe: bool = False):
    data = ROOT / "data"
    print(f"\n- Building Gold table — {city.upper()} -\n")

    if use_probe:
        probe_path = data / "silver" / "image_features" / f"{city}_clip_probe_hex.parquet"
        clip_path  = data / "silver" / "image_features" / f"{city}_clip_hex.parquet"
        if not probe_path.exists():
            raise FileNotFoundError(f"Probe hex not found: {probe_path}")
        clip   = pd.read_parquet(probe_path)
        crash  = pd.read_parquet(data / "silver" / "crash_hex" / f"{city}_crash_hex.parquet")
        roads  = pd.read_parquet(data / "bronze" / f"{city}_road_points.parquet")
        roads_agg = aggregate_roads(roads)
        gold  = clip.copy()
        gold  = gold.merge(roads_agg, on="h3_index", how="left")
        gold  = gold.merge(crash[["h3_index","crash_count","crash_density","injury_rate"]],
                           on="h3_index", how="left")
        gold["crash_count"]   = gold["crash_count"].fillna(0.0)
        gold["crash_density"] = gold["crash_density"].fillna(0.0)
        gold["injury_rate"]   = gold["injury_rate"].fillna(0.0)
        final_cols = ["h3_index","crash_density","crash_count","injury_rate",
                      "clip_risk_prob","road_type_primary","speed_limit_mean",
                      "lanes_mean","dist_to_intersection_mean","point_count","risk_tier"]
        gold_local = data / "gold" / "training_table" / f"{city}_gold_probe.parquet"
        gold_s3key = f"gold/training_table/{city}_gold_probe.parquet"
    else:
        gold       = build_city_gold(city, use_poi=False)
        final_cols = BASE_COLS
        gold_local = data / "gold" / "training_table" / f"{city}_gold.parquet"
        gold_s3key = f"gold/training_table/{city}_gold.parquet"

    gold, p33, p66 = add_risk_tier(gold)
    gold = gold[[c for c in final_cols if c in gold.columns]]

    save_parquet(gold, gold_local)
    upload(gold_local, BUCKET, gold_s3key)
    print_summary(gold, p33, p66, final_cols, label=city)

    print(f"\n  Preview (top 5 by crash_density):")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)
    print(gold.sort_values("crash_density", ascending=False).head(5).to_string(index=False))
    print(f"\n- Done. -\n")
    return gold


# ── multi-city main ───────────────────────────────────────────────────────────

def main_multicity(include_aadt: bool = False, include_orlando: bool = False):
    data   = ROOT / "data"
    v2     = include_aadt
    v3     = include_aadt and include_orlando
    if v3:
        label = "multicity_v3 (Sarasota+Tampa+Orlando)"
    elif v2:
        label = "multicity_v2 (with AADT)"
    else:
        label = "multicity"
    print(f"\n- Building Multi-city Gold table ({label}) -\n")
    cities = ["sarasota", "tampa"] + (["orlando"] if include_orlando else [])

    n_steps = 5 if v2 else 4
    print(f"Step 1/{n_steps}  Building per-city gold tables ...")
    city_golds = []
    for city in cities:
        print(f"\n  [{city}]")
        g = build_city_gold(city, use_poi=True)
        city_golds.append(g)
        print(f"  [ok] {city}: {len(g):,} hexagons")

    print(f"\nStep 2/{n_steps}  Stacking cities ...")
    gold = pd.concat(city_golds, ignore_index=True)
    print(f"  [ok] Combined: {len(gold):,} hexagons")

    if v2:
        print(f"\nStep 3/{n_steps}  Joining AADT features ...")
        aadt_frames = []
        for city in cities:
            aadt_path = data / "silver" / "aadt" / f"{city}_aadt_hex.parquet"
            if not aadt_path.exists():
                raise FileNotFoundError(
                    f"AADT Silver file not found: {aadt_path}\n"
                    f"Run: python pipeline/features/extract_aadt_features.py --city {city}"
                )
            af = pd.read_parquet(aadt_path)
            aadt_frames.append(af)
            print(f"  [ok] {city}: {len(af):,} AADT rows loaded")

        aadt_all = pd.concat(aadt_frames, ignore_index=True)
        gold = gold.merge(aadt_all[["h3_index"] + AADT_COLS], on="h3_index", how="left")
        for col in AADT_COLS:
            n_nan = gold[col].isna().sum()
            if n_nan:
                print(f"  [fill] {col}: {n_nan} NaN -> 0")
            gold[col] = gold[col].fillna(0.0)
        # aadt_segment_count should be int
        gold["aadt_segment_count"] = gold["aadt_segment_count"].astype(int)
        print(f"  [ok] AADT join complete — NaN check: "
              f"{gold[AADT_COLS].isna().sum().sum()} remaining NaN")

        step_risk = 4
    else:
        step_risk = 3

    print(f"\nStep {step_risk}/{n_steps}  Adding risk_tier (cross-city quantiles) ...")
    gold, p33, p66 = add_risk_tier(gold)

    # Enforce column order
    col_list   = MULTICITY_V2_COLS if v2 else MULTICITY_COLS
    final_cols = [c for c in col_list if c in gold.columns]
    gold       = gold[final_cols]

    # Zero-NaN confirmation
    nan_total = gold.isna().sum().sum()
    print(f"  [ok] NaN count in final table: {nan_total}")
    print(f"  [ok] Feature count: {len(final_cols)} columns  "
          f"({len([c for c in final_cols if c not in ['h3_index','city','risk_tier']])} features + 3 meta)")

    step_save = step_risk + 1
    print(f"\nStep {step_save}/{n_steps}  Saving ...")
    fname      = ("multicity_gold_v3.parquet" if v3
                  else "multicity_gold_v2.parquet" if v2
                  else "multicity_gold.parquet")
    gold_s3key = f"gold/training_table/{fname}"
    gold_local = data / "gold" / "training_table" / fname
    save_parquet(gold, gold_local)
    upload(gold_local, BUCKET, gold_s3key)

    print_summary(gold, p33, p66, final_cols, label=label)

    extra_cols = POI_COLS + (AADT_COLS if v2 else [])
    print(f"\n  Feature correlations with crash_density by city:")
    for city in cities:
        grp = gold[gold["city"] == city]
        cols_avail = [c for c in extra_cols if c in grp.columns]
        corrs = (grp[cols_avail + ["crash_density"]]
                 .corr()["crash_density"]
                 .drop("crash_density"))
        print(f"\n  {city}:")
        for col, r in corrs.sort_values(key=abs, ascending=False).items():
            print(f"    {col:<32} {'+' if r>=0 else '-'}{abs(r):.4f}")

    print(f"\n  Preview (top 5 by crash_density):")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(gold.sort_values("crash_density", ascending=False).head(5).to_string(index=False))
    print(f"\n- Done. -\n")
    return gold


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Gold training table(s).")
    parser.add_argument("--city",       choices=["sarasota", "tampa", "orlando"], default="sarasota")
    parser.add_argument("--multicity",  action="store_true", help="Stack sarasota + tampa.")
    parser.add_argument("--aadt",       action="store_true", help="Include AADT features (requires --multicity).")
    parser.add_argument("--orlando",    action="store_true", help="Include Orlando as third city (requires --multicity --aadt). Outputs multicity_gold_v3.parquet.")
    parser.add_argument("--use-probe",  action="store_true", help="Use probe features (sarasota only).")
    args = parser.parse_args()

    if args.multicity:
        main_multicity(include_aadt=args.aadt, include_orlando=args.orlando)
    else:
        main_single(city=args.city, use_probe=args.use_probe)

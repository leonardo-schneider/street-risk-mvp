"""
pipeline/gold/build_gold_table.py

Assembles the final Gold training table by joining:
  - Silver crash hex aggregates  (target variable)
  - Silver CLIP hex features      (image features)
  - Bronze road points            (tabular road features)

Output: data/gold/training_table/sarasota_gold.parquet
        s3://street-risk-mvp/gold/training_table/sarasota_gold.parquet

Usage:
    python pipeline/gold/build_gold_table.py
"""

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

# ── paths ─────────────────────────────────────────────────────────────────────
CRASH_PATH = Path(__file__).parents[2] / "data" / "silver" / "crash_hex"      / "sarasota_crash_hex.parquet"
CLIP_PATH  = Path(__file__).parents[2] / "data" / "silver" / "image_features" / "sarasota_clip_hex.parquet"
ROAD_PATH  = Path(__file__).parents[2] / "data" / "bronze" / "sarasota_road_points.parquet"
GOLD_LOCAL = Path(__file__).parents[2] / "data" / "gold"   / "training_table" / "sarasota_gold.parquet"
GOLD_S3KEY = "gold/training_table/sarasota_gold.parquet"

CLIP_COLS = [
    "clip_heavy_traffic", "clip_poor_lighting", "clip_no_sidewalks",
    "clip_damaged_road",  "clip_clear_road",    "clip_no_signals",
    "clip_parked_cars",
]

FINAL_COLS = [
    "h3_index",
    "crash_density", "crash_count", "injury_rate",
    *CLIP_COLS,
    "road_type_primary", "speed_limit_mean", "lanes_mean",
    "dist_to_intersection_mean", "point_count",
    "risk_tier",
]


# ── road aggregation ──────────────────────────────────────────────────────────

def aggregate_roads(roads: pd.DataFrame) -> pd.DataFrame:
    roads = roads.copy()
    roads["speed_limit"] = roads["speed_limit"].fillna(25.0)
    roads["lanes"]       = roads["lanes"].fillna(1.0)

    agg = roads.groupby("h3_index").agg(
        road_type_primary       = ("road_type",            lambda x: x.mode().iloc[0] if len(x) else "unclassified"),
        speed_limit_mean        = ("speed_limit",          "mean"),
        lanes_mean              = ("lanes",                "mean"),
        dist_to_intersection_mean = ("dist_to_intersection_m", "mean"),
        point_count             = ("h3_index",             "count"),
    ).reset_index()

    return agg


# ── joins ─────────────────────────────────────────────────────────────────────

def build_gold(clip: pd.DataFrame, roads_agg: pd.DataFrame, crash: pd.DataFrame) -> pd.DataFrame:
    # Start with CLIP (image-covered hexagons)
    gold = clip.copy()

    # Left join road aggregates
    gold = gold.merge(roads_agg, on="h3_index", how="left")

    # Left join crash data
    gold = gold.merge(
        crash[["h3_index", "crash_count", "crash_density", "injury_rate"]],
        on="h3_index",
        how="left",
    )

    # Hexagons with no crash records → zero crashes
    gold["crash_count"]   = gold["crash_count"].fillna(0.0)
    gold["crash_density"] = gold["crash_density"].fillna(0.0)
    gold["injury_rate"]   = gold["injury_rate"].fillna(0.0)

    return gold


# ── risk tier ─────────────────────────────────────────────────────────────────

def add_risk_tier(gold: pd.DataFrame) -> pd.DataFrame:
    p33 = gold["crash_density"].quantile(0.33)
    p66 = gold["crash_density"].quantile(0.66)

    def tier(v):
        if v <= p33:
            return "Low"
        elif v <= p66:
            return "Medium"
        return "High"

    gold["risk_tier"] = gold["crash_density"].apply(tier)
    return gold, p33, p66


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
        "s3",
        region_name=REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    if s3_key_exists(client, bucket, key):
        print(f"  [overwrite] s3://{bucket}/{key}")
    client.upload_file(str(local_path), bucket, key)
    print(f"  [ok] Uploaded: s3://{bucket}/{key}")


# ── summary ───────────────────────────────────────────────────────────────────

def print_summary(gold: pd.DataFrame, p33: float, p66: float):
    n_total       = len(gold)
    n_with_crash  = (gold["crash_count"] > 0).sum()
    n_zero_crash  = n_total - n_with_crash

    print("\n" + "=" * 60)
    print("  GOLD TABLE SUMMARY")
    print("=" * 60)
    print(f"\n  Total hexagons          : {n_total}")
    print(f"  With crash data         : {n_with_crash}  ({100*n_with_crash/n_total:.1f}%)")
    print(f"  Zero-crash hexagons     : {n_zero_crash}  ({100*n_zero_crash/n_total:.1f}%)")
    print(f"\n  Risk tier thresholds:")
    print(f"    p33 (Low/Med boundary) : {p33:.2f} crashes/km2")
    print(f"    p66 (Med/High boundary): {p66:.2f} crashes/km2")

    print(f"\n  Risk tier distribution:")
    for tier, count in gold["risk_tier"].value_counts().sort_index().items():
        print(f"    {tier:<8} : {count:>4}  ({100*count/n_total:.1f}%)")

    print(f"\n  crash_density stats:")
    d = gold["crash_density"]
    print(f"    min={d.min():.2f}  mean={d.mean():.2f}  median={d.median():.2f}  max={d.max():.2f}")

    print(f"\n  Feature correlations with crash_density:")
    num_cols = CLIP_COLS + ["speed_limit_mean", "lanes_mean", "dist_to_intersection_mean", "point_count"]
    corrs = gold[num_cols + ["crash_density"]].corr()["crash_density"].drop("crash_density")
    for col, r in corrs.sort_values(key=abs, ascending=False).items():
        bar = "#" * int(abs(r) * 30)
        sign = "+" if r >= 0 else "-"
        print(f"    {col:<32} {sign}{abs(r):.4f}  {bar}")

    print(f"\n  NaN counts per column:")
    nan_counts = gold[FINAL_COLS].isna().sum()
    any_nan = False
    for col, cnt in nan_counts.items():
        if cnt > 0:
            print(f"    {col:<32} {cnt}")
            any_nan = True
    if not any_nan:
        print("    (none)")

    print("\n" + "=" * 60)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n- Building Gold training table -\n")

    # 1. Load
    print("Step 1/5  Loading Silver + Bronze inputs ...")
    crash = pd.read_parquet(CRASH_PATH)
    clip  = pd.read_parquet(CLIP_PATH)
    roads = pd.read_parquet(ROAD_PATH)
    print(f"  [ok] crash_hex     : {len(crash):,} hexagons")
    print(f"  [ok] clip_hex      : {len(clip):,} hexagons")
    print(f"  [ok] road_points   : {len(roads):,} points across {roads['h3_index'].nunique():,} hexagons")

    # 2. Aggregate roads
    print("\nStep 2/5  Aggregating road points to hexagon level ...")
    roads_agg = aggregate_roads(roads)
    print(f"  [ok] {len(roads_agg):,} road hexagons aggregated")

    # 3. Join
    print("\nStep 3/5  Joining tables ...")
    gold = build_gold(clip, roads_agg, crash)
    print(f"  [ok] Gold table: {len(gold):,} rows x {len(gold.columns)} columns")

    # 4. Risk tier
    print("\nStep 4/5  Adding risk_tier column ...")
    gold, p33, p66 = add_risk_tier(gold)

    # Enforce column order
    gold = gold[FINAL_COLS]

    # 5. Save
    print("\nStep 5/5  Saving ...")
    GOLD_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(gold, preserve_index=False), GOLD_LOCAL)
    print(f"  [ok] Saved locally: {GOLD_LOCAL}")
    upload(GOLD_LOCAL, BUCKET, GOLD_S3KEY)

    # Summary
    print_summary(gold, p33, p66)

    print(f"\n  Preview (top 5 by crash_density):")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print(gold.sort_values("crash_density", ascending=False).head(5).to_string(index=False))
    print()

    print("- Done. -\n")
    return gold


if __name__ == "__main__":
    main()

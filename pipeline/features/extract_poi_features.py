"""
pipeline/features/extract_poi_features.py

For each H3 res-9 hexagon covered by road points, downloads OSM points of
interest for the whole city at once, assigns each POI to a hexagon, and
aggregates 6 count features per hex:

  bars_count          amenity in {bar, pub, nightclub}
  schools_count       amenity in {school, university}
  hospitals_count     amenity in {hospital, clinic}
  gas_stations_count  amenity=fuel
  fast_food_count     amenity=fast_food
  traffic_signals_count  highway=traffic_signals (node)

Strategy: download all POIs for the city bounding box at once
(one OSMnx call per tag group), assign H3 index, then groupby-count.

Usage:
    python pipeline/features/extract_poi_features.py                 # sarasota
    python pipeline/features/extract_poi_features.py --city tampa
    python pipeline/features/extract_poi_features.py --city tampa --dry-run
"""

import argparse
import os
from pathlib import Path

import boto3
import h3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import osmnx as ox
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from tqdm import tqdm

# ── env ───────────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parents[2] / ".env", override=True)

BUCKET = os.getenv("S3_BUCKET_NAME", "street-risk-mvp")
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

DATA_ROOT = Path(__file__).parents[2] / "data"
H3_RES    = 9

CITY_PLACE = {
    "sarasota": "Sarasota, Florida, USA",
    "tampa":    "Tampa, Florida, USA",
    "orlando":  "Orlando, Florida, USA",
}

# OSM tag groups to download and the column names they map to
TAG_GROUPS = [
    (
        "bars_count",
        {"amenity": ["bar", "pub", "nightclub"]},
    ),
    (
        "schools_count",
        {"amenity": ["school", "university"]},
    ),
    (
        "hospitals_count",
        {"amenity": ["hospital", "clinic"]},
    ),
    (
        "gas_stations_count",
        {"amenity": "fuel"},
    ),
    (
        "fast_food_count",
        {"amenity": "fast_food"},
    ),
    (
        "traffic_signals_count",
        {"highway": "traffic_signals"},
    ),
]

POI_COLS = [col for col, _ in TAG_GROUPS]


# ── S3 ────────────────────────────────────────────────────────────────────────

def make_s3():
    return boto3.client(
        "s3",
        region_name=REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


def s3_key_exists(client, bucket, key):
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def save_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)
    print(f"  [ok] Saved locally: {path}")


def upload(local_path: Path, s3, s3_key: str):
    if s3_key_exists(s3, BUCKET, s3_key):
        print(f"  [overwrite] s3://{BUCKET}/{s3_key}")
    s3.upload_file(str(local_path), BUCKET, s3_key)
    print(f"  [ok] Uploaded: s3://{BUCKET}/{s3_key}")


# ── OSM helpers ───────────────────────────────────────────────────────────────

def fetch_pois_for_city(place: str, tags: dict) -> pd.DataFrame:
    """
    Download OSM features matching `tags` for the given place.
    Returns DataFrame with lat/lon columns (centroid for polygons).
    Returns empty DataFrame if no features found.
    """
    try:
        gdf = ox.features_from_place(place, tags=tags)
    except Exception:
        return pd.DataFrame(columns=["lat", "lon"])

    if gdf.empty:
        return pd.DataFrame(columns=["lat", "lon"])

    # Use centroid for geometry
    centroids = gdf.geometry.centroid
    result = pd.DataFrame({
        "lat": centroids.y.values,
        "lon": centroids.x.values,
    })
    return result.dropna()


def assign_h3(df: pd.DataFrame) -> pd.Series:
    return df.apply(lambda r: h3.latlng_to_cell(r["lat"], r["lon"], H3_RES), axis=1)


# ── main ──────────────────────────────────────────────────────────────────────

def main(city: str = "sarasota", dry_run: bool = False):
    place      = CITY_PLACE[city]
    local_path = DATA_ROOT / "silver" / "poi_features" / f"{city}_poi_hex.parquet"
    s3_key     = f"silver/poi_features/{city}_poi_hex.parquet"

    print(f"\n- POI feature extraction — {city.upper()} ({place}) -\n")

    # 1. Load road points to know which hexagons to cover
    print("Step 1/3  Loading road points to get hexagon universe ...")
    road_path = DATA_ROOT / "bronze" / f"{city}_road_points.parquet"
    if not road_path.exists():
        raise FileNotFoundError(
            f"Road points not found: {road_path}\n"
            f"Run `python pipeline/ingestion/sample_roads.py --city {city}` first."
        )
    road_df  = pd.read_parquet(road_path)
    hex_list = road_df["h3_index"].unique().tolist()
    hex_set  = set(hex_list)
    print(f"  [ok] {len(hex_list):,} hexagons in road universe")

    if dry_run:
        hex_list = hex_list[:20]
        hex_set  = set(hex_list)
        print(f"  (--dry-run) Limited to {len(hex_list)} hexagons")

    # 2. Download POIs per tag group and count per hex
    print(f"\nStep 2/3  Downloading OSM POIs for {place} ...")
    counts = {col: {h: 0 for h in hex_list} for col in POI_COLS}

    for col, tags in tqdm(TAG_GROUPS, desc="Tag groups", unit="group"):
        pois = fetch_pois_for_city(place, tags)
        if pois.empty:
            tqdm.write(f"  [warn] No features found for {col} ({tags})")
            continue

        tqdm.write(f"  {col}: {len(pois):,} POIs downloaded")

        # Assign H3 index and count within known hexes
        tqdm.pandas(desc=f"  Assigning H3 for {col}")
        poi_hex = pois.progress_apply(
            lambda r: h3.latlng_to_cell(r["lat"], r["lon"], H3_RES), axis=1
        )
        for h in poi_hex:
            if h in hex_set:
                counts[col][h] = counts[col].get(h, 0) + 1

    # 3. Build output DataFrame
    print(f"\nStep 3/3  Building POI hex table ...")
    rows = []
    for h in hex_list:
        row = {"h3_index": h}
        for col in POI_COLS:
            row[col] = counts[col].get(h, 0)
        rows.append(row)

    poi_df = pd.DataFrame(rows)
    for col in POI_COLS:
        poi_df[col] = poi_df[col].astype(int)

    print(f"  [ok] {len(poi_df):,} hexagons  |  {len(POI_COLS)} features")
    print(f"\n  Mean POI counts per hexagon:")
    for col in POI_COLS:
        nonzero = (poi_df[col] > 0).sum()
        print(f"    {col:<28} mean={poi_df[col].mean():.3f}  "
              f"max={poi_df[col].max()}  hexes_with_any={nonzero}")

    # 4. Save
    save_parquet(poi_df, local_path)
    if dry_run:
        print("  (--dry-run) Skipping S3 upload.")
    else:
        upload(local_path, make_s3(), s3_key)

    print(f"\n- Done. -\n")
    return poi_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract OSM POI features per H3 hexagon.")
    parser.add_argument("--city",     choices=["sarasota", "tampa", "orlando"], default="sarasota")
    parser.add_argument("--dry-run",  action="store_true", help="First 20 hexagons only.")
    args = parser.parse_args()
    main(city=args.city, dry_run=args.dry_run)

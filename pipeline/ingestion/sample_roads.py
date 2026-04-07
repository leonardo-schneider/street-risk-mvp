"""
pipeline/ingestion/sample_roads.py

Downloads the Sarasota, FL road network via OSMnx, samples ~1500 points
along roads, extracts tabular features for each point, saves to local
data/bronze/ and uploads to S3 bronze/roads/.

Usage:
    python pipeline/ingestion/sample_roads.py            # full run
    python pipeline/ingestion/sample_roads.py --dry-run  # skip S3 upload
"""

import argparse
import os
from pathlib import Path

import boto3
import h3
import numpy as np
import osmnx as ox
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from tqdm import tqdm

# ── env ───────────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parents[2] / ".env", override=True)

BUCKET     = os.getenv("S3_BUCKET_NAME", "street-risk-mvp")
REGION     = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
S3_KEY     = "bronze/roads/sarasota_road_points.parquet"
LOCAL_PATH = Path(__file__).parents[2] / "data" / "bronze" / "sarasota_road_points.parquet"

TARGET_POINTS = 1500
PLACE         = "Sarasota, Florida, USA"
H3_RES        = 9
NETWORK_TYPE  = "drive"

DEFAULT_SPEED = 35   # mph, typical urban US default
DEFAULT_LANES = 1


# ── helpers ───────────────────────────────────────────────────────────────────

def _scalar(val, default):
    """Unwrap list-valued OSM tags; return first numeric value or default."""
    if isinstance(val, list):
        val = val[0]
    try:
        return float(str(val).split()[0])  # handles "35 mph" -> 35.0
    except (ValueError, TypeError):
        return float(default)


def sample_points_on_edges(G, n_points):
    """
    Interpolate points along every edge proportional to edge length,
    returning ~n_points dicts with lat/lon and edge attributes.
    """
    edges = ox.graph_to_gdfs(G, nodes=False)
    total_length = edges["length"].sum()
    points_per_meter = n_points / total_length

    records = []
    for _, row in tqdm(edges.iterrows(), total=len(edges), desc="Sampling edges", unit="edge"):
        edge_len = row["length"]
        n = max(1, round(edge_len * points_per_meter))
        geom = row["geometry"]

        distances = np.linspace(0, geom.length, n, endpoint=False)
        for d in distances:
            pt = geom.interpolate(d)
            records.append({
                "lon":         pt.x,
                "lat":         pt.y,
                "road_type":   row.get("highway", "unclassified"),
                "speed_limit": _scalar(row.get("maxspeed"), DEFAULT_SPEED),
                "lanes":       _scalar(row.get("lanes"),    DEFAULT_LANES),
            })

    return records


def add_h3_index(df):
    tqdm.pandas(desc="Assigning H3 index")
    df["h3_index"] = df.progress_apply(
        lambda r: h3.latlng_to_cell(r["lat"], r["lon"], H3_RES), axis=1
    )
    return df


def add_distance_to_intersection(G, df):
    """
    For each sampled point find the nearest graph node and compute
    Euclidean distance in metres using a flat-earth approximation
    (accurate enough within a single city).
    """
    nodes_gdf = ox.graph_to_gdfs(G, edges=False)[["x", "y"]]
    node_coords = nodes_gdf[["x", "y"]].values  # columns: lon, lat

    lons = df["lon"].values
    lats = df["lat"].values

    m_per_deg_lat = 111_320
    m_per_deg_lon = 111_320 * np.cos(np.radians(np.mean(lats)))

    distances = []
    for lon, lat in tqdm(
        zip(lons, lats), total=len(lons), desc="Distance to intersection", unit="pt"
    ):
        dlat = (node_coords[:, 1] - lat) * m_per_deg_lat
        dlon = (node_coords[:, 0] - lon) * m_per_deg_lon
        distances.append(np.sqrt(dlat**2 + dlon**2).min())

    df["dist_to_intersection_m"] = distances
    return df


def normalize_road_type(df):
    """Flatten list-valued highway tags to a single string."""
    def _flatten(val):
        if isinstance(val, list):
            return val[0]
        return str(val) if val else "unclassified"

    df["road_type"] = df["road_type"].apply(_flatten)
    return df


# ── S3 ────────────────────────────────────────────────────────────────────────

def s3_key_exists(client, bucket, key):
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise


def upload_to_s3(local_path, bucket, key):
    client = boto3.client(
        "s3",
        region_name=REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    if s3_key_exists(client, bucket, key):
        print(f"  [ok] S3 key already exists, skipping upload: s3://{bucket}/{key}")
        return

    print(f"  Uploading to s3://{bucket}/{key} ...")
    client.upload_file(str(local_path), bucket, key)
    print(f"  [ok] Upload complete.")


# ── main ──────────────────────────────────────────────────────────────────────

def main(dry_run=False):
    print(f"\n- Road sampling - {PLACE} -\n")

    # 1. Download road network
    print("Step 1/5  Downloading OSM road network ...")
    G = ox.graph_from_place(PLACE, network_type=NETWORK_TYPE)
    print(f"  [ok] Graph loaded: {len(G.nodes):,} nodes, {len(G.edges):,} edges")

    # 2. Sample points along edges
    print(f"\nStep 2/5  Sampling ~{TARGET_POINTS} points along edges ...")
    records = sample_points_on_edges(G, TARGET_POINTS)
    df = pd.DataFrame(records)
    print(f"  [ok] Sampled {len(df):,} points")

    # 3. H3 index
    print("\nStep 3/5  Assigning H3 res-9 index ...")
    df = add_h3_index(df)
    print(f"  [ok] {df['h3_index'].nunique():,} unique H3 hexagons covered")

    # 4. Distance to nearest intersection
    print("\nStep 4/5  Computing distance to nearest intersection ...")
    df = add_distance_to_intersection(G, df)

    # 5. Finalize
    df = normalize_road_type(df)
    df = df[[
        "lat", "lon", "h3_index",
        "road_type", "speed_limit", "lanes",
        "dist_to_intersection_m",
    ]]
    df["speed_limit"] = df["speed_limit"].astype(float)
    df["lanes"]       = df["lanes"].astype(float)

    print(f"\n  Preview:\n{df.head(3).to_string(index=False)}\n")
    print(f"  Shape: {df.shape}")

    # 6. Save locally
    print(f"\nStep 5/5  Saving ...")
    LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, LOCAL_PATH)
    print(f"  [ok] Saved locally: {LOCAL_PATH}")

    # 7. Upload to S3
    if dry_run:
        print("  (--dry-run) Skipping S3 upload.")
    else:
        upload_to_s3(LOCAL_PATH, BUCKET, S3_KEY)

    print(f"\n- Done. {len(df):,} road points ready. -\n")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample Sarasota road points from OSM.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip S3 upload (local save only).",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
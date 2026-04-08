"""
pipeline/ingestion/fetch_images.py

Reads data/bronze/sarasota_road_points.parquet, selects 2 representative
points per H3 hexagon (closest to centroid), fetches 2 Street View images
per point (headings 0 and 90), caches them in S3 bronze/images/, and
writes a local manifest CSV.

Usage:
    python pipeline/ingestion/fetch_images.py            # full run
    python pipeline/ingestion/fetch_images.py --dry-run  # 5 hexagons only
"""

import argparse
import io
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
import h3
import numpy as np
import pandas as pd
import requests
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from tqdm import tqdm

# ── env ───────────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parents[2] / ".env", override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
BUCKET         = os.getenv("S3_BUCKET_NAME", "street-risk-mvp")
REGION         = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# ── constants ─────────────────────────────────────────────────────────────────
LOCAL_POINTS   = Path(__file__).parents[2] / "data" / "bronze" / "sarasota_road_points.parquet"
MANIFEST_PATH  = Path(__file__).parents[2] / "data" / "bronze" / "image_manifest.csv"

HEADINGS       = [0, 90]
IMG_SIZE       = "640x640"
FOV            = 90
PITCH          = 0
POINTS_PER_HEX = 2          # representative points per hexagon
RATE_LIMIT_RPS = 5           # requests per second
COST_PER_IMAGE = 0.007       # USD, Street View Static API standard tier
STREETVIEW_URL = "https://maps.googleapis.com/maps/api/streetview"


# ── helpers ───────────────────────────────────────────────────────────────────

def hex_centroid(h3_index: str) -> tuple:
    """Return (lat, lon) of the H3 cell centre."""
    lat, lon = h3.cell_to_latlng(h3_index)
    return lat, lon


def two_closest_to_centroid(group: pd.DataFrame) -> pd.DataFrame:
    """
    Given all road points in one hexagon, return the POINTS_PER_HEX rows
    whose (lat, lon) is closest to the hexagon centroid.
    """
    h3_index = group["h3_index"].iloc[0]
    clat, clon = hex_centroid(h3_index)

    m_per_deg_lat = 111_320
    m_per_deg_lon = 111_320 * np.cos(np.radians(clat))

    dlat = (group["lat"].values - clat) * m_per_deg_lat
    dlon = (group["lon"].values - clon) * m_per_deg_lon
    dist = np.sqrt(dlat**2 + dlon**2)

    idx = np.argsort(dist)[:POINTS_PER_HEX]
    return group.iloc[idx]


def s3_key_for(h3_index: str, heading: int) -> str:
    return f"bronze/images/{h3_index}_{heading}.jpg"


def s3_key_exists(client, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def fetch_streetview_bytes(lat: float, lon: float, heading: int):
    """
    Fetch a Street View image and return raw JPEG bytes, or None on failure.
    Returns None (not an error) when Street View has no imagery at the location.
    """
    params = {
        "size":    IMG_SIZE,
        "location": f"{lat},{lon}",
        "heading": heading,
        "fov":     FOV,
        "pitch":   PITCH,
        "key":     GOOGLE_API_KEY,
        "return_error_code": "true",
    }
    resp = requests.get(STREETVIEW_URL, params=params, timeout=15)
    if resp.status_code == 200:
        return resp.content
    return None


def build_manifest_row(
    h3_index: str,
    lat: float,
    lon: float,
    heading: int,
    s3_key: str,
    status: str,
) -> dict:
    return {
        "h3_index":   h3_index,
        "lat":        lat,
        "lon":        lon,
        "heading":    heading,
        "s3_key":     s3_key,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "status":     status,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main(dry_run: bool = False):
    print("\n- Street View image fetcher -\n")

    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_MAPS_API_KEY is not set in .env")

    # 1. Load road points
    print("Step 1/4  Loading road points ...")
    df = pd.read_parquet(LOCAL_POINTS)
    print(f"  [ok] {len(df):,} points across {df['h3_index'].nunique():,} hexagons loaded")

    # 2. Deduplicate: 2 closest-to-centroid points per hex
    print("\nStep 2/4  Selecting representative points per hexagon ...")
    rep = (
        df.groupby("h3_index", group_keys=False)
          .apply(two_closest_to_centroid)
          .reset_index(drop=True)
    )
    n_hexes = rep["h3_index"].nunique()
    print(f"  [ok] {len(rep):,} representative points across {n_hexes:,} hexagons")

    if dry_run:
        hex_sample = rep["h3_index"].unique()[:5]
        rep = rep[rep["h3_index"].isin(hex_sample)].copy()
        print(f"  (--dry-run) Limited to {rep['h3_index'].nunique()} hexagons "
              f"({len(rep)} points)")

    # 3. Set up S3 client
    s3 = boto3.client(
        "s3",
        region_name=REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    # 4. Fetch & upload
    total_images = len(rep) * len(HEADINGS)
    estimated_cost = total_images * COST_PER_IMAGE
    print(f"\nStep 3/4  Fetching images ...")
    print(f"  Points: {len(rep)}  |  Headings: {HEADINGS}  |  "
          f"Total requests: {total_images}  |  "
          f"Estimated API cost: ${estimated_cost:.2f}")

    manifest_rows = []
    fetched = skipped = failed = 0
    cost_remaining = estimated_cost
    min_interval = 1.0 / RATE_LIMIT_RPS

    progress = tqdm(
        total=total_images,
        desc="Images",
        unit="img",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                   "[cost remaining: ${{postfix}}]",
    )

    for _, row in rep.iterrows():
        h3_index = row["h3_index"]
        lat      = row["lat"]
        lon      = row["lon"]

        for heading in HEADINGS:
            key = s3_key_for(h3_index, heading)
            t0  = time.monotonic()

            if s3_key_exists(s3, BUCKET, key):
                status = "cached"
                skipped += 1
                cost_remaining -= COST_PER_IMAGE
            else:
                img_bytes = fetch_streetview_bytes(lat, lon, heading)
                if img_bytes:
                    s3.put_object(
                        Bucket=BUCKET,
                        Key=key,
                        Body=img_bytes,
                        ContentType="image/jpeg",
                    )
                    status = "ok"
                    fetched += 1
                else:
                    status = "no_imagery"
                    failed += 1
                cost_remaining -= COST_PER_IMAGE

                # rate limiting
                elapsed = time.monotonic() - t0
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)

            manifest_rows.append(
                build_manifest_row(h3_index, lat, lon, heading, key, status)
            )
            progress.set_postfix_str(f"{cost_remaining:.3f}")
            progress.update(1)

    progress.close()
    print(f"\n  [ok] Fetched: {fetched}  |  Cached/skipped: {skipped}  |  "
          f"No imagery: {failed}")

    # 5. Save manifest
    print("\nStep 4/4  Saving manifest ...")
    manifest = pd.DataFrame(manifest_rows)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Append to existing manifest if present
    if MANIFEST_PATH.exists():
        existing = pd.read_csv(MANIFEST_PATH)
        manifest = (
            pd.concat([existing, manifest])
              .drop_duplicates(subset=["h3_index", "heading"], keep="last")
              .reset_index(drop=True)
        )

    manifest.to_csv(MANIFEST_PATH, index=False)
    print(f"  [ok] Manifest saved: {MANIFEST_PATH}  ({len(manifest)} rows)")
    print(f"\n  Preview:\n{manifest.head(10).to_string(index=False)}\n")

    print(f"- Done. {fetched} new images uploaded to s3://{BUCKET}/bronze/images/ -\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch Street View images for Sarasota road points."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only 5 hexagons (for testing).",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)

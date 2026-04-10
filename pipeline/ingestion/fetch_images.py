"""
pipeline/ingestion/fetch_images.py

Reads road points for a given city, selects 2 representative points per H3
hexagon, fetches Street View images, caches in S3, and writes a city-specific
manifest CSV.

Headings per city:
  - sarasota: [180, 270] only — headings 0 & 90 already exist in S3
  - tampa:    [0, 90, 180, 270] — all 4 headings

S3 key scheme:
  - Sarasota headings 0/90: bronze/images/{h3_index}_{heading}.jpg  (legacy, already exist)
  - Sarasota headings 180/270: bronze/images/sarasota/{h3_index}_{heading}.jpg
  - Tampa all headings:        bronze/images/tampa/{h3_index}_{heading}.jpg

Usage:
    python pipeline/ingestion/fetch_images.py                              # sarasota full
    python pipeline/ingestion/fetch_images.py --dry-run                    # sarasota 5 hexes
    python pipeline/ingestion/fetch_images.py --city tampa --limit 414     # tampa urban core
    python pipeline/ingestion/fetch_images.py --city tampa --limit 414 --dry-run
"""

import argparse
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
DATA_ROOT      = Path(__file__).parents[2] / "data"

# Headings per city — sarasota 0/90 already exist in S3
CITY_HEADINGS = {
    "sarasota": [180, 270],
    "tampa":    [0, 90, 180, 270],
}

# Tampa city center (used for geographic clustering when --limit is set)
CITY_CENTER = {
    "sarasota": (27.3364, -82.5307),
    "tampa":    (27.9506, -82.4572),
}

IMG_SIZE       = "640x640"
FOV            = 90
PITCH          = 0
POINTS_PER_HEX = 2
RATE_LIMIT_RPS = 5
COST_PER_IMAGE = 0.007
STREETVIEW_URL = "https://maps.googleapis.com/maps/api/streetview"


# ── S3 key scheme ─────────────────────────────────────────────────────────────

def s3_key_for(city: str, h3_index: str, heading: int) -> str:
    """
    Sarasota headings 0/90 use the legacy path so existing cached images
    are found correctly. All new images include the city prefix.
    """
    if city == "sarasota" and heading in (0, 90):
        return f"bronze/images/{h3_index}_{heading}.jpg"
    return f"bronze/images/{city}/{h3_index}_{heading}.jpg"


# ── hex selection helpers ─────────────────────────────────────────────────────

def hex_centroid(h3_index: str) -> tuple:
    return h3.cell_to_latlng(h3_index)


def two_closest_to_centroid(group: pd.DataFrame) -> pd.DataFrame:
    """Return the POINTS_PER_HEX road points closest to the hexagon centroid."""
    h3_index   = group["h3_index"].iloc[0]
    clat, clon = hex_centroid(h3_index)

    m_per_deg_lat = 111_320
    m_per_deg_lon = 111_320 * np.cos(np.radians(clat))

    dlat = (group["lat"].values - clat) * m_per_deg_lat
    dlon = (group["lon"].values - clon) * m_per_deg_lon
    dist = np.sqrt(dlat**2 + dlon**2)

    return group.iloc[np.argsort(dist)[:POINTS_PER_HEX]]


def select_hexes_by_proximity(all_hexes: list, city_center: tuple, limit: int) -> list:
    """
    Sort hexagons by distance from city_center (flat-earth), return the
    `limit` closest hex indices.
    """
    clat, clon = city_center
    m_per_deg_lat = 111_320
    m_per_deg_lon = 111_320 * np.cos(np.radians(clat))

    records = []
    for h in all_hexes:
        hlat, hlon = hex_centroid(h)
        dlat = (hlat - clat) * m_per_deg_lat
        dlon = (hlon - clon) * m_per_deg_lon
        records.append((h, np.sqrt(dlat**2 + dlon**2)))

    records.sort(key=lambda x: x[1])
    return [h for h, _ in records[:limit]]


# ── S3 helpers ────────────────────────────────────────────────────────────────

def s3_key_exists(client, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def fetch_streetview_bytes(lat: float, lon: float, heading: int):
    params = {
        "size":              IMG_SIZE,
        "location":          f"{lat},{lon}",
        "heading":           heading,
        "fov":               FOV,
        "pitch":             PITCH,
        "key":               GOOGLE_API_KEY,
        "return_error_code": "true",
    }
    resp = requests.get(STREETVIEW_URL, params=params, timeout=15)
    return resp.content if resp.status_code == 200 else None


def build_manifest_row(h3_index, lat, lon, heading, s3_key, status) -> dict:
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

def main(city: str = "sarasota", dry_run: bool = False, limit: int = None):
    headings = CITY_HEADINGS[city]
    print(f"\n- Street View image fetcher — {city.upper()} -")
    print(f"  Headings: {headings}")
    if limit:
        print(f"  Hex limit: {limit} closest to city center {CITY_CENTER[city]}")
    print()

    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_MAPS_API_KEY is not set in .env")

    points_path     = DATA_ROOT / "bronze" / f"{city}_road_points.parquet"
    manifest_path   = DATA_ROOT / "bronze" / f"image_manifest_{city}.csv"
    legacy_manifest = DATA_ROOT / "bronze" / "image_manifest.csv" if city == "sarasota" else None

    # 1. Load road points
    print("Step 1/4  Loading road points ...")
    if not points_path.exists():
        raise FileNotFoundError(
            f"Road points not found: {points_path}\n"
            f"Run `python pipeline/ingestion/sample_roads.py --city {city}` first."
        )
    df = pd.read_parquet(points_path)
    print(f"  [ok] {len(df):,} points across {df['h3_index'].nunique():,} hexagons")

    # 2. Representative points per hex
    print("\nStep 2/4  Selecting representative points per hexagon ...")
    rep = (
        df.groupby("h3_index", group_keys=False)
          .apply(two_closest_to_centroid)
          .reset_index(drop=True)
    )
    n_hexes_total = rep["h3_index"].nunique()
    print(f"  [ok] {len(rep):,} representative points across {n_hexes_total:,} hexagons")

    # Geographic clustering: pick `limit` hexes closest to city center
    if limit and limit < n_hexes_total:
        all_hexes    = rep["h3_index"].unique().tolist()
        selected     = select_hexes_by_proximity(all_hexes, CITY_CENTER[city], limit)
        rep          = rep[rep["h3_index"].isin(selected)].copy()
        n_selected   = rep["h3_index"].nunique()

        # Bounding box of selected hexagons
        lats = [hex_centroid(h)[0] for h in selected]
        lons = [hex_centroid(h)[1] for h in selected]
        print(f"  [ok] Limited to {n_selected} closest hexagons to city center")
        print(f"  Bounding box of selected {n_selected} hexagons:")
        print(f"    lat {min(lats):.4f} to {max(lats):.4f}")
        print(f"    lon {min(lons):.4f} to {max(lons):.4f}")

    if dry_run:
        hex_sample = rep["h3_index"].unique()[:5]
        rep = rep[rep["h3_index"].isin(hex_sample)].copy()
        print(f"  (--dry-run) Limited to {rep['h3_index'].nunique()} hexagons "
              f"({len(rep)} points)")

    # 3. S3 client
    s3 = boto3.client(
        "s3",
        region_name=REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    # 4. Fetch & upload
    total_images   = len(rep) * len(headings)
    estimated_cost = total_images * COST_PER_IMAGE
    print(f"\nStep 3/4  Fetching images ...")
    print(f"  Points: {len(rep)}  |  Headings: {headings}  |  "
          f"Total requests: {total_images}  |  "
          f"Estimated API cost: ${estimated_cost:.2f}")

    manifest_rows  = []
    fetched = skipped = failed = 0
    cost_remaining = estimated_cost
    min_interval   = 1.0 / RATE_LIMIT_RPS

    progress = tqdm(total=total_images, desc="Images", unit="img")

    for _, row in rep.iterrows():
        h3_index = row["h3_index"]
        lat      = row["lat"]
        lon      = row["lon"]

        for heading in headings:
            key = s3_key_for(city, h3_index, heading)
            t0  = time.monotonic()

            if s3_key_exists(s3, BUCKET, key):
                status = "cached"
                skipped += 1
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

                elapsed = time.monotonic() - t0
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)

            cost_remaining -= COST_PER_IMAGE
            manifest_rows.append(
                build_manifest_row(h3_index, lat, lon, heading, key, status)
            )
            progress.set_postfix(cached=skipped, fetched=fetched, failed=failed)
            progress.update(1)

    progress.close()
    print(f"\n  [ok] Fetched: {fetched}  |  Cached/skipped: {skipped}  |  "
          f"No imagery: {failed}")
    actual_cost = fetched * COST_PER_IMAGE
    print(f"  Actual API cost this run: ${actual_cost:.2f}")

    # 5. Save manifest
    print("\nStep 4/4  Saving manifest ...")
    new_manifest = pd.DataFrame(manifest_rows)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        existing = pd.read_csv(manifest_path)
        new_manifest = (
            pd.concat([existing, new_manifest])
              .drop_duplicates(subset=["h3_index", "heading"], keep="last")
              .reset_index(drop=True)
        )

    new_manifest.to_csv(manifest_path, index=False)
    print(f"  [ok] Manifest saved: {manifest_path}  ({len(new_manifest)} rows)")

    # Sarasota backward compat: keep legacy image_manifest.csv in sync
    if legacy_manifest is not None:
        if legacy_manifest.exists():
            legacy_existing = pd.read_csv(legacy_manifest)
            combined = (
                pd.concat([legacy_existing, new_manifest])
                  .drop_duplicates(subset=["h3_index", "heading"], keep="last")
                  .reset_index(drop=True)
            )
        else:
            combined = new_manifest
        combined.to_csv(legacy_manifest, index=False)
        print(f"  [ok] Legacy manifest updated: {legacy_manifest}  ({len(combined)} rows)")

    # Summary by status
    status_counts = new_manifest["status"].value_counts().to_dict()
    print(f"\n  Manifest status breakdown: {status_counts}")
    print(f"\n  Preview:\n{new_manifest.head(6).to_string(index=False)}\n")
    print(f"- Done. {fetched} new images uploaded to s3://{BUCKET}/bronze/images/{city}/ -\n")
    return new_manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch Street View images for road points."
    )
    parser.add_argument(
        "--city",
        choices=["sarasota", "tampa"],
        default="sarasota",
        help="City to fetch images for (default: sarasota).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max hexagons to process, selected by proximity to city center.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only 5 hexagons (for testing).",
    )
    args = parser.parse_args()
    main(city=args.city, dry_run=args.dry_run, limit=args.limit)

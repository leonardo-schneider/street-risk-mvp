"""
pipeline/features/extract_aadt_features.py

Downloads FDOT AADT (Annual Average Daily Traffic) road segment data
via ArcGIS REST, spatially joins to H3 hexagons, and extracts
traffic volume features for Sarasota and Tampa.

Source: https://gis.fdot.gov/arcgis/rest/services/FTO/fto_PROD/MapServer/0

Features produced per hexagon:
  aadt_mean           — mean AADT of intersecting state-road segments
  aadt_max            — max AADT (busiest road in hex)
  aadt_segment_count  — number of segments (road density proxy)

Hexagons with no AADT coverage (residential / local streets not on
the state system) receive 0 for all three fields.

Usage:
    python pipeline/features/extract_aadt_features.py                  # both cities
    python pipeline/features/extract_aadt_features.py --city sarasota
    python pipeline/features/extract_aadt_features.py --city tampa
    python pipeline/features/extract_aadt_features.py --dry-run        # first page only
"""

import argparse
import os
import time
from pathlib import Path

import boto3
import h3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from scipy.stats import spearmanr
from shapely.geometry import shape, Polygon, MultiPolygon
from tqdm import tqdm

# ── env ───────────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parents[2] / ".env", override=True)

BUCKET = os.getenv("S3_BUCKET_NAME", "street-risk-mvp")
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
ROOT   = Path(__file__).parents[2]
DATA   = ROOT / "data"

# ── FDOT ArcGIS REST endpoint ─────────────────────────────────────────────────
FDOT_URL   = "https://gis.fdot.gov/arcgis/rest/services/FTO/fto_PROD/MapServer/7/query"
PAGE_SIZE  = 1000

CITY_BBOX = {
    "sarasota": {"xmin": -82.7, "ymin": 27.2, "xmax": -82.3, "ymax": 27.5},
    "tampa":    {"xmin": -82.6, "ymin": 27.8, "xmax": -82.2, "ymax": 28.1},
    "orlando":  {"xmin": -81.6, "ymin": 28.3, "xmax": -81.1, "ymax": 28.8},
}

GOLD_PATH = DATA / "gold" / "training_table" / "multicity_gold.parquet"


# ── S3 helpers ────────────────────────────────────────────────────────────────

def make_s3():
    return boto3.client(
        "s3", region_name=REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


def s3_key_exists(client, key):
    try:
        client.head_object(Bucket=BUCKET, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def save_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)
    print(f"  [ok] Saved: {path}")


def upload(local_path: Path, s3, key: str):
    if s3_key_exists(s3, key):
        print(f"  [overwrite] s3://{BUCKET}/{key}")
    s3.upload_file(str(local_path), BUCKET, key)
    print(f"  [ok] Uploaded: s3://{BUCKET}/{key}")


# ── FDOT download ─────────────────────────────────────────────────────────────

def fetch_page(bbox: dict, offset: int) -> dict:
    """
    Fetch one page of AADT segments from FDOT ArcGIS REST (layer 7).
    inSR=4326  — bbox coordinates are WGS84 lon/lat
    outSR=4326 — return geometry in WGS84 for Shapely intersection
    """
    params = {
        "geometry":          f"{bbox['xmin']},{bbox['ymin']},{bbox['xmax']},{bbox['ymax']}",
        "geometryType":      "esriGeometryEnvelope",
        "inSR":              "4326",
        "outSR":             "4326",
        "spatialRel":        "esriSpatialRelIntersects",
        "outFields":         "AADT,COUNTY,AADTFLG,ROADWAY",
        "returnGeometry":    "true",
        "f":                 "geojson",
        "resultOffset":      offset,
        "resultRecordCount": PAGE_SIZE,
    }
    r = requests.get(FDOT_URL, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def fetch_page_with_retry(bbox: dict, offset: int, max_retries: int = 3) -> dict:
    """Fetch one page, retrying up to max_retries times on failure."""
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            data = fetch_page(bbox, offset)
            if "error" in data:
                raise RuntimeError(f"ArcGIS error: {data['error']}")
            return data
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = attempt * 2
                print(f"  [retry {attempt}/{max_retries}] offset={offset} failed: {exc} — retrying in {wait}s")
                time.sleep(wait)
    raise RuntimeError(f"Page at offset={offset} failed after {max_retries} retries: {last_exc}")


def download_segments(city: str, dry_run: bool = False) -> list:
    """
    Download all AADT road segments for a city bounding box.

    Pagination:
      - Request resultOffset=0, resultRecordCount=1000
      - Continue while ArcGIS sets exceededTransferLimit=true
        OR the page returned exactly PAGE_SIZE records
        (the flag can be absent in some GeoJSON responses)
      - 0.5 s delay between pages; up to 3 retries per page

    Returns a list of dicts: {aadt, geometry (shapely)}.
    With dry_run=True, fetches only the first page.
    """
    bbox     = CITY_BBOX[city]
    segments = []
    offset   = 0
    page_num = 0

    pbar = tqdm(desc=f"  Downloading {city} AADT pages", unit="page", leave=False)

    while True:
        page_num += 1
        data     = fetch_page_with_retry(bbox, offset)
        features = data.get("features", [])

        if not features:
            pbar.write(f"  No features at offset={offset} — download complete.")
            break

        valid_before = len(segments)
        for feat in features:
            props = feat.get("properties", {})
            geom  = feat.get("geometry")
            if geom is None or props.get("AADT") is None:
                continue
            try:
                aadt_val = int(props["AADT"])
            except (TypeError, ValueError):
                continue
            try:
                geom_shape = shape(geom)
            except Exception:
                continue
            segments.append({"aadt": aadt_val, "geometry": geom_shape})

        page_valid = len(segments) - valid_before
        pbar.update(1)
        pbar.write(
            f"  Page {page_num} (offset={offset}): "
            f"{len(features)} raw  {page_valid} valid  "
            f"total={len(segments)}"
        )

        # Paginate if ArcGIS says limit exceeded, OR we received a full page
        exceeded      = data.get("exceededTransferLimit", False)
        full_page     = len(features) == PAGE_SIZE
        more_pages    = exceeded or full_page

        if not more_pages or dry_run:
            break

        offset += PAGE_SIZE
        time.sleep(0.5)   # 0.5 s between pages

    pbar.close()
    print(f"  {city.title()}: {len(segments):,} segments across {page_num} page(s)")
    return segments


# ── H3 hex polygon helper ─────────────────────────────────────────────────────

def hex_to_polygon(h3_index: str) -> Polygon:
    """Convert H3 cell to a Shapely Polygon."""
    geo = h3.cells_to_geo([h3_index])
    return shape(geo)


# ── spatial join ──────────────────────────────────────────────────────────────

def join_segments_to_hexes(hex_list: list, segments: list) -> pd.DataFrame:
    """
    For each hexagon, find intersecting AADT segments and aggregate.
    Returns DataFrame with h3_index, aadt_mean, aadt_max, aadt_segment_count.
    """
    rows = []
    n = len(hex_list)
    for i, h in enumerate(hex_list):
        if i % 100 == 0:
            print(f"  Joining hex {i+1}/{n} ...", end="\r")
        hex_poly = hex_to_polygon(h)
        hits = [s["aadt"] for s in segments if s["geometry"].intersects(hex_poly)]
        if hits:
            rows.append({
                "h3_index":            h,
                "aadt_mean":           float(np.mean(hits)),
                "aadt_max":            float(max(hits)),
                "aadt_segment_count":  len(hits),
            })
        else:
            rows.append({
                "h3_index":            h,
                "aadt_mean":           0.0,
                "aadt_max":            0.0,
                "aadt_segment_count":  0,
            })

    print()   # newline after \r progress
    return pd.DataFrame(rows)


# ── analysis ──────────────────────────────────────────────────────────────────

def analyze(city: str, df_aadt: pd.DataFrame, dry_run: bool = False):
    """Print coverage, correlation, and distribution stats."""
    mode = " [DRY RUN — partial data]" if dry_run else ""
    print(f"\n{'='*60}")
    print(f"  AADT ANALYSIS — {city.upper()}{mode}")
    print(f"{'='*60}")

    if df_aadt.empty or "aadt_mean" not in df_aadt.columns:
        print("  [warn] Empty AADT DataFrame — skipping analysis.")
        return None
    n_total = len(df_aadt)
    n_covered = (df_aadt["aadt_mean"] > 0).sum()
    pct_covered = 100 * n_covered / n_total if n_total else 0
    print(f"  Total hexagons:        {n_total:,}")
    print(f"  Hexagons with AADT:    {n_covered:,}  ({pct_covered:.1f}%)")
    print(f"  Hexagons with 0 AADT:  {n_total - n_covered:,}  ({100 - pct_covered:.1f}%)")

    # Load crash_density from gold table for correlation
    if not GOLD_PATH.exists():
        print(f"\n  [warn] Gold table not found at {GOLD_PATH} — skipping correlation.")
        return

    gold = pd.read_parquet(GOLD_PATH)[["h3_index", "city", "crash_density"]]
    gold = gold[gold["city"] == city]
    merged = gold.merge(df_aadt, on="h3_index", how="inner")
    print(f"  Merged with gold:      {len(merged):,} hexagons")

    if len(merged) < 5:
        print("  [warn] Too few merged hexagons for correlation.")
        return

    r_mean,  _ = spearmanr(merged["aadt_mean"],  merged["crash_density"])
    r_max,   _ = spearmanr(merged["aadt_max"],   merged["crash_density"])
    r_count, _ = spearmanr(merged["aadt_segment_count"], merged["crash_density"])

    print(f"\n  Spearman correlation with crash_density:")
    print(f"    aadt_mean            r = {r_mean:+.4f}")
    print(f"    aadt_max             r = {r_max:+.4f}")
    print(f"    aadt_segment_count   r = {r_count:+.4f}")

    # Top 5 highest AADT hexagons
    top5 = merged.nlargest(5, "aadt_max")[
        ["h3_index", "aadt_mean", "aadt_max", "aadt_segment_count", "crash_density"]
    ]
    print(f"\n  Top 5 highest-AADT hexagons:")
    print(f"  {'h3_index':<20} {'aadt_mean':>9} {'aadt_max':>8} {'segs':>4} {'crash_density':>13}")
    for _, row in top5.iterrows():
        print(f"  {row['h3_index']:<20} {row['aadt_mean']:>9,.0f} {row['aadt_max']:>8,.0f} "
              f"{row['aadt_segment_count']:>4} {row['crash_density']:>13.1f}")

    # Distribution of aadt_mean for covered hexes
    covered = merged[merged["aadt_mean"] > 0]["aadt_mean"]
    if len(covered) > 0:
        print(f"\n  aadt_mean distribution (covered hexagons only, n={len(covered)}):")
        percentiles = [25, 50, 75, 90, 95]
        for p in percentiles:
            print(f"    p{p:02d}  {np.percentile(covered, p):>10,.0f} vehicles/day")
        print(f"    max  {covered.max():>10,.0f} vehicles/day")

    print(f"\n{'='*60}")
    return r_mean


# ── main per-city ─────────────────────────────────────────────────────────────

def process_city(city: str, dry_run: bool = False) -> pd.DataFrame:
    local_path = DATA / "silver" / "aadt" / f"{city}_aadt_hex.parquet"
    s3_key     = f"silver/aadt/{city}_aadt_hex.parquet"

    print(f"\n- AADT extraction — {city.upper()} -\n")

    # 1. Load hexagon universe — prefer gold table, fall back to road points
    hex_list = []
    if GOLD_PATH.exists():
        gold = pd.read_parquet(GOLD_PATH)
        hex_list = gold[gold["city"] == city]["h3_index"].tolist()
        print(f"  Hexagons in gold ({city}): {len(hex_list):,}")

    if not hex_list:
        road_path = DATA / "bronze" / f"{city}_road_points.parquet"
        if not road_path.exists():
            raise FileNotFoundError(f"Neither gold table nor road points found for {city}")
        roads = pd.read_parquet(road_path)
        hex_list = roads["h3_index"].unique().tolist()
        print(f"  [fallback] Hexagons from road points ({city}): {len(hex_list):,}")

    # 2. Download AADT segments
    segments = download_segments(city, dry_run=dry_run)
    print(f"  Total valid segments downloaded: {len(segments):,}")

    if not segments:
        print(f"  [warn] No segments downloaded for {city}. Returning zeros.")
        df = pd.DataFrame([
            {"h3_index": h, "aadt_mean": 0.0, "aadt_max": 0.0, "aadt_segment_count": 0}
            for h in hex_list
        ])
        return df

    # 3. Spatial join
    print(f"  Joining {len(segments):,} segments to {len(hex_list):,} hexagons ...")
    df = join_segments_to_hexes(hex_list, segments)

    # 4. Analyze
    r_mean = analyze(city, df, dry_run=dry_run)

    # 5. Save only if not dry-run and correlation is meaningful
    if dry_run:
        print(f"\n  [dry-run] Skipping save and upload.")
        return df

    threshold = 0.25
    if r_mean is not None and abs(r_mean) < threshold:
        print(f"\n  [warn] aadt_mean Spearman ({r_mean:+.4f}) < {threshold} — skipping save.")
        return df

    out = df[["h3_index", "aadt_mean", "aadt_max", "aadt_segment_count"]].copy()
    save_parquet(out, local_path)
    s3 = make_s3()
    upload(local_path, s3, s3_key)

    return df


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract FDOT AADT traffic volume features per H3 hexagon."
    )
    parser.add_argument("--city", choices=["sarasota", "tampa", "orlando"], default=None,
                        help="City to process. Omit for all configured cities.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch first page only (1000 segments), run analysis, skip save.")
    args = parser.parse_args()

    cities = [args.city] if args.city else ["sarasota", "tampa"]
    for city in cities:
        process_city(city, dry_run=args.dry_run)

    print("\n- Done. -\n")


if __name__ == "__main__":
    main()

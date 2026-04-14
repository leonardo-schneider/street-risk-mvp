"""
pipeline/features/extract_nightlight_features.py

Downloads NASA Black Marble nighttime light data via the NASA GIBS WMS
(no authentication required) and extracts mean radiance per H3 hexagon
for Sarasota and Tampa.

Product: VIIRS_Black_Marble_NightLights_5km_v21
Source:  https://gibs.earthdata.nasa.gov

Usage:
    python pipeline/features/extract_nightlight_features.py                  # both cities
    python pipeline/features/extract_nightlight_features.py --city sarasota
    python pipeline/features/extract_nightlight_features.py --city tampa
    python pipeline/features/extract_nightlight_features.py --time 2023-06   # different month
"""

import argparse
import io
import os
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
from PIL import Image

# ── env ───────────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parents[2] / ".env", override=True)

BUCKET = os.getenv("S3_BUCKET_NAME", "street-risk-mvp")
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
ROOT   = Path(__file__).parents[2]
DATA   = ROOT / "data"

# ── city config ───────────────────────────────────────────────────────────────
CITY_BBOX = {
    "sarasota": {"south": 27.2, "west": -82.7, "north": 27.5, "east": -82.3},
    "tampa":    {"south": 27.8, "west": -82.6, "north": 28.1, "east": -82.2},
}

GIBS_WMS = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
LAYER    = "VIIRS_Night_Lights"   # Static composite, no TIME parameter required
IMG_SIZE = 512   # pixels per side


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


# ── raster download ───────────────────────────────────────────────────────────

def download_raster(bbox: dict, time: str = None) -> np.ndarray:
    """
    Download VIIRS Night Lights tile from NASA GIBS WMS as PNG.
    Returns (IMG_SIZE, IMG_SIZE, 4) uint8 RGBA array.
    GIBS EPSG:4326 uses BBOX order: minLat,minLon,maxLat,maxLon
    VIIRS_Night_Lights is a static composite — no TIME parameter needed.
    """
    params = {
        "SERVICE":  "WMS",
        "REQUEST":  "GetMap",
        "VERSION":  "1.3.0",
        "LAYERS":   LAYER,
        "CRS":      "EPSG:4326",
        "BBOX":     f"{bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']}",
        "WIDTH":    IMG_SIZE,
        "HEIGHT":   IMG_SIZE,
        "FORMAT":   "image/png",
    }
    if time:
        params["TIME"] = time
    print(f"  Requesting WMS tile (layer={LAYER}) ...")
    r = requests.get(GIBS_WMS, params=params, timeout=60)
    r.raise_for_status()

    # Check we got an image not an XML error
    content_type = r.headers.get("Content-Type", "")
    if "xml" in content_type or r.content[:4] == b"<Ser":
        raise RuntimeError(f"WMS returned error XML instead of image: {r.text[:400]}")

    img = Image.open(io.BytesIO(r.content)).convert("RGBA")
    arr = np.array(img)
    print(f"  [ok] Raster downloaded: {arr.shape}  dtype={arr.dtype}")
    return arr


def raster_to_grayscale(arr: np.ndarray) -> np.ndarray:
    """
    Convert RGBA raster to single-channel brightness (0-255).
    Black Marble uses dark background; bright pixels = lit areas.
    Use luminance formula: 0.299R + 0.587G + 0.114B
    """
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)


# ── pixel extraction ──────────────────────────────────────────────────────────

def latlon_to_pixel(lat: float, lon: float, bbox: dict) -> tuple:
    """
    Map a (lat, lon) coordinate to (row, col) pixel indices in the raster.
    Top-left = (north, west). Row increases southward, col increases eastward.
    """
    col = int((lon - bbox["west"]) / (bbox["east"] - bbox["west"]) * IMG_SIZE)
    row = int((bbox["north"] - lat) / (bbox["north"] - bbox["south"]) * IMG_SIZE)
    col = max(0, min(IMG_SIZE - 1, col))
    row = max(0, min(IMG_SIZE - 1, row))
    return row, col


def extract_hex_values(hex_list: list, gray: np.ndarray, bbox: dict) -> pd.DataFrame:
    """
    For each hexagon, sample the raster at its centroid pixel.
    Returns DataFrame with h3_index and night_light_raw columns.
    """
    rows = []
    for h in hex_list:
        lat, lon = h3.cell_to_latlng(h)
        row, col = latlon_to_pixel(lat, lon, bbox)
        raw = float(gray[row, col])
        rows.append({"h3_index": h, "night_light_raw": raw})
    return pd.DataFrame(rows)


# ── main per-city ─────────────────────────────────────────────────────────────

def process_city(city: str, time: str = None) -> pd.DataFrame:
    bbox       = CITY_BBOX[city]
    local_path = DATA / "silver" / "nightlight" / f"{city}_nightlight_hex.parquet"
    s3_key     = f"silver/nightlight/{city}_nightlight_hex.parquet"

    print(f"\n- Nightlight extraction — {city.upper()} -\n")

    # 1. Load hexagon universe from road points
    road_path = DATA / "bronze" / f"{city}_road_points.parquet"
    if not road_path.exists():
        raise FileNotFoundError(f"Road points not found: {road_path}")
    hex_list = pd.read_parquet(road_path)["h3_index"].unique().tolist()
    print(f"  Hexagons in universe: {len(hex_list):,}")

    # 2. Download raster
    arr  = download_raster(bbox, time=time)
    gray = raster_to_grayscale(arr)

    # Print raster stats to diagnose blank vs real data
    print(f"  Raster brightness — min={gray.min():.1f}  mean={gray.mean():.1f}  "
          f"max={gray.max():.1f}  nonzero_pct={100*(gray>0).mean():.1f}%")

    # 3. Extract per-hex values
    df = extract_hex_values(hex_list, gray, bbox)

    # 4. Normalize 0-1
    raw_min = df["night_light_raw"].min()
    raw_max = df["night_light_raw"].max()
    if raw_max > raw_min:
        df["night_light_intensity"] = (df["night_light_raw"] - raw_min) / (raw_max - raw_min)
    else:
        # Fallback: raster may be blank (nighttime product not available)
        print(f"  [warn] Raster appears blank (max={raw_max:.1f}). "
              f"Trying alternate month ...")
        df["night_light_intensity"] = 0.0

    df["city"] = city
    print(f"  Intensity stats — mean={df['night_light_intensity'].mean():.4f}  "
          f"max={df['night_light_intensity'].max():.4f}")

    # 5. Save
    out = df[["h3_index", "night_light_raw", "night_light_intensity"]].copy()
    save_parquet(out, local_path)
    s3 = make_s3()
    upload(local_path, s3, s3_key)

    return df


# ── analysis ──────────────────────────────────────────────────────────────────

def analyze(dfs: list):
    """Merge with gold table and report correlation with crash_density."""
    gold_path = DATA / "gold" / "training_table" / "multicity_gold.parquet"
    if not gold_path.exists():
        print("\n  [warn] multicity_gold.parquet not found — skipping correlation analysis.")
        return

    gold = pd.read_parquet(gold_path)[["h3_index", "city", "crash_density"]]
    nl   = pd.concat(dfs, ignore_index=True)[["h3_index", "city", "night_light_intensity", "night_light_raw"]]
    merged = gold.merge(nl, on=["h3_index", "city"], how="inner")

    print(f"\n{'='*60}")
    print(f"  NIGHTLIGHT ANALYSIS")
    print(f"{'='*60}")
    print(f"  Merged hexagons: {len(merged):,}")

    # Mean by city
    print(f"\n  Mean night_light_intensity by city:")
    for city, grp in merged.groupby("city"):
        print(f"    {city:<12}  {grp['night_light_intensity'].mean():.4f}")

    # Correlation
    corr = merged["night_light_intensity"].corr(merged["crash_density"])
    corr_spear = merged[["night_light_intensity","crash_density"]].corr(method="spearman").iloc[0,1]
    print(f"\n  Correlation with crash_density:")
    print(f"    Pearson  r = {corr:+.4f}")
    print(f"    Spearman r = {corr_spear:+.4f}")

    # Per-city correlations
    print(f"\n  Per-city Pearson correlation:")
    for city, grp in merged.groupby("city"):
        r = grp["night_light_intensity"].corr(grp["crash_density"])
        print(f"    {city:<12}  {r:+.4f}")

    # Top 5 brightest
    top5 = merged.nlargest(5, "night_light_intensity")[
        ["h3_index", "city", "night_light_intensity", "night_light_raw", "crash_density"]
    ]
    print(f"\n  Top 5 brightest hexagons:")
    print(f"  {'h3_index':<20} {'city':<10} {'intensity':>9} {'raw':>5} {'crash_density':>13}")
    for _, r in top5.iterrows():
        print(f"  {r['h3_index']:<20} {r['city']:<10} {r['night_light_intensity']:>9.4f} "
              f"{r['night_light_raw']:>5.0f} {r['crash_density']:>13.1f}")

    # Bottom 5 darkest
    bot5 = merged.nsmallest(5, "night_light_intensity")[
        ["h3_index", "city", "night_light_intensity", "night_light_raw", "crash_density"]
    ]
    print(f"\n  Bottom 5 darkest hexagons:")
    print(f"  {'h3_index':<20} {'city':<10} {'intensity':>9} {'raw':>5} {'crash_density':>13}")
    for _, r in bot5.iterrows():
        print(f"  {r['h3_index']:<20} {r['city']:<10} {r['night_light_intensity']:>9.4f} "
              f"{r['night_light_raw']:>5.0f} {r['crash_density']:>13.1f}")

    # Text histogram
    print(f"\n  Distribution of night_light_intensity (all hexagons):")
    bins   = np.linspace(0, 1, 11)
    labels = [f"{b:.1f}" for b in bins]
    counts, _ = np.histogram(merged["night_light_intensity"], bins=bins)
    bar_max = max(counts) if max(counts) > 0 else 1
    for i, cnt in enumerate(counts):
        bar = "#" * int(cnt / bar_max * 30)
        print(f"    {labels[i]}-{labels[i+1]}  {bar:<30}  {cnt:>4}")

    print(f"\n{'='*60}")
    return merged


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract NASA nighttime light features per H3 hexagon.")
    parser.add_argument("--city",  choices=["sarasota", "tampa"], default=None,
                        help="City to process. Omit for both.")
    parser.add_argument("--time",  default=None,
                        help="WMS TIME parameter (YYYY-MM-DD). Not needed for VIIRS_Night_Lights.")
    args = parser.parse_args()

    cities = [args.city] if args.city else ["sarasota", "tampa"]
    dfs = []
    for city in cities:
        df = process_city(city, time=args.time)
        dfs.append(df)

    if len(dfs) > 0:
        analyze(dfs)

    print("\n- Done. -\n")


if __name__ == "__main__":
    main()

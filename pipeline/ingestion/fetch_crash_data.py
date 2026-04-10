"""
pipeline/ingestion/fetch_crash_data.py

Downloads crash records from the FDOT Open Data Hub ArcGIS REST endpoint
for a given city/county, assigns H3 res-9 indices, aggregates to hexagon
level, and writes Bronze + Silver Parquet files to local disk and S3.

Usage:
    python pipeline/ingestion/fetch_crash_data.py                      # sarasota full
    python pipeline/ingestion/fetch_crash_data.py --dry-run            # first page only
    python pipeline/ingestion/fetch_crash_data.py --city tampa         # tampa full
    python pipeline/ingestion/fetch_crash_data.py --city tampa --dry-run
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
import requests
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from tqdm import tqdm

# ── env ───────────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parents[2] / ".env", override=True)

BUCKET = os.getenv("S3_BUCKET_NAME", "street-risk-mvp")
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# ── city → county mapping ─────────────────────────────────────────────────────
CITY_COUNTY = {
    "sarasota": "SARASOTA",
    "tampa":    "HILLSBOROUGH",
}

# ── FDOT endpoint ─────────────────────────────────────────────────────────────
FDOT_URL  = "https://gis.fdot.gov/arcgis/rest/services/sso/ssogis/MapServer/2000/query"
PAGE_SIZE = 5000
H3_RES    = 9
HEX_AREA  = 0.1059   # km², H3 res-9 average area

FIELDS = [
    "SAFETYLAT", "SAFETYLON", "CALENDAR_YEAR", "CRASH_DATE",
    "IMPCT_TYP_CD", "INJSEVER", "COUNTY_TXT",
    "NUMBER_OF_INJURED", "NUMBER_OF_KILLED",
]


# ── fetch ─────────────────────────────────────────────────────────────────────

def fetch_page(county: str, offset: int) -> tuple:
    params = {
        "where":             f"COUNTY_TXT='{county}'",
        "outFields":         ",".join(FIELDS),
        "f":                 "json",
        "resultOffset":      offset,
        "resultRecordCount": PAGE_SIZE,
    }
    resp = requests.get(FDOT_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        raise RuntimeError(f"FDOT API error: {data['error']}")

    features = data.get("features", [])
    records  = [feat["attributes"] for feat in features]
    exceeded = data.get("exceededTransferLimit", False)
    return records, exceeded


def download_all(county: str, dry_run: bool = False) -> pd.DataFrame:
    all_records = []
    offset = 0

    with tqdm(desc=f"Downloading crash pages ({county})", unit="page") as pbar:
        while True:
            records, exceeded = fetch_page(county, offset)
            all_records.extend(records)
            pbar.update(1)
            pbar.set_postfix(total=len(all_records))

            if dry_run or not exceeded:
                break
            offset += PAGE_SIZE

    return pd.DataFrame(all_records)


# ── transform ─────────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.upper() for c in df.columns]

    df = df.dropna(subset=["SAFETYLAT", "SAFETYLON"])
    df = df[
        (df["SAFETYLAT"] != 0) & (df["SAFETYLON"] != 0) &
        (df["SAFETYLAT"].between(-90, 90)) &
        (df["SAFETYLON"].between(-180, 180))
    ].copy()

    df["SAFETYLAT"] = df["SAFETYLAT"].astype(float)
    df["SAFETYLON"] = df["SAFETYLON"].astype(float)
    return df


def assign_h3(df: pd.DataFrame) -> pd.DataFrame:
    tqdm.pandas(desc="Assigning H3 index")
    df["h3_index"] = df.progress_apply(
        lambda r: h3.latlng_to_cell(r["SAFETYLAT"], r["SAFETYLON"], H3_RES),
        axis=1,
    )
    return df


def aggregate_hex(df: pd.DataFrame) -> pd.DataFrame:
    df["injured"] = df["INJSEVER"].fillna(5).apply(lambda v: 0 if int(v) == 5 else 1)

    agg = (
        df.groupby("h3_index")
          .agg(
              crash_count  = ("h3_index", "count"),
              injury_count = ("injured", "sum"),
          )
          .reset_index()
    )
    agg["crash_density"] = agg["crash_count"] / HEX_AREA
    agg["injury_rate"]   = agg["injury_count"] / agg["crash_count"]
    return agg


# ── S3 ────────────────────────────────────────────────────────────────────────

def s3_client():
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


def save_parquet(df: pd.DataFrame, local_path: Path, s3_key: str, dry_run: bool, client):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), local_path)
    print(f"  [ok] Saved locally: {local_path}")

    if dry_run:
        print(f"  (--dry-run) Skipping S3 upload for {s3_key}")
        return

    if s3_key_exists(client, BUCKET, s3_key):
        print(f"  [overwrite] s3://{BUCKET}/{s3_key}")

    client.upload_file(str(local_path), BUCKET, s3_key)
    print(f"  [ok] Uploaded: s3://{BUCKET}/{s3_key}")


# ── main ──────────────────────────────────────────────────────────────────────

def main(city: str = "sarasota", dry_run: bool = False):
    county = CITY_COUNTY[city]
    data_root = Path(__file__).parents[2] / "data"

    raw_local = data_root / "bronze" / "crash" / f"{city}_crashes_raw.parquet"
    hex_local = data_root / "silver" / "crash_hex" / f"{city}_crash_hex.parquet"
    raw_s3key = f"bronze/crash/{city}_crashes_raw.parquet"
    hex_s3key = f"silver/crash_hex/{city}_crash_hex.parquet"

    print(f"\n- Crash data ingestion — {city.upper()} (county: {county}) -\n")

    # 1. Download
    print("Step 1/5  Downloading crash records from FDOT ...")
    raw_df = download_all(county, dry_run=dry_run)
    print(f"  [ok] {len(raw_df):,} raw records downloaded")

    # 2. Clean
    print("\nStep 2/5  Cleaning and filtering ...")
    df = clean(raw_df)
    dropped = len(raw_df) - len(df)
    print(f"  [ok] {len(df):,} records with valid coordinates ({dropped} dropped)")

    # 3. Assign H3
    print("\nStep 3/5  Assigning H3 res-9 index ...")
    df = assign_h3(df)

    # 4. Aggregate
    print("\nStep 4/5  Aggregating to hexagon level ...")
    hex_df = aggregate_hex(df)
    n_hexes = len(hex_df)
    print(f"  [ok] {n_hexes:,} hexagons with crash data")

    # 5. Save
    print("\nStep 5/5  Saving ...")
    client = s3_client()
    save_parquet(df, raw_local, raw_s3key, dry_run, client)
    save_parquet(hex_df, hex_local, hex_s3key, dry_run, client)

    print("\n--- Summary ---")
    print(f"  City                   : {city}")
    print(f"  County                 : {county}")
    print(f"  Total crashes          : {len(df):,}")
    print(f"  Hexagons with crashes  : {n_hexes:,}")
    print(f"  crash_density (crashes/km2):")
    print(f"    min  = {hex_df['crash_density'].min():.2f}")
    print(f"    mean = {hex_df['crash_density'].mean():.2f}")
    print(f"    max  = {hex_df['crash_density'].max():.2f}")
    print(f"\n  Hex aggregates preview (top 10 by density):")
    print(hex_df.sort_values("crash_density", ascending=False).head(10).to_string(index=False))
    print()

    print("- Done. -\n")
    return df, hex_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and aggregate crash data from FDOT."
    )
    parser.add_argument(
        "--city",
        choices=["sarasota", "tampa"],
        default="sarasota",
        help="City to download crash data for (default: sarasota).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download first page only (~5000 records), skip S3 upload.",
    )
    args = parser.parse_args()
    main(city=args.city, dry_run=args.dry_run)

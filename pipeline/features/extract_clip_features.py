"""
pipeline/features/extract_clip_features.py

Batch CLIP inference on Street View images stored in S3 Bronze.
Scores each image against 7 road-risk concepts, aggregates to hexagon
level, and writes Silver Parquet files to local disk and S3.

Usage:
    python pipeline/features/extract_clip_features.py              # sarasota full
    python pipeline/features/extract_clip_features.py --city tampa
    python pipeline/features/extract_clip_features.py --dry-run
    python pipeline/features/extract_clip_features.py --batch-size 64
"""

import argparse
import io
import os
from pathlib import Path

import boto3
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# ── env ───────────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parents[2] / ".env", override=True)

BUCKET = os.getenv("S3_BUCKET_NAME", "street-risk-mvp")
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

DATA_ROOT = Path(__file__).parents[2] / "data"

# ── CLIP config ───────────────────────────────────────────────────────────────
MODEL_NAME = "openai/clip-vit-base-patch32"

RISK_CONCEPTS = [
    "a road with heavy traffic",
    "a road with poor lighting",
    "a road with no sidewalks or pedestrian infrastructure",
    "a wet or damaged road surface",
    "a clear, well-maintained suburban road",
    "an intersection with no traffic signals",
    "a road with parked cars blocking visibility",
]

CONCEPT_COLS = [
    "clip_heavy_traffic",
    "clip_poor_lighting",
    "clip_no_sidewalks",
    "clip_damaged_road",
    "clip_clear_road",
    "clip_no_signals",
    "clip_parked_cars",
]

DEVICE = "cpu"


# ── city manifest mapping ─────────────────────────────────────────────────────

def manifest_path_for(city: str) -> Path:
    """
    Sarasota uses the legacy image_manifest.csv which now contains all
    4 headings (0/90 legacy + 180/270 new). Tampa uses its own manifest.
    """
    if city == "sarasota":
        return DATA_ROOT / "bronze" / "image_manifest.csv"
    return DATA_ROOT / "bronze" / f"image_manifest_{city}.csv"


# ── S3 ────────────────────────────────────────────────────────────────────────

def make_s3_client():
    return boto3.client(
        "s3",
        region_name=REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


def download_image(s3, bucket: str, key: str) -> Image.Image:
    buf = io.BytesIO()
    s3.download_fileobj(bucket, key, buf)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def s3_key_exists(s3, bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def upload_parquet(local_path: Path, s3, bucket: str, key: str):
    if s3_key_exists(s3, bucket, key):
        print(f"  [overwrite] s3://{bucket}/{key}")
    s3.upload_file(str(local_path), bucket, key)
    print(f"  [ok] Uploaded: s3://{bucket}/{key}")


def save_parquet(df: pd.DataFrame, local_path: Path):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), local_path)
    print(f"  [ok] Saved locally: {local_path}")


# ── CLIP inference ────────────────────────────────────────────────────────────

def load_clip(device: str):
    print(f"  Loading {MODEL_NAME} on {device} ...")
    model     = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    return model, processor


def encode_texts(model, processor, device: str) -> torch.Tensor:
    inputs = processor(text=RISK_CONCEPTS, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_emb = model.get_text_features(**inputs)
    return F.normalize(text_emb, dim=-1)


def score_batch(images, model, processor, text_emb, device) -> np.ndarray:
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        img_emb = model.get_image_features(**inputs)
    img_emb = F.normalize(img_emb, dim=-1)
    logits  = img_emb @ text_emb.T
    return F.softmax(logits, dim=-1).cpu().numpy()


def extract_embeddings_batch(images, model, processor, device) -> np.ndarray:
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    return F.normalize(emb, dim=-1).cpu().numpy()


# ── main ──────────────────────────────────────────────────────────────────────

def main(city: str = "sarasota", dry_run: bool = False, batch_size: int = 32):
    print(f"\n- CLIP feature extraction — {city.upper()} -\n")

    manifest_csv    = manifest_path_for(city)
    img_feat_local  = DATA_ROOT / "silver" / "image_features" / f"{city}_clip_features.parquet"
    hex_feat_local  = DATA_ROOT / "silver" / "image_features" / f"{city}_clip_hex.parquet"
    img_feat_s3key  = f"silver/image_features/{city}_clip_features.parquet"
    hex_feat_s3key  = f"silver/image_features/{city}_clip_hex.parquet"

    # 1. Load manifest
    print("Step 1/5  Loading image manifest ...")
    if not manifest_csv.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_csv}\n"
            f"Run `python pipeline/ingestion/fetch_images.py --city {city}` first."
        )
    manifest = pd.read_csv(manifest_csv)
    manifest = manifest[manifest["status"].isin(["ok", "cached"])].reset_index(drop=True)
    print(f"  [ok] {len(manifest)} images to process  "
          f"({manifest['heading'].value_counts().to_dict()})")

    if dry_run:
        manifest = manifest.head(10)
        print(f"  (--dry-run) Limited to {len(manifest)} images")

    # 2. Load CLIP
    print("\nStep 2/5  Loading CLIP model ...")
    model, processor = load_clip(DEVICE)
    text_emb = encode_texts(model, processor, DEVICE)
    print(f"  [ok] Text embeddings computed for {len(RISK_CONCEPTS)} concepts")

    # 3. S3 client
    s3 = make_s3_client()

    # 4. Batch inference
    print(f"\nStep 3/5  Running CLIP inference (batch_size={batch_size}) ...")
    rows = []
    n = len(manifest)

    with tqdm(total=n, desc="Images", unit="img") as pbar:
        for batch_start in range(0, n, batch_size):
            batch_meta = manifest.iloc[batch_start : batch_start + batch_size]
            images, meta = [], []

            for _, row in batch_meta.iterrows():
                try:
                    img = download_image(s3, BUCKET, row["s3_key"])
                    images.append(img)
                    meta.append(row)
                except Exception as e:
                    tqdm.write(f"  [warn] Failed {row['s3_key']}: {e}")

            if not images:
                pbar.update(len(batch_meta))
                continue

            scores = score_batch(images, model, processor, text_emb, DEVICE)

            for i, row in enumerate(meta):
                record = {"h3_index": row["h3_index"], "heading": row["heading"]}
                for col, score in zip(CONCEPT_COLS, scores[i]):
                    record[col] = float(score)
                rows.append(record)

            pbar.update(len(images))

    img_df = pd.DataFrame(rows)
    print(f"  [ok] {len(img_df)} images scored  ({img_df['heading'].value_counts().to_dict()})")

    # 5. Aggregate to hex level (mean across all headings)
    print("\nStep 4/5  Aggregating to hexagon level ...")
    hex_df = (
        img_df.groupby("h3_index")[CONCEPT_COLS]
              .mean()
              .reset_index()
    )
    print(f"  [ok] {len(hex_df)} hexagons covered")

    # 6. Save
    print("\nStep 5/5  Saving ...")
    save_parquet(img_df, img_feat_local)
    save_parquet(hex_df, hex_feat_local)

    if dry_run:
        print("  (--dry-run) Skipping S3 upload.")
    else:
        upload_parquet(img_feat_local, s3, BUCKET, img_feat_s3key)
        upload_parquet(hex_feat_local, s3, BUCKET, hex_feat_s3key)

    print("\n--- Summary ---")
    print(f"  City             : {city}")
    print(f"  Images processed : {len(img_df)}")
    print(f"  Hexagons covered : {len(hex_df)}")
    print(f"\n  Mean concept scores:")
    for col, concept in zip(CONCEPT_COLS, RISK_CONCEPTS):
        print(f"    {col:<28} {img_df[col].mean():.4f}  ({concept})")

    print(f"\n  Hex-level preview (top 5 by clip_heavy_traffic):")
    print(hex_df.sort_values("clip_heavy_traffic", ascending=False).head(5).to_string(index=False))
    print(f"\n- Done. -\n")
    return img_df, hex_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract CLIP features from Street View images in S3."
    )
    parser.add_argument("--city",       choices=["sarasota", "tampa", "orlando"], default="sarasota")
    parser.add_argument("--dry-run",    action="store_true", help="Process 10 images only.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--use-probe",  action="store_true", help="Use trained probe (sarasota only).")
    args = parser.parse_args()

    if args.use_probe:
        # probe_mode only makes sense for sarasota (probe was trained on sarasota labels)
        from pipeline.features.extract_clip_features import probe_mode  # noqa: F401
        raise NotImplementedError("--use-probe not yet updated for multi-city; run train_clip_probe.py directly.")
    else:
        main(city=args.city, dry_run=args.dry_run, batch_size=args.batch_size)

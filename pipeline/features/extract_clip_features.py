"""
pipeline/features/extract_clip_features.py

Batch CLIP inference on Street View images stored in S3 Bronze.
Scores each image against 7 road-risk concepts, aggregates to hexagon
level, and writes Silver Parquet files to local disk and S3.

Usage:
    python pipeline/features/extract_clip_features.py              # full run
    python pipeline/features/extract_clip_features.py --dry-run    # 10 images
    python pipeline/features/extract_clip_features.py --batch-size 64
"""

import argparse
import io
import os
from pathlib import Path

import boto3
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

# ── paths ─────────────────────────────────────────────────────────────────────
MANIFEST_PATH  = Path(__file__).parents[2] / "data" / "bronze" / "image_manifest.csv"
IMG_FEAT_LOCAL = Path(__file__).parents[2] / "data" / "silver" / "image_features" / "sarasota_clip_features.parquet"
HEX_FEAT_LOCAL = Path(__file__).parents[2] / "data" / "silver" / "image_features" / "sarasota_clip_hex.parquet"
IMG_FEAT_S3KEY = "silver/image_features/sarasota_clip_features.parquet"
HEX_FEAT_S3KEY = "silver/image_features/sarasota_clip_hex.parquet"

# ── CLIP config ───────────────────────────────────────────────────────────────
MODEL_NAME = "openai/clip-vit-base-patch32"

# Canonical list from CLAUDE.md — do not change without discussion
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
    """Pre-compute normalised text embeddings for all risk concepts (once)."""
    inputs = processor(text=RISK_CONCEPTS, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_emb = model.get_text_features(**inputs)
    text_emb = F.normalize(text_emb, dim=-1)   # (7, D)
    return text_emb


def score_batch(
    images: list,
    model,
    processor,
    text_emb: torch.Tensor,
    device: str,
) -> np.ndarray:
    """
    Run CLIP on a batch of PIL images.
    Returns softmax-normalised cosine similarity scores, shape (N, 7).
    """
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        img_emb = model.get_image_features(**inputs)
    img_emb = F.normalize(img_emb, dim=-1)          # (N, D)
    logits  = img_emb @ text_emb.T                  # (N, 7) cosine similarities
    scores  = F.softmax(logits, dim=-1).cpu().numpy()
    return scores


# ── main ──────────────────────────────────────────────────────────────────────

def main(dry_run: bool = False, batch_size: int = 32):
    print("\n- CLIP feature extraction -\n")

    # 1. Load manifest
    print("Step 1/5  Loading image manifest ...")
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest = manifest[manifest["status"] == "ok"].reset_index(drop=True)
    print(f"  [ok] {len(manifest)} images to process")

    if dry_run:
        manifest = manifest.head(10)
        print(f"  (--dry-run) Limited to {len(manifest)} images")

    # 2. Load CLIP
    print("\nStep 2/5  Loading CLIP model ...")
    device = "cpu"
    model, processor = load_clip(device)
    text_emb = encode_texts(model, processor, device)
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
                    tqdm.write(f"  [warn] Failed to download {row['s3_key']}: {e}")

            if not images:
                pbar.update(len(batch_meta))
                continue

            scores = score_batch(images, model, processor, text_emb, device)

            for i, row in enumerate(meta):
                record = {
                    "h3_index": row["h3_index"],
                    "heading":  row["heading"],
                }
                for col, score in zip(CONCEPT_COLS, scores[i]):
                    record[col] = float(score)
                rows.append(record)

            pbar.update(len(images))

    img_df = pd.DataFrame(rows)
    print(f"  [ok] {len(img_df)} images scored")

    # 5. Aggregate to hex level
    print("\nStep 4/5  Aggregating to hexagon level ...")
    hex_df = (
        img_df.groupby("h3_index")[CONCEPT_COLS]
              .mean()
              .reset_index()
    )
    print(f"  [ok] {len(hex_df)} hexagons covered")

    # 6. Save
    print("\nStep 5/5  Saving ...")
    save_parquet(img_df, IMG_FEAT_LOCAL)
    save_parquet(hex_df, HEX_FEAT_LOCAL)

    if dry_run:
        print("  (--dry-run) Skipping S3 upload.")
    else:
        upload_parquet(IMG_FEAT_LOCAL, s3, BUCKET, IMG_FEAT_S3KEY)
        upload_parquet(HEX_FEAT_LOCAL, s3, BUCKET, HEX_FEAT_S3KEY)

    # Summary
    print("\n--- Summary ---")
    print(f"  Images processed : {len(img_df)}")
    print(f"  Hexagons covered : {len(hex_df)}")
    print(f"\n  Mean concept scores across all images:")
    for col, concept in zip(CONCEPT_COLS, RISK_CONCEPTS):
        mean_score = img_df[col].mean()
        print(f"    {col:<28} {mean_score:.4f}  ({concept})")

    print(f"\n  Image-level preview (first 5 rows):")
    print(img_df.head(5).to_string(index=False))

    print(f"\n  Hex-level preview (top 5 by clip_heavy_traffic):")
    print(
        hex_df.sort_values("clip_heavy_traffic", ascending=False)
              .head(5)
              .to_string(index=False)
    )

    print(f"\n- Done. -\n")
    return img_df, hex_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract CLIP features from Street View images in S3."
    )
    parser.add_argument("--dry-run",    action="store_true", help="Process 10 images only.")
    parser.add_argument("--batch-size", type=int, default=32, help="CLIP inference batch size.")
    args = parser.parse_args()
    main(dry_run=args.dry_run, batch_size=args.batch_size)

"""
model/train_clip_probe.py — CLIP linear probe training and evaluation.

Trains a logistic regression linear probe on top of frozen CLIP embeddings
to classify images as high/low crash-risk, then runs inference over all 770
images, aggregates to hexagon level, rebuilds the Gold table with the probe
feature, retrains LightGBM, and compares Spearman vs the zero-shot baseline.

Usage:
    python model/train_clip_probe.py
"""

import io
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import boto3
import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from model.train import (
    GOLD_PATH, LGBM_PARAMS, TARGET,
    spatial_split, evaluate,
)

# ── env ───────────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parents[1] / ".env", override=True)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

BUCKET = os.getenv("S3_BUCKET_NAME", "street-risk-mvp")
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT             = Path(__file__).parents[1]
LABELS_PATH      = ROOT / "data" / "labels" / "image_labels_v1_backup.csv"
MANIFEST_PATH    = ROOT / "data" / "bronze" / "image_manifest.csv"
PROBE_PKL        = ROOT / "model" / "clip_probe.pkl"
PROBE_HEX_LOCAL  = ROOT / "data" / "silver" / "image_features" / "sarasota_clip_probe_hex.parquet"
PROBE_HEX_S3KEY  = "silver/image_features/sarasota_clip_probe_hex.parquet"
GOLD_PROBE_LOCAL = ROOT / "data" / "gold" / "training_table" / "sarasota_gold_probe.parquet"
GOLD_PROBE_S3KEY = "gold/training_table/sarasota_gold_probe.parquet"

MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE     = "cpu"

PROBE_NUMERIC = [
    "clip_risk_prob",
    "speed_limit_mean", "lanes_mean", "dist_to_intersection_mean", "point_count",
]


# ── S3 helpers ────────────────────────────────────────────────────────────────

def make_s3():
    return boto3.client(
        "s3",
        region_name=REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


def download_image(s3, key: str) -> Image.Image:
    buf = io.BytesIO()
    s3.download_fileobj(BUCKET, key, buf)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def upload(local_path: Path, s3_key: str, s3):
    s3.upload_file(str(local_path), BUCKET, s3_key)
    print(f"  [ok] Uploaded: s3://{BUCKET}/{s3_key}")


def save_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)
    print(f"  [ok] Saved: {path}")


# ── CLIP embedding extraction ─────────────────────────────────────────────────

def load_clip():
    model     = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    return model, processor


def extract_embeddings(images: list, model, processor) -> np.ndarray:
    inputs = processor(images=images, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    return F.normalize(emb, dim=-1).cpu().numpy().astype(np.float32)


def batch_embed(keys: list, s3, model, processor, batch_size: int = 32):
    all_embs, valid_keys = [], []
    for i in tqdm(range(0, len(keys), batch_size), desc="Embedding", unit="batch"):
        batch_keys = keys[i : i + batch_size]
        imgs, ok_keys = [], []
        for key in batch_keys:
            try:
                imgs.append(download_image(s3, key))
                ok_keys.append(key)
            except Exception as e:
                tqdm.write(f"  [warn] {key}: {e}")
        if imgs:
            all_embs.append(extract_embeddings(imgs, model, processor))
            valid_keys.extend(ok_keys)
    return np.vstack(all_embs), valid_keys


# ── probe helpers ─────────────────────────────────────────────────────────────

def train_probe(X: np.ndarray, y: np.ndarray, keys: list):
    X_tr, X_te, y_tr, y_te, k_tr, k_te = train_test_split(
        X, y, keys, test_size=0.2, stratify=y, random_state=42
    )
    probe = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
    probe.fit(X_tr, y_tr)

    y_pred = probe.predict(X_te)
    y_prob = probe.predict_proba(X_te)[:, 1]
    acc    = accuracy_score(y_te, y_pred)
    f1     = f1_score(y_te, y_pred)
    auc    = roc_auc_score(y_te, y_prob)
    cm     = confusion_matrix(y_te, y_pred)

    print(f"\n  Probe test set ({len(y_te)} images):")
    print(f"    Accuracy : {acc:.4f}")
    print(f"    F1       : {f1:.4f}")
    print(f"    ROC-AUC  : {auc:.4f}")
    print(f"    Confusion matrix (actual rows / predicted cols):")
    print(f"      TN={cm[0,0]:>3}  FP={cm[0,1]:>3}   (actual Low)")
    print(f"      FN={cm[1,0]:>3}  TP={cm[1,1]:>3}   (actual High)")

    wrong = [(k_te[i], "high" if y_te[i] else "low", "high" if y_pred[i] else "low", y_prob[i])
             for i in range(len(y_te)) if y_pred[i] != y_te[i]]
    if wrong:
        print(f"\n  Misclassified ({len(wrong)}):")
        for key, actual, pred, prob in wrong:
            print(f"    {key}  actual={actual}  pred={pred}  prob={prob:.3f}")

    return probe, {"accuracy": acc, "f1": f1, "roc_auc": auc}


def build_probe_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    road_tr = pd.get_dummies(train_df["road_type_primary"], prefix="road", drop_first=True)
    road_te = pd.get_dummies(test_df["road_type_primary"],  prefix="road", drop_first=True)
    road_te = road_te.reindex(columns=road_tr.columns, fill_value=0)

    feat_cols = PROBE_NUMERIC + list(road_tr.columns)
    X_tr = pd.concat([train_df[PROBE_NUMERIC].reset_index(drop=True),
                      road_tr.reset_index(drop=True)], axis=1)
    X_te = pd.concat([test_df[PROBE_NUMERIC].reset_index(drop=True),
                      road_te.reset_index(drop=True)], axis=1)
    return X_tr, X_te, train_df[TARGET].values, test_df[TARGET].values, feat_cols


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n- CLIP linear probe -\n")
    s3 = make_s3()

    # 1. Load labels
    print("Step 1/7  Loading labels ...")
    labels_df = pd.read_csv(LABELS_PATH)
    labels_df["y"] = (labels_df["label"] == "high").astype(int)
    print(f"  [ok] {len(labels_df)} labels  "
          f"(high={labels_df['y'].sum()}, low={(labels_df['y']==0).sum()})")

    # 2. Extract labeled embeddings
    print("\nStep 2/7  Extracting labeled image embeddings ...")
    model, processor = load_clip()
    embs_labeled, valid_keys = batch_embed(
        labels_df["s3_key"].tolist(), s3, model, processor
    )
    key_to_label = dict(zip(labels_df["s3_key"], labels_df["y"]))
    y_labeled = np.array([key_to_label[k] for k in valid_keys])
    print(f"  [ok] {len(valid_keys)} embeddings  shape={embs_labeled.shape}")

    # 3. Train probe
    print("\nStep 3/7  Training logistic regression probe ...")
    probe, probe_metrics = train_probe(embs_labeled, y_labeled, valid_keys)
    joblib.dump(probe, PROBE_PKL)
    print(f"\n  [ok] Probe saved: {PROBE_PKL}")

    # 4. Inference on all 770 images
    print("\nStep 4/7  Probe inference on all 770 images ...")
    manifest = (pd.read_csv(MANIFEST_PATH)
                  .query("status == 'ok'")
                  .drop_duplicates("s3_key")
                  .reset_index(drop=True))
    all_embs, all_keys = batch_embed(manifest["s3_key"].tolist(), s3, model, processor)
    probs = probe.predict_proba(all_embs)[:, 1]

    key_to_h3 = manifest.set_index("s3_key")["h3_index"].to_dict()
    img_probe = pd.DataFrame({
        "s3_key":         all_keys,
        "h3_index":       [key_to_h3.get(k) for k in all_keys],
        "clip_risk_prob": probs.astype(float),
    }).dropna(subset=["h3_index"])

    hex_probe = img_probe.groupby("h3_index")["clip_risk_prob"].mean().reset_index()
    p = hex_probe["clip_risk_prob"]
    print(f"  [ok] {len(hex_probe)} hexagons  "
          f"min={p.min():.3f}  mean={p.mean():.3f}  max={p.max():.3f}")

    # 5. Save probe hex parquet
    print("\nStep 5/7  Saving probe hex features ...")
    save_parquet(hex_probe, PROBE_HEX_LOCAL)
    upload(PROBE_HEX_LOCAL, PROBE_HEX_S3KEY, s3)

    # 6. Rebuild Gold table
    print("\nStep 6/7  Rebuilding Gold table with clip_risk_prob ...")
    gold = pd.read_parquet(GOLD_PATH)
    clip_cols = [c for c in gold.columns if c.startswith("clip_")]
    gold = gold.drop(columns=clip_cols)
    gold = gold.merge(hex_probe[["h3_index", "clip_risk_prob"]], on="h3_index", how="left")
    gold["clip_risk_prob"] = gold["clip_risk_prob"].fillna(gold["clip_risk_prob"].median())
    save_parquet(gold, GOLD_PROBE_LOCAL)
    upload(GOLD_PROBE_LOCAL, GOLD_PROBE_S3KEY, s3)
    print(f"  [ok] Probe gold: {gold.shape}  cols: {list(gold.columns)}")

    # 7. Retrain LightGBM
    print("\nStep 7/7  Retraining LightGBM on probe Gold table ...")
    train_df, test_df = spatial_split(gold)
    X_tr, X_te, y_tr, y_te, feat_cols = build_probe_features(train_df, test_df)
    lgbm_probe = lgb.LGBMRegressor(**LGBM_PARAMS)
    lgbm_probe.fit(X_tr, y_tr)
    m = evaluate(y_te, lgbm_probe.predict(X_te))
    print(f"  [ok] Spearman={m['Spearman_r']:.4f}  RMSE={m['RMSE']:.2f}")

    # MLflow
    mlflow.set_experiment("sarasota-risk-model")
    with mlflow.start_run(run_name="clip-linear-probe") as run:
        mlflow.log_params({
            "probe": "LogisticRegression", "C": 1.0, "max_iter": 1000,
            "clip_backbone": MODEL_NAME, "n_labeled": len(valid_keys),
        })
        mlflow.log_metrics({
            "probe_accuracy": probe_metrics["accuracy"],
            "probe_f1":       probe_metrics["f1"],
            "probe_roc_auc":  probe_metrics["roc_auc"],
            "lgbm_spearman":  m["Spearman_r"],
            "lgbm_rmse":      m["RMSE"],
        })
        mlflow.sklearn.log_model(probe, artifact_path="clip_probe")
        mlflow.lightgbm.log_model(lgbm_probe, artifact_path="lgbm_probe")
        run_id = run.info.run_id

    # Summary
    BASELINE = 0.6664
    delta    = m["Spearman_r"] - BASELINE
    print("\n" + "=" * 52)
    print("  COMPARISON TABLE")
    print("=" * 52)
    print(f"  {'Model':<28} | {'Spearman':>8}")
    print(f"  {'-'*28}-+-{'-'*8}")
    print(f"  {'LightGBM (zero-shot CLIP)':<28} | {BASELINE:>8.4f}")
    print(f"  {'LightGBM (linear probe)':<28} | {m['Spearman_r']:>8.4f}")
    print(f"\n  Delta: {'+' if delta>=0 else ''}{delta:.4f}  "
          f"({'probe wins' if delta > 0 else 'zero-shot wins'})")
    print("=" * 52)
    print(f"\n  Probe: accuracy={probe_metrics['accuracy']:.4f}  "
          f"F1={probe_metrics['f1']:.4f}  AUC={probe_metrics['roc_auc']:.4f}")
    print(f"  MLflow run ID: {run_id}")
    print("\n- Done. -\n")


if __name__ == "__main__":
    main()


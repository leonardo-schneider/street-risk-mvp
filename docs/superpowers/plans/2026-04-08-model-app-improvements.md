# Model & App Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CLIP linear probe (binary risk labels → learned visual feature) and two app improvements: animated risk map with click-to-hex panel, and a side-by-side hex comparison tab with SHAP explainability.

**Architecture:** Label 770 Street View images via a Streamlit labeling tab → train a logistic regression on frozen CLIP embeddings → replace zero-shot CLIP concept scores with a single `clip_risk_prob` feature → retrain LightGBM → add SHAP values to the API → wire Compare tab and animated map in the Streamlit frontend.

**Tech Stack:** scikit-learn (LogisticRegression), shap (TreeExplainer), boto3, Folium, Plotly, Streamlit tabs

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `.gitignore` | Modify | Ignore `data/labels/` and `model/clip_probe.pkl` |
| `app/requirements.txt` | Modify | Add `boto3` for S3 image fetch in labeling tab |
| `api/requirements.txt` | Modify | Add `shap` for SHAP value computation |
| `app/streamlit_app.py` | Modify | Add Label tab, animated map, Compare tab |
| `model/train_clip_probe.py` | Create | Extract embeddings + train logistic regression |
| `model/clip_probe.pkl` | Artifact | Saved probe (gitignored) |
| `pipeline/features/extract_clip_features.py` | Modify | Add `--use-probe` flag |
| `pipeline/gold/build_gold_table.py` | Modify | Add `--use-probe` flag to join probe hex file |
| `model/train.py` | Modify | Auto-detect `clip_risk_prob` column |
| `model/predict.py` | Modify | Add `build_feature_matrix()` + `compute_shap_values()` |
| `api/schemas.py` | Modify | Add `shap_values` field to `HexRiskResponse` |
| `api/main.py` | Modify | Compute SHAP at startup, expose in responses |
| `tests/test_clip_probe.py` | Create | Unit tests for probe training and inference |
| `tests/test_shap.py` | Create | Unit tests for SHAP computation |

---

## Task 1: Gitignore and requirements setup

**Files:**
- Modify: `.gitignore`
- Modify: `api/requirements.txt`
- Modify: `app/requirements.txt`

- [ ] **Step 1: Add entries to .gitignore**

Open `.gitignore` and add at the bottom:

```
# CLIP probe artifacts and image labels
data/labels/
model/clip_probe.pkl
```

- [ ] **Step 2: Add shap to api/requirements.txt**

Open `api/requirements.txt` and add:

```
shap
boto3
```

(boto3 is already used at runtime but not listed — add it now.)

- [ ] **Step 3: Add boto3 and matplotlib to app/requirements.txt**

Open `app/requirements.txt` and add:

```
boto3
matplotlib
```

- [ ] **Step 4: Create labels directory placeholder**

```bash
mkdir -p data/labels
touch data/labels/.gitkeep
```

- [ ] **Step 5: Commit**

```bash
git add .gitignore api/requirements.txt app/requirements.txt data/labels/.gitkeep
git commit -m "chore: gitignore probe artifacts, add shap+boto3 to requirements"
```

---

## Task 2: Labeling tab in Streamlit app

**Files:**
- Modify: `app/streamlit_app.py`

Add a "Label Images" tab that shows one Street View image at a time from S3 Bronze and writes `data/labels/image_labels.csv`. Gate it behind a local-only check so it is hidden on Streamlit Cloud.

- [ ] **Step 1: Add imports and constants at the top of streamlit_app.py**

After the existing imports block (after `from streamlit_folium import st_folium`), add:

```python
import csv
import io as _io
import boto3
```

After the existing constants block (after `STREETVIEW_URL = ...`), add:

```python
LABELS_CSV    = Path(__file__).parents[1] / "data" / "labels" / "image_labels.csv"
MANIFEST_PATH = Path(__file__).parents[1] / "data" / "bronze" / "image_manifest.csv"
S3_BUCKET     = os.getenv("S3_BUCKET_NAME", "street-risk-mvp")
IS_LOCAL      = os.getenv("RENDER", "").lower() != "true"
```

- [ ] **Step 2: Add S3 image fetch helper**

After the `geocode_address` function, add:

```python
@st.cache_data(ttl=3600)
def fetch_s3_image_bytes(s3_key: str) -> bytes:
    """Download image bytes from S3 Bronze. Cached per key."""
    s3 = boto3.client(
        "s3",
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    buf = _io.BytesIO()
    s3.download_fileobj(S3_BUCKET, s3_key, buf)
    return buf.getvalue()
```

- [ ] **Step 3: Add label loading/saving helpers**

After `fetch_s3_image_bytes`, add:

```python
def load_labels() -> dict:
    """Return {s3_key: label} from labels CSV. Returns empty dict if file missing."""
    if not LABELS_CSV.exists():
        return {}
    with open(LABELS_CSV, newline="") as f:
        return {row["s3_key"]: row["label"] for row in csv.DictReader(f)}


def save_label(s3_key: str, label: str):
    """Append or overwrite a single label in the CSV."""
    existing = load_labels()
    existing[s3_key] = label
    LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(LABELS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["s3_key", "label"])
        writer.writeheader()
        for key, lbl in existing.items():
            writer.writerow({"s3_key": key, "label": lbl})
```

- [ ] **Step 4: Add the labeling tab render function**

After `save_label`, add:

```python
def render_label_tab():
    """Labeling UI — local-only. Hidden on Streamlit Cloud (RENDER=true)."""
    st.subheader("Label Images")
    st.caption("Mark each Street View image as High Risk or Low Risk to train the CLIP probe.")

    if not MANIFEST_PATH.exists():
        st.warning("Manifest not found at data/bronze/image_manifest.csv — run the ingestion pipeline first.")
        return

    manifest = pd.read_csv(MANIFEST_PATH)
    manifest = manifest[manifest["status"] == "ok"].reset_index(drop=True)
    labels   = load_labels()

    labeled_keys   = set(labels.keys())
    unlabeled      = manifest[~manifest["s3_key"].isin(labeled_keys)].reset_index(drop=True)
    n_total        = len(manifest)
    n_labeled      = len(labeled_keys)

    st.progress(n_labeled / n_total, text=f"{n_labeled} / {n_total} images labeled")

    if unlabeled.empty:
        st.success("All images labeled! Run `python model/train_clip_probe.py` to train the probe.")
        return

    # Show the first unlabeled image
    row     = unlabeled.iloc[0]
    s3_key  = row["s3_key"]
    h3_idx  = row["h3_index"]

    st.markdown(f"**Image {n_labeled + 1} of {n_total}** — hex `{h3_idx}`")

    try:
        img_bytes = fetch_s3_image_bytes(s3_key)
        st.image(img_bytes, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load image from S3: {e}")
        if st.button("Skip this image"):
            save_label(s3_key, "skip")
            st.rerun()
        return

    col_hi, col_lo = st.columns(2)
    if col_hi.button("🔴 High Risk", use_container_width=True, type="primary"):
        save_label(s3_key, "high")
        fetch_s3_image_bytes.clear()
        st.rerun()
    if col_lo.button("🟢 Low Risk", use_container_width=True):
        save_label(s3_key, "low")
        fetch_s3_image_bytes.clear()
        st.rerun()
```

- [ ] **Step 5: Wire tabs into the main UI**

Find the existing main UI section (the `st.title(...)` and `st.divider()` block). Replace the content that currently renders `col_map, col_card = st.columns(...)` and everything after it with:

```python
# Tabs
if IS_LOCAL:
    tab_map, tab_compare, tab_label = st.tabs(["Risk Map", "Compare", "Label Images"])
else:
    tab_map, tab_compare = st.tabs(["Risk Map", "Compare"])

with tab_map:
    col_map, col_card = st.columns([6, 4])

    with col_map:
        st.subheader("Risk map — Sarasota, FL")
        m = build_map(
            geojson,
            scored_lat=st.session_state.score_lat,
            scored_lon=st.session_state.score_lon,
            scored_data=st.session_state.score_data,
        )
        st_folium(m, width=None, height=520, returned_objects=[])

    with col_card:
        st.subheader("Risk assessment")

        if not st.session_state.scored:
            st.info("Enter an address in the sidebar and click **Score this location**.")
        else:
            data = st.session_state.score_data
            tier = data.get("risk_tier", "Unknown")

            if tier == "Unknown":
                st.warning(data.get("message", "No data available for this location."))
                st.caption(f"H3 index: `{data.get('h3_index','')}`")
            else:
                st.markdown(risk_badge(tier), unsafe_allow_html=True)
                st.write("")
                m1, m2 = st.columns(2)
                m1.metric("Crashes per km²", f"{data.get('crash_density', 0):.1f}")
                pct = data.get("percentile", 0)
                m2.metric("Riskier than", f"{pct:.0f}% of zones")
                st.caption("Top contributing risk factors (CLIP scores)")
                top3   = data.get("top_risk_factors", [])
                detail = st.session_state.hex_detail or data
                if top3:
                    fig = top_factors_chart(top3, detail)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                lat = st.session_state.score_lat
                lon = st.session_state.score_lon
                if GOOGLE_API_KEY and lat and lon:
                    sv_url = STREETVIEW_URL.format(lat=lat, lon=lon, key=GOOGLE_API_KEY)
                    st.image(sv_url, caption="Street View at scored location", use_container_width=True)
                st.info(RISK_EXPLANATION.get(tier, ""))
                st.caption(f"H3 index: `{data.get('h3_index','')}`")

with tab_compare:
    st.info("Compare tab — coming in Task 8.")

if IS_LOCAL:
    with tab_label:
        render_label_tab()
```

- [ ] **Step 6: Run the app locally to verify the Label tab appears**

```bash
streamlit run app/streamlit_app.py
```

Expected: three tabs appear locally — "Risk Map", "Compare", "Label Images". The Label tab shows the first image with High Risk / Low Risk buttons.

- [ ] **Step 7: Commit**

```bash
git add app/streamlit_app.py
git commit -m "feat: add Label Images tab to Streamlit app for CLIP probe labeling"
```

---

## Task 3: CLIP probe training script

**Files:**
- Create: `model/train_clip_probe.py`
- Create: `tests/test_clip_probe.py`

Train a logistic regression on frozen CLIP embeddings extracted from labeled images. Save the probe to `model/clip_probe.pkl`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_clip_probe.py`:

```python
"""Tests for the CLIP linear probe training script."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from sklearn.linear_model import LogisticRegression


def test_train_probe_returns_fitted_model():
    """train_probe() returns a fitted LogisticRegression given embeddings + labels."""
    from model.train_clip_probe import train_probe

    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((20, 512)).astype(np.float32)
    labels = np.array([1, 0] * 10)

    probe = train_probe(embeddings, labels)

    assert isinstance(probe, LogisticRegression)
    assert hasattr(probe, "coef_"), "Probe must be fitted"
    preds = probe.predict(embeddings)
    assert preds.shape == (20,)
    assert set(preds).issubset({0, 1})


def test_train_probe_raises_on_single_class():
    """train_probe() raises ValueError when all labels are the same class."""
    from model.train_clip_probe import train_probe

    embeddings = np.ones((10, 512), dtype=np.float32)
    labels = np.ones(10, dtype=int)

    with pytest.raises(ValueError, match="at least two classes"):
        train_probe(embeddings, labels)


def test_aggregate_probe_scores_to_hex():
    """aggregate_to_hex() returns mean clip_risk_prob per h3_index."""
    from model.train_clip_probe import aggregate_to_hex
    import pandas as pd

    rows = pd.DataFrame({
        "h3_index": ["aaa", "aaa", "bbb"],
        "clip_risk_prob": [0.8, 0.6, 0.3],
    })

    result = aggregate_to_hex(rows)

    assert set(result.columns) == {"h3_index", "clip_risk_prob"}
    assert result.set_index("h3_index").loc["aaa", "clip_risk_prob"] == pytest.approx(0.7)
    assert result.set_index("h3_index").loc["bbb", "clip_risk_prob"] == pytest.approx(0.3)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd C:/Users/leona/street-risk-mvp
python -m pytest tests/test_clip_probe.py -v
```

Expected: `ImportError: cannot import name 'train_probe' from 'model.train_clip_probe'`

- [ ] **Step 3: Create model/train_clip_probe.py**

```python
"""
model/train_clip_probe.py

Train a logistic regression probe on frozen CLIP image embeddings.
Labels come from data/labels/image_labels.csv (generated by the Label tab).

Usage:
    python model/train_clip_probe.py
    python model/train_clip_probe.py --dry-run   # first 20 labeled images only
"""

import argparse
import io
import os
from pathlib import Path

import boto3
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# ── env ───────────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parents[1] / ".env", override=True)

BUCKET        = os.getenv("S3_BUCKET_NAME", "street-risk-mvp")
REGION        = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
MODEL_NAME    = "openai/clip-vit-base-patch32"

LABELS_CSV    = Path(__file__).parents[1] / "data" / "labels" / "image_labels.csv"
PROBE_PATH    = Path(__file__).parent / "clip_probe.pkl"


# ── public API (imported by tests) ───────────────────────────────────────────

def train_probe(embeddings: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    """
    Fit a logistic regression on CLIP embeddings.
    embeddings: (N, 512) float32
    labels:     (N,)    int  — 1=high risk, 0=low risk
    Raises ValueError if only one class is present.
    """
    unique = np.unique(labels)
    if len(unique) < 2:
        raise ValueError("Training requires at least two classes (high and low risk).")
    probe = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    probe.fit(embeddings, labels)
    return probe


def aggregate_to_hex(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-image clip_risk_prob to per-hex mean.
    df must have columns: h3_index, clip_risk_prob
    Returns DataFrame with columns: h3_index, clip_risk_prob
    """
    return (
        df.groupby("h3_index")["clip_risk_prob"]
          .mean()
          .reset_index()
    )


# ── helpers ───────────────────────────────────────────────────────────────────

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


def load_clip(device: str):
    print(f"  Loading {MODEL_NAME} on {device} ...")
    model     = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    return model, processor


def extract_embedding(image: Image.Image, model, processor, device: str) -> np.ndarray:
    """Extract a single 512-dim L2-normalised CLIP image embedding."""
    inputs = processor(images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    emb = F.normalize(emb, dim=-1)
    return emb.cpu().numpy()[0]   # (512,)


# ── main ──────────────────────────────────────────────────────────────────────

def main(dry_run: bool = False):
    print("\n- CLIP probe training -\n")

    # 1. Load labels
    print("Step 1/4  Loading labels ...")
    if not LABELS_CSV.exists():
        raise FileNotFoundError(
            f"Labels file not found: {LABELS_CSV}\n"
            "Run the Streamlit app locally and label images first."
        )
    labels_df = pd.read_csv(LABELS_CSV)
    labels_df = labels_df[labels_df["label"].isin(["high", "low"])].reset_index(drop=True)
    print(f"  [ok] {len(labels_df)} labeled images  (high={( labels_df['label']=='high').sum()}, low={(labels_df['label']=='low').sum()})")

    if dry_run:
        labels_df = labels_df.head(20)
        print(f"  (--dry-run) Limited to {len(labels_df)} images")

    y = (labels_df["label"] == "high").astype(int).values

    # 2. Load CLIP + S3
    print("\nStep 2/4  Loading CLIP model and S3 client ...")
    device = "cpu"
    model, processor = load_clip(device)
    s3 = make_s3()

    # 3. Extract embeddings for labeled images
    print(f"\nStep 3/4  Extracting embeddings for {len(labels_df)} labeled images ...")
    embeddings = []
    valid_idx  = []

    for i, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Embedding"):
        try:
            img = download_image(s3, row["s3_key"])
            emb = extract_embedding(img, model, processor, device)
            embeddings.append(emb)
            valid_idx.append(i)
        except Exception as e:
            tqdm.write(f"  [warn] Skipping {row['s3_key']}: {e}")

    embeddings = np.stack(embeddings)   # (N, 512)
    y_valid    = y[valid_idx]
    print(f"  [ok] {len(embeddings)} embeddings extracted")

    # 4. Train probe + 5-fold CV
    print("\nStep 4/4  Training logistic regression probe ...")
    probe  = train_probe(embeddings, y_valid)
    cv_acc = cross_val_score(probe, embeddings, y_valid, cv=min(5, len(embeddings) // 2), scoring="accuracy")
    print(f"  [ok] 5-fold CV accuracy: {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")

    # Save
    joblib.dump(probe, PROBE_PATH)
    print(f"  [ok] Probe saved: {PROBE_PATH}")

    print("\n- Done. Run extract_clip_features.py --use-probe to score all images. -\n")
    return probe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CLIP linear probe.")
    parser.add_argument("--dry-run", action="store_true", help="Use first 20 labeled images only.")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_clip_probe.py -v
```

Expected:
```
PASSED tests/test_clip_probe.py::test_train_probe_returns_fitted_model
PASSED tests/test_clip_probe.py::test_train_probe_raises_on_single_class
PASSED tests/test_clip_probe.py::test_aggregate_probe_scores_to_hex
```

- [ ] **Step 5: Commit**

```bash
git add model/train_clip_probe.py tests/test_clip_probe.py
git commit -m "feat: CLIP linear probe training script with unit tests"
```

---

## Task 4: extract_clip_features.py --use-probe flag

**Files:**
- Modify: `pipeline/features/extract_clip_features.py`

Add a `--use-probe` mode that loads the trained probe, extracts raw CLIP embeddings for all 770 images, applies the probe, aggregates per hex, and writes `sarasota_clip_probe_hex.parquet` to Silver.

- [ ] **Step 1: Add probe hex path constants**

In `extract_clip_features.py`, after the existing path constants block (after `HEX_FEAT_S3KEY = ...`), add:

```python
PROBE_PATH          = Path(__file__).parents[2] / "model" / "clip_probe.pkl"
PROBE_HEX_LOCAL     = Path(__file__).parents[2] / "data" / "silver" / "image_features" / "sarasota_clip_probe_hex.parquet"
PROBE_HEX_S3KEY     = "silver/image_features/sarasota_clip_probe_hex.parquet"
```

- [ ] **Step 2: Add extract_embedding helper**

In `extract_clip_features.py`, after the `score_batch` function, add:

```python
def extract_embeddings_batch(
    images: list,
    model,
    processor,
    device: str,
) -> np.ndarray:
    """
    Extract raw L2-normalised CLIP image embeddings.
    Returns (N, 512) float32 array.
    """
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    emb = F.normalize(emb, dim=-1)
    return emb.cpu().numpy()
```

- [ ] **Step 3: Add probe_mode() function**

In `extract_clip_features.py`, after `extract_embeddings_batch`, add:

```python
def probe_mode(dry_run: bool = False, batch_size: int = 32):
    """
    Run probe-based scoring: extract CLIP embeddings → apply trained probe
    → aggregate clip_risk_prob per hex → save Silver Parquet.
    Requires model/clip_probe.pkl to exist (run train_clip_probe.py first).
    """
    import joblib

    if not PROBE_PATH.exists():
        raise FileNotFoundError(
            f"Probe not found: {PROBE_PATH}\n"
            "Run `python model/train_clip_probe.py` first."
        )
    probe = joblib.load(PROBE_PATH)
    print(f"  [ok] Probe loaded: {PROBE_PATH}")

    # Load manifest
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest = manifest[manifest["status"] == "ok"].reset_index(drop=True)
    if dry_run:
        manifest = manifest.head(10)
        print(f"  (--dry-run) Limited to {len(manifest)} images")

    # Load CLIP + S3
    device = "cpu"
    model, processor = load_clip(device)
    s3 = make_s3_client()

    # Batch inference
    rows = []
    n = len(manifest)
    with tqdm(total=n, desc="Probe inference", unit="img") as pbar:
        for start in range(0, n, batch_size):
            batch_meta = manifest.iloc[start : start + batch_size]
            images, meta = [], []
            for _, row in batch_meta.iterrows():
                try:
                    img = download_image(s3, BUCKET, row["s3_key"])
                    images.append(img)
                    meta.append(row)
                except Exception as e:
                    tqdm.write(f"  [warn] {row['s3_key']}: {e}")

            if not images:
                pbar.update(len(batch_meta))
                continue

            embs  = extract_embeddings_batch(images, model, processor, device)  # (N, 512)
            probs = probe.predict_proba(embs)[:, 1]                              # P(high risk)

            for i, row in enumerate(meta):
                rows.append({"h3_index": row["h3_index"], "clip_risk_prob": float(probs[i])})

            pbar.update(len(images))

    img_probe_df = pd.DataFrame(rows)

    # Aggregate to hex
    hex_probe_df = (
        img_probe_df.groupby("h3_index")["clip_risk_prob"]
                    .mean()
                    .reset_index()
    )
    print(f"  [ok] {len(img_probe_df)} images scored → {len(hex_probe_df)} hexagons")

    # Save
    save_parquet(hex_probe_df, PROBE_HEX_LOCAL)
    if not dry_run:
        upload_parquet(PROBE_HEX_LOCAL, s3, BUCKET, PROBE_HEX_S3KEY)

    print(f"\n  clip_risk_prob stats:")
    p = hex_probe_df["clip_risk_prob"]
    print(f"    min={p.min():.4f}  mean={p.mean():.4f}  max={p.max():.4f}")
    print("\n- Probe hex features saved. Run build_gold_table.py --use-probe next. -\n")
    return hex_probe_df
```

- [ ] **Step 4: Update the argparse block**

Replace the existing `if __name__ == "__main__":` block with:

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract CLIP features from Street View images in S3."
    )
    parser.add_argument("--dry-run",    action="store_true", help="Process 10 images only.")
    parser.add_argument("--batch-size", type=int, default=32, help="CLIP inference batch size.")
    parser.add_argument("--use-probe",  action="store_true", help="Use trained probe instead of zero-shot scoring.")
    args = parser.parse_args()

    if args.use_probe:
        probe_mode(dry_run=args.dry_run, batch_size=args.batch_size)
    else:
        main(dry_run=args.dry_run, batch_size=args.batch_size)
```

- [ ] **Step 5: Verify dry-run works (requires AWS credentials)**

```bash
python pipeline/features/extract_clip_features.py --use-probe --dry-run
```

Expected output: `[ok] Probe loaded: ...clip_probe.pkl` then processes 10 images and prints `clip_risk_prob stats`.

- [ ] **Step 6: Commit**

```bash
git add pipeline/features/extract_clip_features.py
git commit -m "feat: add --use-probe flag to extract_clip_features.py"
```

---

## Task 5: build_gold_table.py --use-probe and train.py auto-detect

**Files:**
- Modify: `pipeline/gold/build_gold_table.py`
- Modify: `model/train.py`

Wire the probe hex file through the Gold layer and model training.

- [ ] **Step 1: Add probe constants to build_gold_table.py**

In `build_gold_table.py`, after the existing `CLIP_PATH = ...` line, add:

```python
PROBE_PATH = Path(__file__).parents[2] / "data" / "silver" / "image_features" / "sarasota_clip_probe_hex.parquet"
```

After `FINAL_COLS = [...]`, add:

```python
FINAL_COLS_PROBE = [
    "h3_index",
    "crash_density", "crash_count", "injury_rate",
    "clip_risk_prob",
    "road_type_primary", "speed_limit_mean", "lanes_mean",
    "dist_to_intersection_mean", "point_count",
    "risk_tier",
]
```

- [ ] **Step 2: Update build_gold_table.py main() with --use-probe flag**

Replace the `def main():` function signature and its first lines with:

```python
def main(use_probe: bool = False):
    print("\n- Building Gold training table -\n")
    if use_probe:
        print("  [probe mode] Using clip_risk_prob feature instead of 7 zero-shot CLIP cols.")

    # 1. Load
    print("Step 1/5  Loading Silver + Bronze inputs ...")
    crash = pd.read_parquet(CRASH_PATH)
    roads = pd.read_parquet(ROAD_PATH)

    if use_probe:
        if not PROBE_PATH.exists():
            raise FileNotFoundError(
                f"Probe hex file not found: {PROBE_PATH}\n"
                "Run `python pipeline/features/extract_clip_features.py --use-probe` first."
            )
        clip = pd.read_parquet(PROBE_PATH)
        final_cols = FINAL_COLS_PROBE
    else:
        clip = pd.read_parquet(CLIP_PATH)
        final_cols = FINAL_COLS

    print(f"  [ok] crash_hex     : {len(crash):,} hexagons")
    print(f"  [ok] clip_hex      : {len(clip):,} hexagons")
    print(f"  [ok] road_points   : {len(roads):,} points across {roads['h3_index'].nunique():,} hexagons")
```

Also update the `gold = gold[FINAL_COLS]` line near the end of `main()` to:

```python
    gold = gold[final_cols]
```

And update the argparse block at the bottom to:

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the Gold training table.")
    parser.add_argument("--use-probe", action="store_true", help="Use probe hex features instead of zero-shot CLIP.")
    args = parser.parse_args()
    main(use_probe=args.use_probe)
```

- [ ] **Step 3: Update model/train.py to auto-detect clip_risk_prob**

In `model/train.py`, find the `build_features` function. Replace it with:

```python
def build_features(train: pd.DataFrame, test: pd.DataFrame):
    road_dummies_train = pd.get_dummies(train["road_type_primary"], prefix="road", drop_first=True)
    road_dummies_test  = pd.get_dummies(test["road_type_primary"],  prefix="road", drop_first=True)
    road_dummies_test  = road_dummies_test.reindex(columns=road_dummies_train.columns, fill_value=0)

    # Auto-detect whether to use probe feature or zero-shot CLIP features
    if "clip_risk_prob" in train.columns:
        clip_feats = ["clip_risk_prob"]
        print("  [probe mode] Using clip_risk_prob feature.")
    else:
        clip_feats = CLIP_FEATURES
        print("  [zero-shot mode] Using 7 CLIP concept features.")

    base_cols    = clip_feats + NUMERIC_FEATURES
    dummy_cols   = list(road_dummies_train.columns)
    feature_cols = base_cols + dummy_cols

    X_train = pd.concat([train[base_cols].reset_index(drop=True),
                         road_dummies_train.reset_index(drop=True)], axis=1)
    X_test  = pd.concat([test[base_cols].reset_index(drop=True),
                         road_dummies_test.reset_index(drop=True)], axis=1)

    y_train = train[TARGET].values
    y_test  = test[TARGET].values

    return X_train, X_test, y_train, y_test, feature_cols
```

- [ ] **Step 4: Verify the pipeline runs end-to-end (dry check)**

```bash
python pipeline/gold/build_gold_table.py --use-probe
```

Expected: `FileNotFoundError: Probe hex file not found` (correct — probe hasn't been run yet). This confirms the flag is wired.

- [ ] **Step 5: Commit**

```bash
git add pipeline/gold/build_gold_table.py model/train.py
git commit -m "feat: --use-probe flag in build_gold_table.py, auto-detect clip_risk_prob in train.py"
```

---

## Task 6: SHAP values in model/predict.py and API

**Files:**
- Modify: `model/predict.py`
- Modify: `api/schemas.py`
- Modify: `api/main.py`
- Create: `tests/test_shap.py`

Compute SHAP values for all hexagons at API startup and expose them in the `/predict` and `/hex/{h3_index}` responses.

- [ ] **Step 1: Write the failing test**

Create `tests/test_shap.py`:

```python
"""Tests for SHAP computation in model/predict.py."""

import numpy as np
import pandas as pd
import pytest
import lightgbm as lgb


@pytest.fixture
def tiny_model_and_X():
    """Train a minimal LightGBM on synthetic data."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((50, 3)), columns=["f1", "f2", "f3"])
    y = X["f1"] * 2 + rng.standard_normal(50) * 0.1
    model = lgb.LGBMRegressor(n_estimators=10, verbose=-1)
    model.fit(X, y)
    return model, X


def test_build_feature_matrix_columns_match_feat_cols(tiny_model_and_X):
    """build_feature_matrix() returns a DataFrame whose columns match feat_cols."""
    from model.predict import build_feature_matrix

    _, X = tiny_model_and_X
    # Simulate a gold table with road_type_primary
    df = X.copy()
    df["h3_index"] = [f"hex_{i}" for i in range(len(df))]
    df["road_type_primary"] = "residential"
    feat_cols = ["f1", "f2", "f3"]

    result = build_feature_matrix(df, feat_cols)

    assert list(result.columns) == feat_cols
    assert len(result) == len(df)


def test_compute_shap_values_shape(tiny_model_and_X):
    """compute_shap_values() returns a DataFrame with same shape as X."""
    from model.predict import compute_shap_values

    model, X = tiny_model_and_X
    shap_df = compute_shap_values(model, X)

    assert shap_df.shape == X.shape
    assert list(shap_df.columns) == list(X.columns)


def test_compute_shap_values_index_preserved(tiny_model_and_X):
    """compute_shap_values() preserves the index of X."""
    from model.predict import compute_shap_values

    model, X = tiny_model_and_X
    X.index = [f"hex_{i}" for i in range(len(X))]
    shap_df = compute_shap_values(model, X)

    assert list(shap_df.index) == list(X.index)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_shap.py -v
```

Expected: `ImportError: cannot import name 'build_feature_matrix' from 'model.predict'`

- [ ] **Step 3: Implement model/predict.py**

Replace the entire file with:

```python
"""
model/predict.py — feature matrix reconstruction and SHAP utilities.

Used by api/main.py at startup to compute SHAP values for all hexagons.
"""

import numpy as np
import pandas as pd
import shap


def build_feature_matrix(df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    """
    Reconstruct the model feature matrix from a Gold table DataFrame.

    feat_cols lists every column the model was trained on (from feature_columns.json).
    Columns starting with "road_" are dummy-encoded from road_type_primary.
    All other columns are taken directly from df.

    Returns a DataFrame with columns == feat_cols, index == df.index.
    """
    base_cols  = [c for c in feat_cols if not c.startswith("road_")]
    dummy_cols = [c for c in feat_cols if c.startswith("road_")]

    X_base = df[base_cols].reset_index(drop=True)

    if dummy_cols:
        road_dummies = pd.get_dummies(df["road_type_primary"], prefix="road", drop_first=True)
        road_dummies = road_dummies.reindex(columns=dummy_cols, fill_value=0).reset_index(drop=True)
        X = pd.concat([X_base, road_dummies], axis=1)
    else:
        X = X_base

    X.index = df.index
    return X[feat_cols]


def compute_shap_values(model, X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute SHAP values for every row in X using TreeExplainer.

    Returns a DataFrame with same shape, columns, and index as X.
    Each value is the SHAP contribution of that feature for that row.
    """
    explainer = shap.TreeExplainer(model)
    values    = explainer.shap_values(X)
    return pd.DataFrame(values, columns=X.columns, index=X.index)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_shap.py -v
```

Expected:
```
PASSED tests/test_shap.py::test_build_feature_matrix_columns_match_feat_cols
PASSED tests/test_shap.py::test_compute_shap_values_shape
PASSED tests/test_shap.py::test_compute_shap_values_index_preserved
```

- [ ] **Step 5: Add shap_values field to HexRiskResponse in api/schemas.py**

In `api/schemas.py`, replace the `HexRiskResponse` class with:

```python
class HexRiskResponse(BaseModel):
    h3_index: str
    crash_density: float
    risk_tier: str
    risk_score_normalized: float = Field(..., description="Min-max scaled crash density, 0-1")
    top_risk_factors: List[str] = Field(..., description="Top 3 CLIP feature names sorted by score")
    clip_scores: Dict[str, float] = Field(..., description="Raw CLIP softmax scores for all 7 risk concepts")
    hex_center: HexCenter
    percentile: float = Field(..., description="Percentile of this hex vs all hexagons (0-100)")
    shap_values: Dict[str, float] = Field(default_factory=dict, description="SHAP value per feature for this hexagon")
```

- [ ] **Step 6: Update api/main.py — compute SHAP at startup**

In `api/main.py`, add this import at the top (after `from api.schemas import ...`):

```python
from model.predict import build_feature_matrix, compute_shap_values
```

In the `startup()` function, after the `_state["percentile"]` line and before the `_state["df"] = df.set_index(...)` line, add:

```python
    # Compute SHAP values for all hexagons
    try:
        X_all   = build_feature_matrix(df.reset_index(drop=True), feat_cols)
        shap_df = compute_shap_values(model, X_all)
        shap_df.index = df["h3_index"].values if "h3_index" in df.columns else df.index
        _state["shap_df"] = shap_df
        print(f"[startup] SHAP values computed for {len(shap_df)} hexagons")
    except Exception as e:
        print(f"[startup] SHAP computation failed (non-fatal): {e}")
        _state["shap_df"] = None
```

Note: the `df` at that point in `startup()` still has `h3_index` as a column (before `set_index`). Adjust the SHAP index line accordingly — use `df["h3_index"].values`.

- [ ] **Step 7: Update _row_to_hex_response() to include shap_values**

Replace `_row_to_hex_response` with:

```python
def _row_to_hex_response(h3_index: str, row: pd.Series) -> HexRiskResponse:
    """Convert a Gold table row into a HexRiskResponse."""
    clip_scores      = {col: round(float(row[col]), 6) for col in CLIP_COLS if col in row.index}
    top_risk_factors = sorted(clip_scores, key=clip_scores.get, reverse=True)[:3]
    lat, lon         = h3.cell_to_latlng(h3_index)

    shap_values = {}
    shap_df = _state.get("shap_df")
    if shap_df is not None and h3_index in shap_df.index:
        shap_values = {col: round(float(v), 6) for col, v in shap_df.loc[h3_index].items()}

    return HexRiskResponse(
        h3_index=h3_index,
        crash_density=round(float(row["crash_density"]), 4),
        risk_tier=str(row["risk_tier"]),
        risk_score_normalized=round(float(row["risk_score_normalized"]), 6),
        top_risk_factors=top_risk_factors,
        clip_scores=clip_scores,
        hex_center=HexCenter(lat=round(lat, 6), lon=round(lon, 6)),
        percentile=round(float(row["percentile"]), 2),
        shap_values=shap_values,
    )
```

- [ ] **Step 8: Verify API starts and /predict returns shap_values**

```bash
uvicorn api.main:app --reload --port 8000
```

Then in another terminal:
```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"lat": 27.3364, "lon": -82.5307}' | python -m json.tool | grep shap
```

Expected: `"shap_values": { "speed_limit_mean": ..., "lanes_mean": ..., ... }`

- [ ] **Step 9: Commit**

```bash
git add model/predict.py api/schemas.py api/main.py tests/test_shap.py
git commit -m "feat: SHAP values in model/predict.py and API /predict response"
```

---

## Task 7: Animated risk map

**Files:**
- Modify: `app/streamlit_app.py`

Replace the 3-tier categorical coloring with a continuous green→red gradient and add click-to-panel interactivity.

- [ ] **Step 1: Add matplotlib.colors import**

In `app/streamlit_app.py`, add to the imports block:

```python
import matplotlib.colors as mcolors
```

- [ ] **Step 2: Add risk_color helper**

After the `risk_badge` function, add:

```python
def risk_color(score: float) -> str:
    """Map a 0-1 risk score to a hex color on a green→yellow→red gradient."""
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "risk", ["#1a9850", "#fee090", "#d73027"]
    )
    return mcolors.to_hex(cmap(max(0.0, min(1.0, float(score)))))
```

- [ ] **Step 3: Replace build_map() with an animated version**

Replace the entire `build_map` function with:

```python
def build_map(geojson: dict, scored_lat=None, scored_lon=None, scored_data=None) -> folium.Map:
    m = folium.Map(location=SARASOTA_CENTER, zoom_start=13, tiles="CartoDB positron")

    if geojson:
        for feature in geojson.get("features", []):
            props   = feature["properties"]
            score   = props.get("risk_score_normalized", 0.0)
            tier    = props.get("risk_tier", "Low")
            density = props.get("crash_density", 0)
            h3idx   = props.get("h3_index", "")
            color   = risk_color(score)

            folium.GeoJson(
                feature,
                style_function=lambda f, c=color: {
                    "fillColor":   c,
                    "color":       "#222222",
                    "weight":      0.4,
                    "fillOpacity": 0.65,
                },
                highlight_function=lambda f: {
                    "fillOpacity": 0.9,
                    "weight":      2,
                    "color":       "#ffffff",
                },
                tooltip=folium.Tooltip(
                    f"<b>{tier} risk</b> &nbsp;|&nbsp; {score:.2f}<br>"
                    f"Density: {density:.1f} crashes/km²<br>"
                    f"<span style='font-size:11px;color:#888'>{h3idx}</span>",
                    sticky=False,
                ),
            ).add_to(m)

    # Gradient legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
                padding:10px 14px;border-radius:8px;border:1px solid #ccc;font-size:12px;">
        <b>Risk score</b><br>
        <div style="background:linear-gradient(to right,#1a9850,#fee090,#d73027);
                    width:120px;height:12px;border-radius:3px;margin:4px 0;"></div>
        <div style="display:flex;justify-content:space-between;width:120px;">
            <span>Low</span><span>High</span>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    if scored_lat is not None and scored_data and "risk_tier" in scored_data:
        tier    = scored_data.get("risk_tier", "Unknown")
        density = scored_data.get("crash_density", 0)
        pct     = scored_data.get("percentile", 0)
        popup_html = (
            f"<b>{TIER_EMOJI.get(tier, '')} {tier} risk</b><br>"
            f"Density: {density:.1f} crashes/km²<br>"
            f"Percentile: {pct:.0f}th"
        )
        folium.Marker(
            location=[scored_lat, scored_lon],
            popup=folium.Popup(popup_html, max_width=200),
            icon=folium.Icon(
                color="red" if tier == "High" else "orange" if tier == "Medium" else "green",
                icon="info-sign",
            ),
        ).add_to(m)

    return m
```

- [ ] **Step 4: Run app and verify gradient coloring**

```bash
streamlit run app/streamlit_app.py
```

Expected: map shows smooth green→yellow→red gradient across hexagons instead of 3-tier categorical colors. Hovering a hex highlights its border in white.

- [ ] **Step 5: Commit**

```bash
git add app/streamlit_app.py
git commit -m "feat: animated risk map with continuous green-red gradient and highlight on hover"
```

---

## Task 8: Compare tab

**Files:**
- Modify: `app/streamlit_app.py`

Implement the Compare tab with two address inputs, side-by-side SHAP bar charts, Street View thumbnails, and a verdict banner.

- [ ] **Step 1: Add SHAP label map constant**

In `app/streamlit_app.py`, after the `CLIP_LABEL` dict, add:

```python
FEATURE_LABEL = {
    **CLIP_LABEL,
    "speed_limit_mean":           "Speed limit (mean)",
    "lanes_mean":                 "Lane count (mean)",
    "dist_to_intersection_mean":  "Distance to intersection",
    "point_count":                "Road point density",
    "clip_risk_prob":             "Visual risk (probe)",
}
```

- [ ] **Step 2: Add shap_chart helper**

After the `top_factors_chart` function, add:

```python
def shap_chart(shap_values: dict, tier: str) -> go.Figure:
    """Horizontal bar chart of SHAP values sorted by absolute magnitude."""
    if not shap_values:
        return go.Figure()

    items  = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
    labels = [FEATURE_LABEL.get(k, k) for k, _ in items]
    values = [v for _, v in items]
    colors = ["#d73027" if v > 0 else "#1a9850" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=0, r=50, t=10, b=10),
        xaxis=dict(title="SHAP contribution", zeroline=True, zerolinecolor="#cccccc"),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig
```

- [ ] **Step 3: Add render_compare_tab() function**

After `render_label_tab`, add:

```python
def render_compare_tab():
    st.subheader("Compare two locations")
    st.caption("Score two addresses side-by-side and see what drives the risk difference.")

    col_a, col_b = st.columns(2)
    with col_a:
        addr_a = st.text_input("Location A", value="1 N Tamiami Trail, Sarasota, FL", key="cmp_a")
    with col_b:
        addr_b = st.text_input("Location B", value="2100 Ringling Blvd, Sarasota, FL", key="cmp_b")

    compare_btn = st.button("Compare", type="primary")

    if not compare_btn:
        st.info("Enter two addresses above and click **Compare**.")
        return

    # Geocode both addresses
    coords_a = geocode_address(addr_a.strip()) if addr_a.strip() else None
    coords_b = geocode_address(addr_b.strip()) if addr_b.strip() else None

    if coords_a is None:
        st.error(f"Could not geocode: '{addr_a}'")
        return
    if coords_b is None:
        st.error(f"Could not geocode: '{addr_b}'")
        return

    # Score both
    data_a = score_location(*coords_a)
    data_b = score_location(*coords_b)

    if "error" in data_a:
        st.error(f"API error for Location A: {data_a['error']}")
        return
    if "error" in data_b:
        st.error(f"API error for Location B: {data_b['error']}")
        return

    tier_a = data_a.get("risk_tier", "Unknown")
    tier_b = data_b.get("risk_tier", "Unknown")

    # Verdict banner
    score_a = data_a.get("risk_score_normalized", 0)
    score_b = data_b.get("risk_score_normalized", 0)
    if tier_a != "Unknown" and tier_b != "Unknown":
        if abs(score_a - score_b) < 0.05:
            verdict = "Both locations have similar risk scores."
        elif score_a > score_b:
            pct_diff = round((score_a - score_b) / max(score_b, 0.001) * 100)
            verdict  = f"Location A is **{pct_diff}% higher risk** than Location B."
        else:
            pct_diff = round((score_b - score_a) / max(score_a, 0.001) * 100)
            verdict  = f"Location B is **{pct_diff}% higher risk** than Location A."
        st.info(verdict)

    st.divider()

    col_left, col_right = st.columns(2)

    for col, data, addr, coords, label in [
        (col_left,  data_a, addr_a, coords_a, "A"),
        (col_right, data_b, addr_b, coords_b, "B"),
    ]:
        tier = data.get("risk_tier", "Unknown")
        with col:
            st.markdown(f"**Location {label}** — {addr}")
            st.markdown(risk_badge(tier), unsafe_allow_html=True)
            st.write("")

            if tier == "Unknown":
                st.warning(data.get("message", "Outside coverage area."))
                continue

            m1, m2 = st.columns(2)
            m1.metric("Crashes per km²", f"{data.get('crash_density', 0):.1f}")
            m2.metric("Riskier than", f"{data.get('percentile', 0):.0f}% of zones")

            shap_values = data.get("shap_values", {})
            if shap_values:
                st.caption("Feature contributions (SHAP)")
                fig = shap_chart(shap_values, tier)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.caption("SHAP values not available (API may be loading).")

            lat, lon = coords
            if GOOGLE_API_KEY:
                sv_url = STREETVIEW_URL.format(lat=lat, lon=lon, key=GOOGLE_API_KEY)
                st.image(sv_url, caption="Street View", use_container_width=True)

            st.caption(f"H3 index: `{data.get('h3_index', '')}`")
```

- [ ] **Step 4: Wire render_compare_tab() into the Compare tab**

Find the placeholder in `tab_compare`:

```python
with tab_compare:
    st.info("Compare tab — coming in Task 8.")
```

Replace it with:

```python
with tab_compare:
    render_compare_tab()
```

- [ ] **Step 5: Run app and verify the Compare tab**

```bash
streamlit run app/streamlit_app.py
```

Expected:
- "Compare" tab shows two address inputs and a "Compare" button
- After clicking Compare, two columns appear with risk badges, SHAP charts, and Street View images
- Verdict banner shows which location is higher risk

- [ ] **Step 6: Commit**

```bash
git add app/streamlit_app.py
git commit -m "feat: Compare tab with SHAP bar charts, Street View thumbnails, and verdict banner"
```

---

## Full pipeline run order (after all tasks complete)

When you have enough labels (recommended: ≥100, ideally all 770):

```bash
# 1. Train the probe
python model/train_clip_probe.py

# 2. Score all 770 images with the probe → Silver
python pipeline/features/extract_clip_features.py --use-probe

# 3. Rebuild Gold table with probe feature
python pipeline/gold/build_gold_table.py --use-probe

# 4. Retrain LightGBM (auto-detects clip_risk_prob)
python model/train.py

# 5. Start API and Streamlit
uvicorn api.main:app --reload --port 8000
streamlit run app/streamlit_app.py
```

# Design: Street Risk MVP — Model & App Improvements

**Date:** 2026-04-08
**Goal:** Portfolio-focused improvements — stronger model signal via CLIP fine-tuning, and more visually impressive app features.

---

## 1. CLIP Linear Probe (Model Improvement)

### Problem
The current model uses zero-shot CLIP scoring with 7 hand-written risk concept prompts. Each hexagon averages ~1.3 images, making the zero-shot scores noisy. A task-specific linear probe trained on labeled Street View images will produce a stronger, more focused risk signal.

### Labeling Workflow
- New "Label Images" tab added to the Streamlit app
- Fetches each of the 770 Street View images from S3 Bronze one at a time
- User clicks **High Risk** or **Low Risk** per image
- Labels saved to `data/labels/image_labels.csv` (gitignored, persists across sessions)
- Progress indicator shows how many images remain to label

### Linear Probe Training
New script: `model/train_clip_probe.py`

Steps:
1. Load labeled images from S3, extract frozen CLIP image embeddings (512-dim, `openai/clip-vit-base-patch32`)
2. Train a logistic regression (sklearn) on binary labels (high=1, low=0)
3. Save probe to `model/clip_probe.pkl`
4. Run inference on all 770 images → aggregate per-hex mean predicted probability → single feature `clip_risk_prob`

### Pipeline Integration
- `pipeline/features/extract_clip_features.py` gains a `--use-probe` flag
  - Default (no flag): existing zero-shot scoring, 7 concept features
  - With flag: probe-based scoring, single `clip_risk_prob` feature replaces 7 zero-shot scores
- Gold table rebuild and LightGBM retrain follow the existing pipeline unchanged

### Success Metric
Spearman rank correlation on the same geographic KMeans test split (cluster 0, 55 hexagons). Target: beat current baseline of **0.666**.

---

## 2. Animated Risk Map (App — Visualization)

### Problem
The current Folium map is static — no interactivity beyond hover tooltips.

### Design
Replace or augment the existing Folium map with a more interactive experience:
- **Smooth color gradient** on hex risk scores (green → yellow → red)
- **Click a hex** → map zooms in, side panel slides open
- Side panel shows:
  - Risk score badge (color-coded)
  - Street View thumbnail (fetched from existing API response)
  - Top 3 risk factors (feature contributions)
  - Crash stats (density, count)

### Implementation Notes
- Folium supports click callbacks via `GeoJson` with `on_each_feature` JS
- Alternatively, swap to `pydeck` or `plotly` choropleth for smoother animations
- Street View thumbnail already available via the API's `/hex/{hex_id}` endpoint

---

## 3. Side-by-Side Hex Comparison (App — User Tool)

### Problem
No way to compare two locations directly — the key use case for insurance pricing demos.

### Design
New **"Compare"** tab in the Streamlit app:
- Two address input fields (or click two hexes on the map)
- Calls the existing `/predict` API endpoint for each address
- Two-column layout per hex:
  - Risk score badge (color-coded, large)
  - Bar chart of SHAP feature contributions (LightGBM `shap` values)
  - Street View thumbnail
  - Crash stats (density, count, hex area)
- Verdict banner: "Hex A is X% higher risk than Hex B"

### Implementation Notes
- Requires adding SHAP value computation to `model/predict.py` and exposing via API (new field in `/predict` response)
- `shap` library already compatible with LightGBM
- SHAP values computed at inference time (fast for a single prediction)

---

## Out of Scope
- Multi-city expansion (Sarasota MVP only per CLAUDE.md)
- CLIP weight fine-tuning (LoRA or full fine-tune) — linear probe chosen for simplicity
- Active learning loop — overkill for 770 images
- PDF export of risk report

---

## Files Affected

| File | Change |
|------|--------|
| `app/streamlit_app.py` | Add Label tab, Compare tab, animated map |
| `model/train_clip_probe.py` | New — linear probe training script |
| `model/clip_probe.pkl` | New — saved probe artifact |
| `model/predict.py` | Add SHAP value computation |
| `pipeline/features/extract_clip_features.py` | Add `--use-probe` flag |
| `api/main.py` | Add `shap_values` field to `/predict` response |
| `api/schemas.py` | Update Pydantic schema for SHAP values |
| `data/labels/image_labels.csv` | New — gitignored label file |

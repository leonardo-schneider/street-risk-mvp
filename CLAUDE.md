# CLAUDE.md — Street-Level Photo → Auto Insurance Zone Risk

## What this project is
A distributed ML pipeline that predicts micro-zone road risk scores
from street-level imagery and geospatial features, served as a live web app.

## City scope
Sarasota, FL — single city MVP only. Do not expand scope.

## Architecture — Medallion Layers (all stored in S3)
- Bronze: raw Street View images, raw crash CSV, raw OSM road geometry
- Silver: cleaned road points (Parquet), CLIP image features (Parquet), crash aggregated to H3 hex (Parquet)
- Gold: final ML training table (Parquet), scored hexagons (Parquet)

## Geographic unit
H3 hexagons, resolution 9 (~0.1 km² per cell)

## Prediction target
crash_density = crash_count_in_hex / hex_area_km2  (regression, continuous)

## Tech stack
- Storage: AWS S3 (boto3)
- Road sampling: OSMnx + OpenStreetMap
- Images: Google Street View Static API (cached in S3 Bronze after first fetch)
- Crash data: FDOT Open Data Hub (Sarasota County CSV)
- Image features: CLIP zero-shot (openai/clip-vit-base-patch32 via HuggingFace)
- Tabular features: road type, speed limit, lane count, distance to intersection
- Model: LightGBM regressor
- Tracking: MLflow (local SQLite backend, artifacts → S3)
- API: FastAPI (deployed on Render)
- Frontend: Streamlit (deployed on Streamlit Cloud)
- Visualization: Folium map showing H3 hexagons colored by risk score

## Distributed/cloud stages (rubric requirement)
1. Image ingestion pipeline writes raw images to S3 Bronze
2. CLIP feature extraction batch job reads Bronze, writes Silver Parquet to S3

## Folder structure
pipeline/ingestion/   - road sampling, image fetching, crash data download
pipeline/features/    - CLIP feature extraction, OSM tabular features
pipeline/gold/        - assemble final training table
model/                - train.py, evaluate.py, predict.py
api/                  - FastAPI app
app/                  - Streamlit frontend
infrastructure/       - S3 setup, AWS config
tests/                - unit tests and e2e test script
data/                 - local bronze/silver/gold mirrors (gitignored)

## Environment variables (never hardcode these)
GOOGLE_MAPS_API_KEY   - Street View Static API
AWS_ACCESS_KEY_ID     - AWS credentials
AWS_SECRET_ACCESS_KEY - AWS credentials
AWS_DEFAULT_REGION    - us-east-1
S3_BUCKET_NAME        - street-risk-mvp

## Key constraints
- Never re-fetch Street View images if they exist in S3 (always check cache first)
- Always use H3 res-9 for all spatial joins
- No end-to-end CNN training — use CLIP embeddings only
- One city only — do not generalize to multi-city in MVP
- Keep data/bronze, data/silver, data/gold gitignored (large files)

## CLIP risk concepts (canonical list — do not change without discussion)
RISK_CONCEPTS = [
    "a road with heavy traffic",
    "a road with poor lighting",
    "a road with no sidewalks or pedestrian infrastructure",
    "a wet or damaged road surface",
    "a clear, well-maintained suburban road",
    "an intersection with no traffic signals",
    "a road with parked cars blocking visibility",
]

## Validation strategy
- Spatial train/test split by H3 hex cluster (no geographic leakage)
- Report RMSE, MAE, Spearman rank correlation
- Never split randomly — neighboring hexagons share visual features

## Current phase
Phase 2 complete — repo structure and infrastructure initialized.
Next: Phase 3 — road sampling pipeline (pipeline/ingestion/sample_roads.py)
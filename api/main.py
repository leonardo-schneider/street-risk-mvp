"""
api/main.py — Street Risk MVP FastAPI application.

Serves precomputed H3 hexagon risk scores for Sarasota, FL.
Loads the Gold table and trained LightGBM model on startup.

Run with:
    uvicorn api.main:app --reload --port 8000
"""

import io
import json
import os
from pathlib import Path
from typing import Union

import boto3
import h3
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    GeoJSONFeature,
    GeoJSONFeatureCollection,
    GeoJSONGeometry,
    HealthResponse,
    HexCenter,
    HexRiskResponse,
    HexUnknownResponse,
    PredictRequest,
    StatsResponse,
    TopHex,
)

# ── env ───────────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parents[1] / ".env", override=True)

REGION      = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "street-risk-mvp")

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).parents[1]
GOLD_PATH      = ROOT / "data" / "gold" / "training_table" / "sarasota_gold.parquet"
MODEL_PATH     = ROOT / "model" / "risk_model.pkl"
FEAT_COLS_PATH = ROOT / "model" / "feature_columns.json"

# S3 keys for production (RENDER=true)
S3_GOLD_KEY   = "gold/training_table/sarasota_gold.parquet"
S3_MODEL_KEY  = "gold/risk_model.pkl"
S3_FEAT_KEY   = "gold/feature_columns.json"

CLIP_COLS = [
    "clip_heavy_traffic", "clip_poor_lighting", "clip_no_sidewalks",
    "clip_damaged_road",  "clip_clear_road",    "clip_no_signals",
    "clip_parked_cars",
]

# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Street Risk API",
    description="Micro-zone road risk scores from street-level imagery — Sarasota, FL",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── startup state ─────────────────────────────────────────────────────────────
_state: dict = {}


def _s3_client():
    return boto3.client(
        "s3",
        region_name=REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


def _load_from_s3():
    """Load Gold table, model, and feature columns directly from S3 (production)."""
    s3 = _s3_client()

    print(f"[startup] Loading from S3: s3://{BUCKET_NAME}/")

    buf = io.BytesIO()
    s3.download_fileobj(BUCKET_NAME,S3_GOLD_KEY, buf)
    buf.seek(0)
    df = pd.read_parquet(buf)
    print(f"[startup] Gold table loaded from S3: {len(df)} hexagons")

    buf = io.BytesIO()
    s3.download_fileobj(BUCKET_NAME,S3_MODEL_KEY, buf)
    buf.seek(0)
    model = joblib.load(buf)
    print(f"[startup] Model loaded from S3: {S3_MODEL_KEY}")

    buf = io.BytesIO()
    s3.download_fileobj(BUCKET_NAME,S3_FEAT_KEY, buf)
    feat_cols = json.loads(buf.getvalue().decode("utf-8"))
    print(f"[startup] Feature columns loaded from S3: {len(feat_cols)} features")

    return df, model, feat_cols


def _load_from_local():
    """Load Gold table, model, and feature columns from local disk (development)."""
    df = pd.read_parquet(GOLD_PATH)
    print(f"[startup] Gold table loaded from local: {len(df)} hexagons")
    model = joblib.load(MODEL_PATH)
    print(f"[startup] Model loaded from local: {MODEL_PATH.name}")
    feat_cols = json.loads(FEAT_COLS_PATH.read_text())
    return df, model, feat_cols


@app.on_event("startup")
def startup():
    """
    Load Gold table, model, and feature columns into memory on startup.
    Uses S3 when RENDER=true (production), local disk otherwise (development).
    """
    is_render = os.getenv("RENDER", "").lower() == "true"

    if is_render:
        df, model, feat_cols = _load_from_s3()
    else:
        df, model, feat_cols = _load_from_local()

    # Precompute min-max normalised risk score (0-1)
    mn, mx = df["crash_density"].min(), df["crash_density"].max()
    df["risk_score_normalized"] = (df["crash_density"] - mn) / (mx - mn) if mx > mn else 0.0

    # Precompute percentile rank for each hex
    df["percentile"] = df["crash_density"].rank(pct=True) * 100

    # Index by h3_index for O(1) lookups
    _state["df"]         = df.set_index("h3_index")
    _state["model"]      = model
    _state["feat_cols"]  = feat_cols
    _state["model_loaded"] = True

    print(f"[startup] Ready. {'S3' if is_render else 'Local'} mode.")


# ── helpers ───────────────────────────────────────────────────────────────────

def _row_to_hex_response(h3_index: str, row: pd.Series) -> HexRiskResponse:
    """Convert a Gold table row into a HexRiskResponse."""
    clip_scores = {col: round(float(row[col]), 6) for col in CLIP_COLS}
    top_risk_factors = sorted(clip_scores, key=clip_scores.get, reverse=True)[:3]
    lat, lon = h3.cell_to_latlng(h3_index)
    return HexRiskResponse(
        h3_index=h3_index,
        crash_density=round(float(row["crash_density"]), 4),
        risk_tier=str(row["risk_tier"]),
        risk_score_normalized=round(float(row["risk_score_normalized"]), 6),
        top_risk_factors=top_risk_factors,
        clip_scores=clip_scores,
        hex_center=HexCenter(lat=round(lat, 6), lon=round(lon, 6)),
        percentile=round(float(row["percentile"]), 2),
    )


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health():
    """
    Health check endpoint.
    Returns API status, model load status, and city scope.
    """
    return HealthResponse(
        status="ok",
        model="loaded" if _state.get("model_loaded") else "not loaded",
        city="Sarasota, FL",
    )


@app.get(
    "/hex/{h3_index}",
    response_model=HexRiskResponse,
    tags=["Risk Scores"],
)
def get_hex(h3_index: str):
    """
    Return the precomputed risk score for a specific H3 res-9 hexagon.

    Includes crash density, risk tier, normalised score (0-1), top 3 CLIP
    risk factors, hexagon centroid coordinates, and crash density percentile
    relative to all hexagons in the Gold table.
    """
    df: pd.DataFrame = _state["df"]
    if h3_index not in df.index:
        raise HTTPException(status_code=404, detail=f"Hexagon {h3_index!r} not found in Gold table.")
    return _row_to_hex_response(h3_index, df.loc[h3_index])


@app.post(
    "/predict",
    response_model=Union[HexRiskResponse, HexUnknownResponse],
    tags=["Risk Scores"],
)
def predict(body: PredictRequest):
    """
    Convert a lat/lon coordinate to an H3 res-9 hexagon and return its risk score.

    If the hexagon is present in the Gold table the full risk response is returned.
    If it falls outside the covered area an 'Unknown' response is returned with a
    message indicating no data is available.
    """
    h3_index = h3.latlng_to_cell(body.lat, body.lon, 9)
    df: pd.DataFrame = _state["df"]

    if h3_index not in df.index:
        return HexUnknownResponse(
            h3_index=h3_index,
            risk_tier="Unknown",
            message="No data available for this location",
        )

    return _row_to_hex_response(h3_index, df.loc[h3_index])


@app.get(
    "/map-data",
    response_model=GeoJSONFeatureCollection,
    tags=["Map"],
)
def map_data():
    """
    Return all hexagons as a GeoJSON FeatureCollection.

    Each Feature contains the H3 hexagon boundary polygon as geometry and
    h3_index, crash_density, risk_tier, and risk_score_normalized as properties.
    Intended for direct consumption by Folium / Leaflet map layers.
    """
    df: pd.DataFrame = _state["df"].reset_index()
    features = []

    for _, row in df.iterrows():
        # h3.cell_to_boundary returns list of (lat, lon) tuples — GeoJSON needs [lon, lat]
        boundary = h3.cell_to_boundary(row["h3_index"])
        coords   = [[lon, lat] for lat, lon in boundary]
        coords.append(coords[0])  # close the ring

        features.append(GeoJSONFeature(
            geometry=GeoJSONGeometry(type="Polygon", coordinates=[coords]),
            properties={
                "h3_index":             row["h3_index"],
                "crash_density":        round(float(row["crash_density"]), 4),
                "risk_tier":            row["risk_tier"],
                "risk_score_normalized": round(float(row["risk_score_normalized"]), 6),
            },
        ))

    return GeoJSONFeatureCollection(features=features)


@app.get("/stats", response_model=StatsResponse, tags=["Meta"])
def stats():
    """
    Return summary statistics for the entire Gold table.

    Includes total hexagon count, mean crash density, risk tier distribution,
    and the five highest-risk hexagons by crash density.
    """
    df: pd.DataFrame = _state["df"].reset_index()

    top5 = df.nlargest(5, "crash_density")
    top5_list = [
        TopHex(
            h3_index=row["h3_index"],
            crash_density=round(float(row["crash_density"]), 4),
            risk_tier=row["risk_tier"],
            risk_score_normalized=round(float(row["risk_score_normalized"]), 6),
        )
        for _, row in top5.iterrows()
    ]

    return StatsResponse(
        total_hexagons=len(df),
        mean_crash_density=round(float(df["crash_density"].mean()), 4),
        risk_tier_distribution=df["risk_tier"].value_counts().to_dict(),
        top_5_highest_risk_hexagons=top5_list,
    )

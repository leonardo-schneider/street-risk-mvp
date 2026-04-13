"""
api/main.py — Street Risk MVP FastAPI application.

Serves precomputed H3 hexagon risk scores for Sarasota, FL and Tampa, FL.
Loads the multi-city Gold table and final trained LightGBM model on startup.

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

from model.predict import build_feature_matrix, compute_shap_values

from api.schemas import (
    CityStats,
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
ROOT            = Path(__file__).parents[1]
GOLD_PATH       = ROOT / "data" / "gold" / "training_table" / "multicity_gold.parquet"
MODEL_PATH      = ROOT / "model" / "final_model.pkl"
FEAT_COLS_PATH  = ROOT / "model" / "final_feature_columns.json"
SCALE_PATH      = ROOT / "model" / "city_scale_factors.json"

# S3 keys for production (RENDER=true)
S3_GOLD_KEY   = "gold/training_table/multicity_gold.parquet"
S3_MODEL_KEY  = "gold/final_model.pkl"
S3_FEAT_KEY   = "gold/final_feature_columns.json"
S3_SCALE_KEY  = "gold/city_scale_factors.json"

CLIP_COLS = [
    "clip_heavy_traffic", "clip_poor_lighting", "clip_no_sidewalks",
    "clip_damaged_road",  "clip_clear_road",    "clip_no_signals",
    "clip_parked_cars",
]

# ── city bounding boxes for auto-detection ────────────────────────────────────
CITY_BOUNDS = {
    "sarasota": {"lat": (27.2, 27.5), "lon": (-82.7, -82.3)},
    "tampa":    {"lat": (27.8, 28.1), "lon": (-82.7, -82.2)},
}


def detect_city(lat: float, lon: float) -> str:
    for city, bounds in CITY_BOUNDS.items():
        if (bounds["lat"][0] <= lat <= bounds["lat"][1] and
                bounds["lon"][0] <= lon <= bounds["lon"][1]):
            return city
    return "unknown"


# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Street Risk API",
    description="Micro-zone road risk scores from street-level imagery — Sarasota & Tampa, FL",
    version="2.0.0",
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


def _s3_load_bytes(s3, key: str) -> io.BytesIO:
    buf = io.BytesIO()
    s3.download_fileobj(BUCKET_NAME, key, buf)
    buf.seek(0)
    return buf


def _load_from_s3():
    """Load Gold table, model, feature columns, and scale factors from S3."""
    s3 = _s3_client()
    print(f"[startup] Loading from S3: s3://{BUCKET_NAME}/")

    df = pd.read_parquet(_s3_load_bytes(s3, S3_GOLD_KEY))
    print(f"[startup] Gold table: {len(df)} hexagons  cities={df['city'].value_counts().to_dict()}")

    model = joblib.load(_s3_load_bytes(s3, S3_MODEL_KEY))
    print(f"[startup] Model loaded: {S3_MODEL_KEY}")

    feat_cols = json.loads(_s3_load_bytes(s3, S3_FEAT_KEY).read().decode("utf-8"))
    print(f"[startup] Features: {len(feat_cols)}")

    scale_factors = json.loads(_s3_load_bytes(s3, S3_SCALE_KEY).read().decode("utf-8"))
    print(f"[startup] Scale factors: {scale_factors}")

    return df, model, feat_cols, scale_factors


def _load_from_local():
    """Load Gold table, model, feature columns, and scale factors from local disk."""
    df = pd.read_parquet(GOLD_PATH)
    print(f"[startup] Gold table: {len(df)} hexagons  cities={df['city'].value_counts().to_dict()}")
    model = joblib.load(MODEL_PATH)
    print(f"[startup] Model loaded: {MODEL_PATH.name}")
    feat_cols = json.loads(FEAT_COLS_PATH.read_text())
    scale_factors = json.loads(SCALE_PATH.read_text())
    print(f"[startup] Scale factors: {scale_factors}")
    return df, model, feat_cols, scale_factors


@app.on_event("startup")
def startup():
    """
    Load Gold table, model, feature columns, and city scale factors into memory.
    Uses S3 when RENDER=true (production), local disk otherwise (development).
    """
    is_render = os.getenv("RENDER", "").lower() == "true"

    if is_render:
        df, model, feat_cols, scale_factors = _load_from_s3()
    else:
        df, model, feat_cols, scale_factors = _load_from_local()

    # Min-max normalise within the full multicity table
    mn, mx = df["crash_density"].min(), df["crash_density"].max()
    df["risk_score_normalized"] = (df["crash_density"] - mn) / (mx - mn) if mx > mn else 0.0

    # Percentile rank across all hexagons
    df["percentile"] = df["crash_density"].rank(pct=True) * 100

    # Precompute SHAP values for all hexagons
    try:
        X_all   = build_feature_matrix(df.reset_index(drop=True), feat_cols)
        shap_df = compute_shap_values(model, X_all)
        shap_df.index = df["h3_index"].values if "h3_index" in df.columns else df.index
        _state["shap_df"] = shap_df
        print(f"[startup] SHAP values computed for {len(shap_df)} hexagons")
    except Exception as e:
        print(f"[startup] SHAP computation failed (non-fatal): {e}")
        _state["shap_df"] = None

    # Index by h3_index for O(1) lookups
    _state["df"]            = df.set_index("h3_index")
    _state["model"]         = model
    _state["feat_cols"]     = feat_cols
    _state["scale_factors"] = scale_factors
    _state["model_loaded"]  = True

    print(f"[startup] Ready. {'S3' if is_render else 'Local'} mode.")


# ── helpers ───────────────────────────────────────────────────────────────────

def _row_to_hex_response(h3_index: str, row: pd.Series) -> HexRiskResponse:
    clip_scores = {col: round(float(row[col]), 6) for col in CLIP_COLS if col in row.index}
    top_risk_factors = sorted(clip_scores, key=clip_scores.get, reverse=True)[:3]
    lat, lon = h3.cell_to_latlng(h3_index)

    shap_df = _state.get("shap_df")
    if shap_df is not None and h3_index in shap_df.index:
        shap_values = {col: round(float(shap_df.at[h3_index, col]), 6) for col in shap_df.columns}
    else:
        shap_values = {}

    city = str(row["city"]) if "city" in row.index else None

    return HexRiskResponse(
        h3_index=h3_index,
        city=city,
        crash_density=round(float(row["crash_density"]), 4),
        risk_tier=str(row["risk_tier"]),
        risk_score_normalized=round(float(row["risk_score_normalized"]), 6),
        top_risk_factors=top_risk_factors,
        clip_scores=clip_scores,
        hex_center=HexCenter(lat=round(lat, 6), lon=round(lon, 6)),
        percentile=round(float(row["percentile"]), 2),
        shap_values=shap_values,
    )


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health():
    """Health check — returns API status and model load state."""
    return HealthResponse(
        status="ok",
        model="loaded" if _state.get("model_loaded") else "not loaded",
        city="Sarasota, FL + Tampa, FL",
    )


@app.get(
    "/hex/{h3_index}",
    response_model=HexRiskResponse,
    tags=["Risk Scores"],
)
def get_hex(h3_index: str):
    """
    Return the precomputed risk score for a specific H3 res-9 hexagon.

    Works for hexagons from both Sarasota and Tampa.
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

    City is auto-detected from coordinates (Sarasota or Tampa bounding boxes)
    or can be passed explicitly. If the hexagon falls outside covered areas an
    'Unknown' response is returned.
    """
    h3_index = h3.latlng_to_cell(body.lat, body.lon, 9)
    df: pd.DataFrame = _state["df"]

    city = body.city if body.city else detect_city(body.lat, body.lon)

    if h3_index not in df.index:
        return HexUnknownResponse(
            h3_index=h3_index,
            city=city,
            risk_tier="Unknown",
            message=f"No data available for this location (city={city})",
        )

    return _row_to_hex_response(h3_index, df.loc[h3_index])


@app.get(
    "/map-data",
    response_model=GeoJSONFeatureCollection,
    tags=["Map"],
)
def map_data():
    """
    Return all hexagons (Sarasota + Tampa) as a GeoJSON FeatureCollection.

    Each Feature includes h3_index, crash_density, risk_tier,
    risk_score_normalized, percentile, and city as properties.
    """
    df: pd.DataFrame = _state["df"].reset_index()
    features = []

    for _, row in df.iterrows():
        boundary = h3.cell_to_boundary(row["h3_index"])
        coords   = [[lon, lat] for lat, lon in boundary]
        coords.append(coords[0])

        features.append(GeoJSONFeature(
            geometry=GeoJSONGeometry(type="Polygon", coordinates=[coords]),
            properties={
                "h3_index":              row["h3_index"],
                "city":                  row.get("city", None),
                "crash_density":         round(float(row["crash_density"]), 4),
                "risk_tier":             row["risk_tier"],
                "risk_score_normalized": round(float(row["risk_score_normalized"]), 6),
                "percentile":            round(float(row["percentile"]), 2),
            },
        ))

    return GeoJSONFeatureCollection(features=features)


@app.get("/stats", response_model=StatsResponse, tags=["Meta"])
def stats():
    """
    Return summary statistics for the full multi-city Gold table.

    Includes per-city breakdown, overall mean crash density, risk tier
    distribution, and the five highest-risk hexagons.
    """
    df: pd.DataFrame = _state["df"].reset_index()

    by_city = {}
    if "city" in df.columns:
        for city, grp in df.groupby("city"):
            by_city[city] = CityStats(
                hexagons=int(len(grp)),
                mean_crash_density=round(float(grp["crash_density"].mean()), 4),
            )

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
        by_city=by_city,
        mean_crash_density=round(float(df["crash_density"].mean()), 4),
        risk_tier_distribution=df["risk_tier"].value_counts().to_dict(),
        top_5_highest_risk_hexagons=top5_list,
    )

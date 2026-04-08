"""
api/schemas.py — Pydantic request/response models for the Street Risk API.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


# ── requests ──────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    lat: float = Field(..., description="Latitude of the location", ge=-90, le=90)
    lon: float = Field(..., description="Longitude of the location", ge=-180, le=180)


# ── responses ─────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model: str
    city: str


class HexCenter(BaseModel):
    lat: float
    lon: float


class HexRiskResponse(BaseModel):
    h3_index: str
    crash_density: float
    risk_tier: str
    risk_score_normalized: float = Field(..., description="Min-max scaled crash density, 0-1")
    top_risk_factors: List[str] = Field(..., description="Top 3 CLIP feature names sorted by score")
    clip_scores: Dict[str, float] = Field(..., description="Raw CLIP softmax scores for all 7 risk concepts")
    hex_center: HexCenter
    percentile: float = Field(..., description="Percentile of this hex vs all hexagons (0-100)")


class HexUnknownResponse(BaseModel):
    h3_index: str
    risk_tier: str
    message: str


class GeoJSONGeometry(BaseModel):
    type: str
    coordinates: List[Any]


class GeoJSONFeature(BaseModel):
    type: str = "Feature"
    geometry: GeoJSONGeometry
    properties: Dict[str, Any]


class GeoJSONFeatureCollection(BaseModel):
    type: str = "FeatureCollection"
    features: List[GeoJSONFeature]


class TopHex(BaseModel):
    h3_index: str
    crash_density: float
    risk_tier: str
    risk_score_normalized: float


class StatsResponse(BaseModel):
    total_hexagons: int
    mean_crash_density: float
    risk_tier_distribution: Dict[str, int]
    top_5_highest_risk_hexagons: List[TopHex]

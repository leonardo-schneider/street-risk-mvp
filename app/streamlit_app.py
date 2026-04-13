"""
app/streamlit_app.py

Streamlit frontend for the Street Risk MVP.
Displays a Folium map of Sarasota H3 hexagons coloured by risk tier,
lets users score any address, and shows a detailed risk card.

Run with:
    streamlit run app/streamlit_app.py
"""

import csv
import io as _io
import os
from pathlib import Path

import boto3
import folium
import matplotlib.colors as mcolors
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from streamlit_folium import st_folium

# ── env ───────────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parents[1] / ".env", override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
API_BASE       = os.getenv("API_BASE_URL", "https://street-risk-mvp.onrender.com")

# ── constants ─────────────────────────────────────────────────────────────────
SARASOTA_CENTER = (27.3364, -82.5307)
TAMPA_CENTER    = (27.9506, -82.4572)
CITY_CENTERS    = {"Sarasota, FL": SARASOTA_CENTER, "Tampa, FL": TAMPA_CENTER}
TIER_COLORS     = {"High": "#d73027", "Medium": "#fee090", "Low": "#1a9850"}
TIER_EMOJI      = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
DEFAULT_ADDRESS = "1 N Tamiami Trail, Sarasota, FL"

CLIP_LABEL = {
    "clip_heavy_traffic": "Heavy traffic",
    "clip_poor_lighting": "Poor lighting",
    "clip_no_sidewalks":  "No sidewalks",
    "clip_damaged_road":  "Damaged road",
    "clip_clear_road":    "Clear road",
    "clip_no_signals":    "No signals",
    "clip_parked_cars":   "Parked cars blocking view",
}

FEATURE_LABEL = {
    **CLIP_LABEL,
    "speed_limit_mean":           "Speed limit (mean)",
    "lanes_mean":                 "Lane count (mean)",
    "dist_to_intersection_mean":  "Distance to intersection",
    "point_count":                "Road point density",
    "clip_risk_prob":             "Visual risk (probe)",
}

# 5 dimensions used by the radar chart (feat_key, display_label)
RADAR_FEATURES = [
    ("speed_limit_mean",   "Speed limit"),
    ("lanes_mean",         "Lanes"),
    ("clip_no_signals",    "No signals"),
    ("clip_clear_road",    "Clear road"),
    ("clip_heavy_traffic", "Heavy traffic"),
]

RISK_EXPLANATION = {
    "High":    "This zone has significantly elevated crash activity — drive with extra caution.",
    "Medium":  "This zone has moderate crash risk, consistent with typical urban streets.",
    "Low":     "This zone has low crash density and appears relatively safe for drivers.",
    "Unknown": "This location is outside our Sarasota and Tampa coverage area.",
}

STREETVIEW_URL = (
    "https://maps.googleapis.com/maps/api/streetview"
    "?size=600x300&location={lat},{lon}&fov=90&pitch=0&key={key}"
)

LABELS_CSV    = Path(__file__).parents[1] / "data" / "labels" / "image_labels.csv"
MANIFEST_PATH = Path(__file__).parents[1] / "data" / "bronze" / "image_manifest.csv"
S3_BUCKET     = os.getenv("S3_BUCKET_NAME", "street-risk-mvp")
IS_LOCAL      = os.getenv("RENDER", "").lower() != "true"

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Street Risk",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── SpaceX design system ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;700&display=swap');

/* Canvas */
.stApp, .main, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #000000 !important;
}
.main .block-container {
    background: #000000 !important;
    padding-top: 0 !important;
    max-width: 100% !important;
}

/* Global text */
body, p, span, div, li, td, th, a {
    color: #f0f0fa;
    font-family: 'Barlow Condensed', Arial, Verdana, sans-serif !important;
}
h1, h2, h3, h4, h5, h6, [data-testid="stHeading"] {
    color: #f0f0fa !important;
    text-transform: uppercase !important;
    letter-spacing: 1.17px !important;
    font-weight: 700 !important;
    font-family: 'Barlow Condensed', Arial, Verdana, sans-serif !important;
    margin-bottom: 0.5rem !important;
}

/* Sidebar */
[data-testid="stSidebar"], [data-testid="stSidebar"] > div {
    background: #060606 !important;
    border-right: 1px solid rgba(240,240,250,0.12) !important;
}
[data-testid="stSidebar"] *:not(svg):not(path):not(canvas) {
    color: #f0f0fa !important;
}

/* Buttons */
.stButton > button {
    background: rgba(240,240,250,0.07) !important;
    color: #f0f0fa !important;
    border: 1px solid rgba(240,240,250,0.35) !important;
    border-radius: 32px !important;
    font-family: 'Barlow Condensed', Arial, Verdana, sans-serif !important;
    font-weight: 700 !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    padding: 10px 26px !important;
    box-shadow: none !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background: rgba(240,240,250,0.16) !important;
    border-color: rgba(240,240,250,0.65) !important;
    color: #ffffff !important;
    box-shadow: none !important;
}

/* Text inputs */
.stTextInput > div > div,
.stTextInput > div > div > input {
    background: rgba(240,240,250,0.04) !important;
    border: 1px solid rgba(240,240,250,0.18) !important;
    border-radius: 2px !important;
    color: #f0f0fa !important;
    caret-color: #f0f0fa !important;
    font-family: 'Barlow Condensed', Arial, Verdana, sans-serif !important;
}
.stTextInput > div > div > input::placeholder {
    color: rgba(240,240,250,0.3) !important;
}
.stTextInput label, [data-testid="stWidgetLabel"] > div > p {
    color: rgba(240,240,250,0.55) !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
    font-size: 10px !important;
    font-weight: 700 !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background: rgba(240,240,250,0.02) !important;
    border: 1px solid rgba(240,240,250,0.1) !important;
    border-radius: 2px !important;
    padding: 14px 12px !important;
}
[data-testid="stMetricValue"] {
    color: #f0f0fa !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    font-family: 'Barlow Condensed', Arial, Verdana, sans-serif !important;
}
[data-testid="stMetricLabel"] p {
    color: rgba(240,240,250,0.45) !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
    font-size: 9px !important;
    font-weight: 700 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(240,240,250,0.12) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: rgba(240,240,250,0.38) !important;
    text-transform: uppercase !important;
    letter-spacing: 1.4px !important;
    font-weight: 700 !important;
    font-size: 11px !important;
    border: none !important;
    padding: 14px 32px !important;
    border-radius: 0 !important;
}
.stTabs [aria-selected="true"] {
    color: #f0f0fa !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    background: #f0f0fa !important;
    height: 2px !important;
}

/* Dividers */
hr {
    border: none !important;
    border-top: 1px solid rgba(240,240,250,0.1) !important;
    margin: 18px 0 !important;
}
[data-testid="stDivider"] {
    border-color: rgba(240,240,250,0.1) !important;
}

/* Alerts */
[data-testid="stAlert"] {
    background: rgba(240,240,250,0.03) !important;
    border: 1px solid rgba(240,240,250,0.14) !important;
    border-left: 3px solid rgba(240,240,250,0.45) !important;
    border-radius: 2px !important;
}
[data-testid="stAlert"] p { color: #f0f0fa !important; }

/* Captions */
[data-testid="stCaptionContainer"] p, .stCaption {
    color: rgba(240,240,250,0.4) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    font-size: 9px !important;
    font-weight: 700 !important;
}

/* Map / iframe */
[data-testid="stIframe"], .element-container iframe {
    border: 1px solid rgba(240,240,250,0.15) !important;
    border-radius: 2px !important;
}

/* Images */
[data-testid="stImage"] img {
    border: 1px solid rgba(240,240,250,0.1) !important;
    border-radius: 2px !important;
}

/* Progress bar */
.stProgress > div > div > div { background: #f0f0fa !important; }

/* Spinner */
.stSpinner > div { border-top-color: #f0f0fa !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; background: #000000; }
::-webkit-scrollbar-thumb { background: rgba(240,240,250,0.18); }

/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
.stDeployButton { display: none; }

/* Fix sidebar toggle — hide broken text button, style native collapse control */
button[kind="header"] {
    display: none !important;
}

section[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem;
}

[data-testid="collapsedControl"] {
    background-color: #1a1a1a !important;
    border: 1px solid #333 !important;
    border-radius: 0 !important;
    color: #ffffff !important;
}

[data-testid="collapsedControl"]:hover {
    background-color: #333 !important;
    border-color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# ── API helpers ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch_stats():
    try:
        r = requests.get(f"{API_BASE}/stats", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def fetch_map_data():
    try:
        r = requests.get(f"{API_BASE}/map-data", timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def score_location(lat: float, lon: float) -> dict:
    try:
        r = requests.post(
            f"{API_BASE}/predict",
            json={"lat": lat, "lon": lon},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def geocode_address(address: str):
    """Convert an address string to (lat, lon) via Google Geocoding API."""
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_MAPS_API_KEY is not set. Cannot geocode addresses.")
        return None
    try:
        r = requests.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"address": address, "key": GOOGLE_API_KEY},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        if data["status"] != "OK" or not data["results"]:
            return None
        loc = data["results"][0]["geometry"]["location"]
        return loc["lat"], loc["lng"]
    except Exception:
        return None


@st.cache_data(ttl=3600)
def load_gold_cached():
    """Load the multi-city Gold table for feature-value lookups (radar chart). Cached 1 h."""
    gold_local = Path(__file__).parents[1] / "data" / "gold" / "training_table" / "multicity_gold.parquet"
    if gold_local.exists():
        return pd.read_parquet(gold_local).set_index("h3_index")
    try:
        s3 = boto3.client(
            "s3",
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        buf = _io.BytesIO()
        s3.download_fileobj(S3_BUCKET, "gold/training_table/multicity_gold.parquet", buf)
        buf.seek(0)
        return pd.read_parquet(buf).set_index("h3_index")
    except Exception:
        return None


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
        st.rerun()
    if col_lo.button("🟢 Low Risk", use_container_width=True):
        save_label(s3_key, "low")
        st.rerun()


# ── map builder ───────────────────────────────────────────────────────────────

def build_map(geojson: dict, scored_lat=None, scored_lon=None, scored_data=None,
              map_center=None, zoom=13) -> folium.Map:
    center = map_center if map_center else SARASOTA_CENTER
    m = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB dark_matter")

    if geojson:
        for feature in geojson.get("features", []):
            props   = feature["properties"]
            score   = props.get("risk_score_normalized", 0.0)
            pct     = props.get("percentile", score * 100) / 100
            tier    = props.get("risk_tier", "Low")
            density = props.get("crash_density", 0)
            h3idx   = props.get("h3_index", "")
            city_lbl= props.get("city", "")
            color   = risk_color(score, tier)

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
                    f"<span style='font-size:11px;color:#aaa'>{city_lbl.upper()}</span>&nbsp;"
                    f"<span style='font-size:11px;color:#888'>{h3idx}</span>",
                    sticky=False,
                ),
            ).add_to(m)

    # Gradient legend
    legend_html = """
    <div style="position:fixed;bottom:24px;left:24px;z-index:1000;
                background:rgba(0,0,0,0.85);
                padding:12px 16px;
                border:1px solid rgba(240,240,250,0.18);
                border-radius:2px;
                font-family:Arial,Verdana,sans-serif;
                font-size:9px;font-weight:700;
                letter-spacing:1.2px;text-transform:uppercase;
                color:#f0f0fa;">
        RISK SCORE<br>
        <div style="background:linear-gradient(to right,#1a9850,#fee090,#d73027);
                    width:110px;height:6px;border-radius:1px;margin:6px 0 4px 0;"></div>
        <div style="display:flex;justify-content:space-between;width:110px;color:rgba(240,240,250,0.55);">
            <span>LOW</span><span>HIGH</span>
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


# ── risk card helpers ─────────────────────────────────────────────────────────

def risk_badge(tier: str) -> str:
    border = {"High": "#cf2b2b", "Medium": "#9e6200", "Low": "#1a6b3a"}.get(tier, "rgba(240,240,250,0.25)")
    text   = {"High": "#e87070", "Medium": "#d4983c", "Low": "#4caf80"}.get(tier, "#f0f0fa")
    label  = {"High": "HIGH RISK", "Medium": "MEDIUM RISK", "Low": "LOW RISK"}.get(tier, "UNKNOWN")
    return (
        f'<div style="'
        f'display:inline-block;'
        f'background:rgba(240,240,250,0.04);'
        f'color:{text};'
        f'border:1px solid {border};'
        f'padding:10px 22px;'
        f'border-radius:2px;'
        f'font-family:Barlow Condensed,Arial,Verdana,sans-serif;'
        f'font-size:0.85rem;'
        f'font-weight:700;'
        f'letter-spacing:2.5px;'
        f'text-transform:uppercase;'
        f'">{label}</div>'
    )


def risk_color(score: float, tier: str = "Medium") -> str:
    """Map a 0-1 risk score to a hex color, constrained to the tier's color band."""
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "risk", ["#1a9850", "#fee090", "#d73027"]
    )
    tier_range = {"Low": (0.0, 0.38), "Medium": (0.38, 0.65), "High": (0.65, 1.0)}
    lo, hi = tier_range.get(tier, (0.0, 1.0))
    mapped = lo + float(score) * (hi - lo)
    return mcolors.to_hex(cmap(max(0.0, min(1.0, mapped))))


def top_factors_chart(top_risk_factors: list, hex_data: dict) -> go.Figure:
    labels = [CLIP_LABEL.get(f, f).upper() for f in top_risk_factors]
    clip_scores = hex_data.get("clip_scores", {})
    scores = [round(clip_scores.get(f, 0) * 100, 3) for f in top_risk_factors]
    colors = ["#cf2b2b", "#9e4a00", "#6b6b00"]

    fig = go.Figure(go.Bar(
        x=scores, y=labels, orientation="h",
        marker_color=colors[:len(labels)],
        text=[f"{s:.1f}%" for s in scores],
        textposition="outside",
        textfont=dict(color="#f0f0fa", size=10),
    ))
    fig.update_layout(
        height=180,
        margin=dict(l=0, r=50, t=10, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(color="rgba(240,240,250,0.55)", size=9, family="Arial"),
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def shap_chart(shap_values: dict, tier: str) -> go.Figure:
    """Horizontal bar chart of SHAP values sorted by absolute magnitude (top 8)."""
    if not shap_values:
        return go.Figure()

    items  = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
    labels = [FEATURE_LABEL.get(k, k).upper() for k, _ in items]
    values = [v for _, v in items]
    colors = ["#cf2b2b" if v > 0 else "#1a6b3a" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.1f}" for v in values],
        textposition="outside",
        textfont=dict(color="#f0f0fa", size=10),
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=0, r=50, t=10, b=10),
        xaxis=dict(
            title=dict(text="IMPACT ON CRASH DENSITY", font=dict(color="rgba(240,240,250,0.35)", size=9)),
            tickfont=dict(color="rgba(240,240,250,0.35)", size=9),
            zeroline=True,
            zerolinecolor="rgba(240,240,250,0.15)",
            gridcolor="rgba(240,240,250,0.05)",
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(color="rgba(240,240,250,0.55)", size=9, family="Arial"),
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def radar_chart(data_a: dict, data_b: dict, gold_df) -> go.Figure:  # noqa: C901
    """Spider chart comparing two hexagons across 5 normalized dimensions."""
    feat_labels = [lab for _, lab in RADAR_FEATURES]

    def norm_vals(data):
        h3idx = data.get("h3_index", "")
        clip  = data.get("clip_scores", {})
        vals  = []
        for feat, _ in RADAR_FEATURES:
            if feat.startswith("clip_"):
                vals.append(float(clip.get(feat, 0.0)))
            elif gold_df is not None and h3idx in gold_df.index and feat in gold_df.columns:
                raw  = float(gold_df.at[h3idx, feat])
                lo   = float(gold_df[feat].min())
                hi   = float(gold_df[feat].max())
                vals.append((raw - lo) / (hi - lo) if hi > lo else 0.5)
            else:
                vals.append(0.5)
        return vals

    va = norm_vals(data_a)
    vb = norm_vals(data_b)

    # Close polygon
    theta = feat_labels + [feat_labels[0]]
    ra    = va + [va[0]]
    rb    = vb + [vb[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=ra, theta=theta, fill="toself", name="ZONE A",
        line=dict(color="#cf2b2b", width=1.5),
        fillcolor="rgba(207,43,43,0.12)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=rb, theta=theta, fill="toself", name="ZONE B",
        line=dict(color="#1a6b3a", width=1.5),
        fillcolor="rgba(26,107,58,0.12)",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(color="rgba(240,240,250,0.25)", size=8),
                gridcolor="rgba(240,240,250,0.08)",
                linecolor="rgba(240,240,250,0.1)",
            ),
            angularaxis=dict(
                tickfont=dict(color="rgba(240,240,250,0.55)", size=9, family="Arial"),
                gridcolor="rgba(240,240,250,0.08)",
                linecolor="rgba(240,240,250,0.1)",
            ),
        ),
        showlegend=True,
        legend=dict(
            font=dict(color="#f0f0fa", size=10),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(240,240,250,0.12)",
            borderwidth=1,
        ),
        height=340,
        margin=dict(l=40, r=40, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _score_address(addr_key: str, data_key: str, coords_key: str):
    """Geocode and score the address stored in session_state[addr_key]; store results."""
    addr = st.session_state.get(addr_key, "").strip()
    if not addr:
        st.warning("Enter an address first.")
        return
    with st.spinner(f"Geocoding…"):
        coords = geocode_address(addr)
    if coords is None:
        st.warning(f"Could not geocode: '{addr}'")
        return
    with st.spinner("Scoring…"):
        data = score_location(*coords)
    if "error" in data:
        st.error(f"API error: {data['error']}")
        return
    st.session_state[data_key]   = data
    st.session_state[coords_key] = coords


def render_compare_tab():
    st.markdown("""<div style="font-family:Barlow Condensed,Arial,Verdana,sans-serif;
        font-size:0.62rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;
        color:rgba(240,240,250,0.35);margin-bottom:4px;">COMPARE ZONES</div>
        <div style="font-family:Barlow Condensed,Arial,Verdana,sans-serif;
        font-size:0.72rem;color:rgba(240,240,250,0.2);letter-spacing:1px;text-transform:uppercase;
        margin-bottom:18px;">SCORE TWO ADDRESSES — SIDE BY SIDE RISK ANALYSIS</div>""",
        unsafe_allow_html=True)

    # Per-side session state
    for key in ("cmp_data_a", "cmp_coords_a", "cmp_data_b", "cmp_coords_b"):
        if key not in st.session_state:
            st.session_state[key] = None

    # ── input row ────────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.text_input("Address A", value="1 N Tamiami Trail, Sarasota, FL", key="cmp_addr_a")
        if st.button("Score A", key="btn_score_a", type="primary", use_container_width=True):
            _score_address("cmp_addr_a", "cmp_data_a", "cmp_coords_a")

    with col_b:
        st.text_input("Address B", value="700 N Ashley Dr, Tampa, FL", key="cmp_addr_b")
        if st.button("Score B", key="btn_score_b", type="primary", use_container_width=True):
            _score_address("cmp_addr_b", "cmp_data_b", "cmp_coords_b")

    data_a   = st.session_state.cmp_data_a
    data_b   = st.session_state.cmp_data_b
    coords_a = st.session_state.cmp_coords_a
    coords_b = st.session_state.cmp_coords_b

    if data_a is None and data_b is None:
        st.info("Score both addresses above to compare risk zones.")
        return

    st.divider()

    # ── side-by-side cards ───────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    for col, data, coords, label in [
        (col_left,  data_a, coords_a, "A"),
        (col_right, data_b, coords_b, "B"),
    ]:
        with col:
            if data is None:
                st.markdown(
                    f'<div style="'
                    f'border:1px dashed rgba(240,240,250,0.15);'
                    f'border-radius:2px;'
                    f'padding:40px 24px;'
                    f'text-align:center;'
                    f'color:rgba(240,240,250,0.2);'
                    f'font-family:Barlow Condensed,Arial,Verdana,sans-serif;'
                    f'font-size:9px;font-weight:700;letter-spacing:2px;text-transform:uppercase;">'
                    f'SCORE ADDRESS {label} TO SEE RESULTS</div>',
                    unsafe_allow_html=True,
                )
                continue

            tier      = data.get("risk_tier", "Unknown")
            city_name = data.get("city", "")
            st.markdown(f"**Location {label}**")
            st.markdown(risk_badge(tier), unsafe_allow_html=True)
            if city_name:
                st.markdown(
                    f'<div style="font-family:Barlow Condensed,Arial,Verdana,sans-serif;'
                    f'font-size:0.65rem;font-weight:700;letter-spacing:2px;'
                    f'text-transform:uppercase;color:rgba(240,240,250,0.38);'
                    f'margin-top:6px;">{city_name.upper()}, FL</div>',
                    unsafe_allow_html=True,
                )
            st.write("")

            if tier == "Unknown":
                st.warning(data.get("message", "Outside coverage area."))
                st.caption(f"H3: `{data.get('h3_index','')}`")
                continue

            m1, m2 = st.columns(2)
            m1.metric("Crashes / km²", f"{data.get('crash_density', 0):.1f}")
            m2.metric("Riskier than", f"{data.get('percentile', 0):.0f}% of zones")

            if GOOGLE_API_KEY and coords:
                lat, lon = coords
                sv_url = STREETVIEW_URL.format(lat=lat, lon=lon, key=GOOGLE_API_KEY)
                st.image(sv_url, use_container_width=True)

            shap_vals = data.get("shap_values", {})
            if shap_vals:
                st.caption("Feature contributions (SHAP)")
                st.plotly_chart(
                    shap_chart(shap_vals, tier),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            else:
                st.caption("SHAP values not available (API still loading).")

            st.caption(f"H3: `{data.get('h3_index','')}`")

    # ── comparison summary (both scored and in coverage) ─────────────────────
    if not (data_a and data_b):
        return
    if data_a.get("risk_tier") == "Unknown" or data_b.get("risk_tier") == "Unknown":
        return

    st.markdown("<hr style='border-color:rgba(240,240,250,0.08);margin:18px 0;'>", unsafe_allow_html=True)
    st.markdown("""<div style="font-family:Barlow Condensed,Arial,Verdana,sans-serif;
        font-size:0.62rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;
        color:rgba(240,240,250,0.35);margin-bottom:14px;">COMPARATIVE ANALYSIS</div>""",
        unsafe_allow_html=True)

    density_a = data_a.get("crash_density", 0) or 0.0
    density_b = data_b.get("crash_density", 0) or 0.0

    if abs(density_a - density_b) < 0.01:
        st.info("Both zones have nearly identical crash density.")
    elif density_a >= density_b:
        times = density_a / max(density_b, 0.001)
        st.markdown(
            f"**Zone A is {times:.1f}x riskier than Zone B** "
            f"({density_a:.1f} vs {density_b:.1f} crashes/km²)"
        )
    else:
        times = density_b / max(density_a, 0.001)
        st.markdown(
            f"**Zone B is {times:.1f}x riskier than Zone A** "
            f"({density_b:.1f} vs {density_a:.1f} crashes/km²)"
        )

    # Primary risk driver difference
    shap_a = data_a.get("shap_values", {})
    shap_b = data_b.get("shap_values", {})
    if shap_a and shap_b:
        common = set(shap_a) & set(shap_b)
        if common:
            top_feat = max(common, key=lambda f: abs(shap_a[f] - shap_b[f]))
            lab_feat = FEATURE_LABEL.get(top_feat, top_feat)
            va, vb   = shap_a[top_feat], shap_b[top_feat]
            st.markdown(
                f"**Primary risk driver difference:** {lab_feat}  "
                f"(A: `{va:+.2f}` vs B: `{vb:+.2f}` SHAP contribution)"
            )

    # Radar chart
    gold_df = load_gold_cached()
    st.plotly_chart(
        radar_chart(data_a, data_b, gold_df),
        use_container_width=True,
        config={"displayModeBar": False},
    )


# ── main UI ───────────────────────────────────────────────────────────────────

# Header banner
st.markdown("""
<div style="
    width:100%;
    background:#000000;
    border-bottom:1px solid rgba(240,240,250,0.12);
    padding:28px 0 22px 0;
    margin-bottom:4px;
    text-align:left;
">
  <div style="
    font-family:Barlow Condensed,Arial,Verdana,sans-serif;
    font-size:2.6rem;
    font-weight:700;
    letter-spacing:3.5px;
    text-transform:uppercase;
    color:#f0f0fa;
    line-height:1;
    margin-bottom:8px;
  ">STREET RISK</div>
  <div style="
    font-family:Barlow Condensed,Arial,Verdana,sans-serif;
    font-size:0.68rem;
    font-weight:700;
    letter-spacing:2.5px;
    text-transform:uppercase;
    color:rgba(240,240,250,0.38);
    line-height:1;
  ">MICRO-ZONE ROAD RISK INTELLIGENCE &mdash; SARASOTA, FL + TAMPA, FL</div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="
        font-family:Barlow Condensed,Arial,Verdana,sans-serif;
        font-size:0.62rem;font-weight:700;
        letter-spacing:2px;text-transform:uppercase;
        color:rgba(240,240,250,0.35);
        margin-bottom:10px;margin-top:6px;
    ">CITY</div>
    """, unsafe_allow_html=True)
    selected_city = st.selectbox("City", ["Sarasota, FL", "Tampa, FL"], label_visibility="collapsed")

    st.markdown("<hr style='border-color:rgba(240,240,250,0.08);margin:14px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="
        font-family:Barlow Condensed,Arial,Verdana,sans-serif;
        font-size:0.62rem;font-weight:700;
        letter-spacing:2px;text-transform:uppercase;
        color:rgba(240,240,250,0.35);
        margin-bottom:14px;
    ">SCORE A LOCATION</div>
    """, unsafe_allow_html=True)
    address = st.text_input("Address", value=DEFAULT_ADDRESS)
    score_btn = st.button("Score this location", type="primary", use_container_width=True)

    st.markdown("<hr style='border-color:rgba(240,240,250,0.08);margin:18px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="
        font-family:Barlow Condensed,Arial,Verdana,sans-serif;
        font-size:0.62rem;font-weight:700;
        letter-spacing:2px;text-transform:uppercase;
        color:rgba(240,240,250,0.35);
        margin-bottom:14px;
    ">SYSTEM STATUS</div>
    """, unsafe_allow_html=True)

    stats = fetch_stats()
    if stats:
        # Show stats for selected city
        city_key = selected_city.split(",")[0].lower()  # "sarasota" or "tampa"
        by_city  = stats.get("by_city", {})
        city_stats = by_city.get(city_key, {})

        n_hexes      = city_stats.get("hexagons",          stats["total_hexagons"])
        mean_density = city_stats.get("mean_crash_density", stats["mean_crash_density"])

        st.metric("Hexagons scored", n_hexes)
        st.metric("Mean crash density", f"{mean_density:.1f} /km²")
        tier_dist = stats.get("risk_tier_distribution", {})
        if tier_dist:
            order  = ["High", "Medium", "Low"]
            labels = [t for t in order if t in tier_dist]
            values = [tier_dist[t] for t in labels]
            colors = {"High": "#cf2b2b", "Medium": "#9e6200", "Low": "#1a6b3a"}
            bar_colors = [colors[t] for t in labels]
            fig_sb = go.Figure(go.Bar(
                x=labels,
                y=values,
                marker_color=bar_colors,
                text=values,
                textposition="outside",
                textfont=dict(color="rgba(240,240,250,0.55)", size=9),
            ))
            fig_sb.update_layout(
                height=140,
                margin=dict(l=0, r=0, t=8, b=0),
                xaxis=dict(
                    tickfont=dict(color="rgba(240,240,250,0.4)", size=9, family="Arial"),
                    showgrid=False, zeroline=False, showline=False,
                ),
                yaxis=dict(visible=False),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(fig_sb, use_container_width=True, config={"displayModeBar": False})
    else:
        st.warning("API unavailable — start the FastAPI server on port 8000.")

# Session state
if "scored"    not in st.session_state: st.session_state.scored    = False
if "score_lat" not in st.session_state: st.session_state.score_lat = None
if "score_lon" not in st.session_state: st.session_state.score_lon = None
if "score_data" not in st.session_state: st.session_state.score_data = None
if "hex_detail" not in st.session_state: st.session_state.hex_detail = None

# Scoring action
if score_btn and address.strip():
    coords = geocode_address(address.strip())
    if coords is None:
        st.warning(f"Could not geocode address: '{address}'. Try a more specific address.")
    else:
        lat, lon = coords
        result   = score_location(lat, lon)
        if "error" in result:
            st.error(f"API error: {result['error']}")
        else:
            st.session_state.scored    = True
            st.session_state.score_lat = lat
            st.session_state.score_lon = lon
            st.session_state.score_data = result

            # clip_scores are already in the /predict response
            st.session_state.hex_detail = result

# Load map data
geojson = fetch_map_data()

# Tabs
if IS_LOCAL:
    tab_map, tab_compare, tab_label = st.tabs(["Risk Map", "Compare", "Label Images"])
else:
    tab_map, tab_compare = st.tabs(["Risk Map", "Compare"])

with tab_map:
    col_map, col_card = st.columns([6, 4])

    with col_map:
        st.markdown(
            f'<div style="font-family:Barlow Condensed,Arial,Verdana,sans-serif;'
            f'font-size:0.62rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;'
            f'color:rgba(240,240,250,0.35);margin-bottom:8px;">'
            f'RISK MAP — {selected_city.upper()}</div>',
            unsafe_allow_html=True,
        )
        map_center = CITY_CENTERS.get(selected_city, SARASOTA_CENTER)
        m = build_map(
            geojson,
            scored_lat=st.session_state.score_lat,
            scored_lon=st.session_state.score_lon,
            scored_data=st.session_state.score_data,
            map_center=map_center,
            zoom=13,
        )
        st_folium(m, width=None, height=520, returned_objects=[])

    with col_card:
        st.markdown("""<div style="font-family:Barlow Condensed,Arial,Verdana,sans-serif;
            font-size:0.62rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;
            color:rgba(240,240,250,0.35);margin-bottom:8px;">ZONE ASSESSMENT</div>""",
            unsafe_allow_html=True)

        if not st.session_state.scored:
            st.markdown("""
            <div style="border:1px dashed rgba(240,240,250,0.12);border-radius:2px;
                padding:32px 20px;text-align:center;
                font-family:Barlow Condensed,Arial,Verdana,sans-serif;
                font-size:9px;font-weight:700;letter-spacing:2px;text-transform:uppercase;
                color:rgba(240,240,250,0.2);">
                ENTER AN ADDRESS AND SCORE TO BEGIN
            </div>""", unsafe_allow_html=True)
        else:
            data = st.session_state.score_data
            tier = data.get("risk_tier", "Unknown")

            if tier == "Unknown":
                st.warning(data.get("message", "No data available for this location."))
                st.caption(f"H3 index: {data.get('h3_index','')}")
            else:
                city_name = data.get("city", "")
                st.markdown(risk_badge(tier), unsafe_allow_html=True)
                if city_name:
                    st.markdown(
                        f'<div style="font-family:Barlow Condensed,Arial,Verdana,sans-serif;'
                        f'font-size:0.65rem;font-weight:700;letter-spacing:2px;'
                        f'text-transform:uppercase;color:rgba(240,240,250,0.38);'
                        f'margin-top:6px;">{city_name.upper()}, FL</div>',
                        unsafe_allow_html=True,
                    )
                st.write("")
                m1, m2 = st.columns(2)
                m1.metric("Crashes / km²", f"{data.get('crash_density', 0):.1f}")
                pct = data.get("percentile", 0)
                m2.metric("Riskier than", f"{pct:.0f}% of zones")
                st.caption("Top visual risk factors (CLIP scores)")
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
                st.caption(f"H3 {data.get('h3_index','')}")

with tab_compare:
    render_compare_tab()

if IS_LOCAL:
    with tab_label:
        render_label_tab()

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

RISK_EXPLANATION = {
    "High":    "This zone has significantly elevated crash activity — drive with extra caution.",
    "Medium":  "This zone has moderate crash risk, consistent with typical urban streets.",
    "Low":     "This zone has low crash density and appears relatively safe for drivers.",
    "Unknown": "This location is outside our Sarasota coverage area.",
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
    page_title="Street Risk — Sarasota Zone Risk Scorer",
    layout="wide",
    page_icon="🛣️",
)

# ── API helpers ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch_stats():
    try:
        r = requests.get(f"{API_BASE}/stats", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=300)
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


# ── risk card helpers ─────────────────────────────────────────────────────────

def risk_badge(tier: str) -> str:
    color = {"High": "#d73027", "Medium": "#e07b00", "Low": "#1a9850"}.get(tier, "#888")
    return (
        f'<div style="display:inline-block;background:{color};color:white;'
        f'padding:8px 22px;border-radius:20px;font-size:1.3rem;font-weight:700;">'
        f'{TIER_EMOJI.get(tier,"")} {tier} Risk</div>'
    )


def risk_color(score: float) -> str:
    """Map a 0-1 risk score to a hex color on a green→yellow→red gradient."""
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "risk", ["#1a9850", "#fee090", "#d73027"]
    )
    return mcolors.to_hex(cmap(max(0.0, min(1.0, float(score)))))


def top_factors_chart(top_risk_factors: list, hex_data: dict) -> go.Figure:
    labels = [CLIP_LABEL.get(f, f) for f in top_risk_factors]
    clip_scores = hex_data.get("clip_scores", {})
    scores = [round(clip_scores.get(f, 0) * 100, 3) for f in top_risk_factors]
    colors = ["#d73027", "#fc8d59", "#fee090"]

    fig = go.Figure(go.Bar(
        x=scores, y=labels, orientation="h",
        marker_color=colors[:len(labels)],
        text=[f"{s:.1f}%" for s in scores],
        textposition="outside",
    ))
    fig.update_layout(
        height=180,
        margin=dict(l=0, r=40, t=10, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


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


# ── main UI ───────────────────────────────────────────────────────────────────

# Header
st.title("🛣️ Street Risk — Sarasota Zone Risk Scorer")
st.markdown(
    "Predict micro-zone road crash risk from street-level imagery and road features. "
    "**Powered by Street View imagery + CLIP + LightGBM**"
)
st.divider()

# Sidebar
with st.sidebar:
    st.header("Score a location")
    address = st.text_input("Address", value=DEFAULT_ADDRESS)
    score_btn = st.button("Score this location", type="primary", use_container_width=True)

    st.divider()
    st.subheader("Project stats")

    stats = fetch_stats()
    if stats:
        st.metric("Hexagons scored", stats["total_hexagons"])
        st.metric("Mean crash density", f"{stats['mean_crash_density']:.1f} /km²")
        tier_dist = stats.get("risk_tier_distribution", {})
        st.bar_chart(
            pd.Series(tier_dist).rename("Count"),
            color=["#d73027"],
            height=160,
        )
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
    render_compare_tab()

if IS_LOCAL:
    with tab_label:
        render_label_tab()

"""
app/streamlit_app.py

Streamlit frontend for the Street Risk MVP.
Displays a Folium map of Sarasota H3 hexagons coloured by risk tier,
lets users score any address, and shows a detailed risk card.

Run with:
    streamlit run app/streamlit_app.py
"""

import os
from pathlib import Path

import folium
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from streamlit_folium import st_folium

# ── env ───────────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parents[1] / ".env", override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
API_BASE       = os.getenv("API_BASE_URL", "http://localhost:8000")

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


# ── map builder ───────────────────────────────────────────────────────────────

def build_map(geojson: dict, scored_lat=None, scored_lon=None, scored_data=None) -> folium.Map:
    m = folium.Map(location=SARASOTA_CENTER, zoom_start=13, tiles="CartoDB positron")

    if geojson:
        for feature in geojson.get("features", []):
            props   = feature["properties"]
            tier    = props.get("risk_tier", "Low")
            color   = TIER_COLORS.get(tier, "#cccccc")
            density = props.get("crash_density", 0)
            h3idx   = props.get("h3_index", "")

            folium.GeoJson(
                feature,
                style_function=lambda f, c=color: {
                    "fillColor":   c,
                    "color":       "#333333",
                    "weight":      0.5,
                    "fillOpacity": 0.55,
                },
                tooltip=folium.Tooltip(
                    f"<b>{tier} risk</b><br>Density: {density:.1f} crashes/km²<br>{h3idx}",
                    sticky=False,
                ),
            ).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
                padding:10px 14px;border-radius:8px;border:1px solid #ccc;font-size:13px;">
        <b>Risk tier</b><br>
        <span style="color:#d73027;">&#9632;</span> High<br>
        <span style="color:#fee090;">&#9632;</span> Medium<br>
        <span style="color:#1a9850;">&#9632;</span> Low
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Scored location marker
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
            icon=folium.Icon(color="red" if tier == "High" else "orange" if tier == "Medium" else "green",
                             icon="info-sign"),
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

# Layout
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
            # Badge
            st.markdown(risk_badge(tier), unsafe_allow_html=True)
            st.write("")

            # Metrics row
            m1, m2 = st.columns(2)
            m1.metric("Crashes per km²", f"{data.get('crash_density', 0):.1f}")
            pct = data.get("percentile", 0)
            m2.metric("Riskier than", f"{pct:.0f}% of zones")

            # Top risk factors chart
            st.caption("Top contributing risk factors (CLIP scores)")
            top3   = data.get("top_risk_factors", [])
            detail = st.session_state.hex_detail or data
            if top3:
                fig = top_factors_chart(top3, detail)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # Street View image
            lat = st.session_state.score_lat
            lon = st.session_state.score_lon
            if GOOGLE_API_KEY and lat and lon:
                sv_url = STREETVIEW_URL.format(lat=lat, lon=lon, key=GOOGLE_API_KEY)
                st.image(sv_url, caption="Street View at scored location", use_container_width=True)

            # Plain-English explanation
            st.info(RISK_EXPLANATION.get(tier, ""))
            st.caption(f"H3 index: `{data.get('h3_index','')}`")

"""
test_project.py -- Live smoke test for Street Risk prediction API.

Tests all three covered cities: Sarasota, Tampa, Orlando.

Usage:
    python test_project.py

Exits 0 if all cities pass, 1 if any fail.
"""

import sys
import time

import requests

BASE    = "https://street-risk-mvp.onrender.com"
TIMEOUT = 15

CITIES = [
    {"label": "Sarasota, FL", "lat": 27.3364, "lon": -82.5307, "expected_city": "sarasota"},
    {"label": "Tampa, FL",    "lat": 27.9506, "lon": -82.4572, "expected_city": "tampa"},
    {"label": "Orlando, FL",  "lat": 28.5383, "lon": -81.3792, "expected_city": "orlando"},
]


def wake_api():
    """GET /health, waiting up to 120 s for Render free tier to spin up."""
    url    = f"{BASE}/health"
    start  = time.time()
    warned = False

    while True:
        elapsed = time.time() - start
        if elapsed > 120:
            print("FAIL -- API did not respond within 120 seconds.")
            sys.exit(1)
        try:
            r = requests.get(url, timeout=TIMEOUT)
            if r.status_code == 200:
                if warned:
                    print(f"  API awake ({elapsed:.1f}s)")
                return
        except requests.exceptions.RequestException:
            pass

        if not warned and (time.time() - start) > 2:
            print("Waking up API...", flush=True)
            warned = True

        time.sleep(2)


def test_city(city):
    """POST /predict for one city. Returns (passed: bool, summary: str, failures: list)."""
    lat, lon = city["lat"], city["lon"]
    try:
        r = requests.post(
            f"{BASE}/predict",
            json={"lat": lat, "lon": lon},
            timeout=TIMEOUT,
        )
    except requests.exceptions.RequestException as exc:
        return False, f"request error: {exc}", [str(exc)]

    failures = []

    if r.status_code != 200:
        failures.append(f"HTTP {r.status_code}, expected 200")

    try:
        data = r.json()
    except ValueError:
        failures.append(f"non-JSON response: {r.text[:120]}")
        return False, "", failures

    if "risk_tier" not in data:
        failures.append("'risk_tier' missing")
    elif data["risk_tier"] not in ("Low", "Medium", "High", "Unknown"):
        failures.append(f"unexpected risk_tier: {data['risk_tier']!r}")

    tier_val = data.get("risk_tier")
    if tier_val in ("Low", "Medium", "High"):
        # Scored response must include crash_density
        if "crash_density" not in data:
            failures.append("'crash_density' missing from scored response")
    elif tier_val == "Unknown":
        # Hex outside deployed gold table — valid response, flag as coverage gap
        failures.append(
            f"outside coverage (risk_tier=Unknown) — "
            f"API may need redeployment with v5 gold table"
        )

    tier    = data.get("risk_tier", "--")
    density = data.get("crash_density", 0) or 0
    pct     = data.get("percentile", 0) or 0
    h3idx   = data.get("h3_index", "--")
    city_r  = data.get("city") or "--"

    summary = (
        f"  Location      : {lat}, {lon} ({city['label']})\n"
        f"  H3 Index      : {h3idx}\n"
        f"  City (API)    : {city_r}\n"
        f"  Risk Tier     : {tier}\n"
        f"  Crash Density : {density:.1f} crashes/km2\n"
        f"  Percentile    : {pct:.0f}th"
    )
    return len(failures) == 0, summary, failures


def run():
    print()
    print("Street Risk -- Prediction Test (3 cities)")
    print("=" * 42)

    wake_api()
    print()

    overall_pass = True

    for city in CITIES:
        print(f"[{city['label']}]")
        passed, summary, failures = test_city(city)
        print(summary)
        if failures:
            for msg in failures:
                print(f"  ASSERTION FAILED: {msg}")
            print(f"  Status: FAIL [X]")
            overall_pass = False
        else:
            print(f"  Status: PASS [OK]")
        print()

    print("=" * 42)
    if overall_pass:
        print("All 3 cities: PASS [OK]")
        sys.exit(0)
    else:
        print("One or more cities: FAIL [X]")
        sys.exit(1)


if __name__ == "__main__":
    run()

"""
tests/e2e_test.py

End-to-end test suite that validates the entire Street Risk MVP pipeline
against the live production API and local data files.

Usage:
    python tests/e2e_test.py              # test production API
    python tests/e2e_test.py --local      # test http://localhost:8000
"""

import argparse
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parents[1] / ".env", override=True)

DEFAULT_API = os.getenv("API_BASE_URL", "https://street-risk-mvp.onrender.com")
GOLD_PATH   = Path(__file__).parents[1] / "data" / "gold" / "training_table" / "sarasota_gold.parquet"

PASS = "PASS"
FAIL = "FAIL"

results = []   # list of (test_name, status, reason)


def run(name, fn):
    try:
        fn()
        results.append((name, PASS, ""))
        print(f"  [{PASS}] {name}")
    except AssertionError as e:
        results.append((name, FAIL, str(e)))
        print(f"  [FAIL] {name}: {e}")
    except Exception as e:
        results.append((name, FAIL, f"{type(e).__name__}: {e}"))
        print(f"  [FAIL] {name}: {type(e).__name__}: {e}")


# ── tests ─────────────────────────────────────────────────────────────────────

def test_health(base):
    r = requests.get(f"{base}/health", timeout=30)
    assert r.status_code == 200, f"HTTP {r.status_code}"
    body = r.json()
    assert body.get("status") == "ok", f"status={body.get('status')!r}"
    assert "city" in body, "missing 'city' field"


def test_stats(base):
    r = requests.get(f"{base}/stats", timeout=30)
    assert r.status_code == 200, f"HTTP {r.status_code}"
    body = r.json()
    assert body["total_hexagons"] == 385, f"total_hexagons={body['total_hexagons']}"
    assert body["mean_crash_density"] > 0, f"mean_crash_density={body['mean_crash_density']}"
    dist = body.get("risk_tier_distribution", {})
    assert len(dist) == 3, f"risk_tier_distribution has {len(dist)} keys, expected 3"


def test_high_risk_hex(base):
    h3 = "89441e702bbffff"
    r = requests.get(f"{base}/hex/{h3}", timeout=30)
    assert r.status_code == 200, f"HTTP {r.status_code}"
    body = r.json()
    assert body["risk_tier"] == "High", f"risk_tier={body['risk_tier']!r}"
    assert body["crash_density"] > 1000, f"crash_density={body['crash_density']}"
    assert body["percentile"] > 95, f"percentile={body['percentile']}"


def test_low_risk_hex(base):
    h3 = "89441e70017ffff"  # confirmed Low-tier hex in gold table
    r = requests.get(f"{base}/hex/{h3}", timeout=30)
    assert r.status_code == 200, f"HTTP {r.status_code}"
    body = r.json()
    required = ["h3_index", "crash_density", "risk_tier", "risk_score_normalized",
                "top_risk_factors", "hex_center", "percentile"]
    missing = [f for f in required if f not in body]
    assert not missing, f"Missing fields: {missing}"
    assert body["h3_index"] == h3, f"h3_index={body['h3_index']!r}"


def test_predict_known(base):
    r = requests.post(f"{base}/predict", json={"lat": 27.3257, "lon": -82.5590}, timeout=30)
    assert r.status_code == 200, f"HTTP {r.status_code}"
    body = r.json()
    assert "h3_index" in body, "missing 'h3_index'"
    valid_tiers = {"Low", "Medium", "High", "Unknown"}
    assert body.get("risk_tier") in valid_tiers, f"risk_tier={body.get('risk_tier')!r}"


def test_predict_outside_coverage(base):
    r = requests.post(f"{base}/predict", json={"lat": 27.1000, "lon": -82.3000}, timeout=30)
    assert r.status_code == 200, f"HTTP {r.status_code}"
    body = r.json()
    assert (
        body.get("risk_tier") == "Unknown" or "message" in body
    ), f"Expected Unknown tier or message field, got: {body}"


def test_map_data(base):
    r = requests.get(f"{base}/map-data", timeout=60)
    assert r.status_code == 200, f"HTTP {r.status_code}"
    body = r.json()
    assert body.get("type") == "FeatureCollection", f"type={body.get('type')!r}"
    features = body.get("features", [])
    assert len(features) == 385, f"len(features)={len(features)}, expected 385"
    for i, feat in enumerate(features[:5]):   # spot-check first 5
        assert "geometry" in feat, f"feature[{i}] missing 'geometry'"
        assert "properties" in feat, f"feature[{i}] missing 'properties'"
        props = feat["properties"]
        for key in ("h3_index", "crash_density", "risk_tier", "risk_score_normalized"):
            assert key in props, f"feature[{i}] properties missing '{key}'"


def test_data_integrity():
    try:
        import pandas as pd
    except ImportError:
        raise AssertionError("pandas not installed — cannot run local data check")

    assert GOLD_PATH.exists(), f"Gold table not found at {GOLD_PATH}"
    df = pd.read_parquet(GOLD_PATH)

    assert len(df) == 385, f"Expected 385 rows, got {len(df)}"

    # 16 data columns (excluding risk_tier which is derived)
    expected_cols = [
        "h3_index", "crash_density", "crash_count", "injury_rate",
        "clip_heavy_traffic", "clip_poor_lighting", "clip_no_sidewalks",
        "clip_damaged_road", "clip_clear_road", "clip_no_signals", "clip_parked_cars",
        "road_type_primary", "speed_limit_mean", "lanes_mean",
        "dist_to_intersection_mean", "point_count",
    ]
    missing_cols = [c for c in expected_cols if c not in df.columns]
    assert not missing_cols, f"Missing columns: {missing_cols}"
    assert len(df.columns) >= 16, f"Expected >= 16 columns, got {len(df.columns)}"

    nan_counts = df[expected_cols].isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0].to_dict()
    assert not cols_with_nan, f"NaN values found: {cols_with_nan}"

    neg = (df["crash_density"] < 0).sum()
    assert neg == 0, f"{neg} rows have negative crash_density"


# ── runner ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Street Risk MVP end-to-end tests")
    parser.add_argument("--local", action="store_true", help="Test localhost:8000 instead of production")
    args = parser.parse_args()

    base = "http://localhost:8000" if args.local else DEFAULT_API

    print(f"\n=== Street Risk MVP — E2E Test Suite ===")
    print(f"    Target: {base}\n")

    run("TEST 1 — Health check",             lambda: test_health(base))
    run("TEST 2 — Stats endpoint",           lambda: test_stats(base))
    run("TEST 3 — High risk hex lookup",     lambda: test_high_risk_hex(base))
    run("TEST 4 — Low risk hex lookup",      lambda: test_low_risk_hex(base))
    run("TEST 5 — Predict (known location)", lambda: test_predict_known(base))
    run("TEST 6 — Predict (outside coverage)", lambda: test_predict_outside_coverage(base))
    run("TEST 7 — Map data endpoint",        lambda: test_map_data(base))
    run("TEST 8 — Local data integrity",     test_data_integrity)

    passed = sum(1 for _, s, _ in results if s == PASS)
    failed = [(n, r) for n, s, r in results if s == FAIL]
    total  = len(results)

    print(f"\n{'='*42}")
    print(f"  {passed}/{total} tests passed")
    if failed:
        print(f"\n  Failed tests:")
        for name, reason in failed:
            print(f"    - {name}")
            print(f"      {reason}")
    print(f"{'='*42}\n")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()

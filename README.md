# Street Risk — Micro-Zone Road Risk Scorer

> Predict auto insurance risk at H3 hexagon resolution (~0.1 km²) using
> street-level imagery, road geometry, and crash history.

---

## The Problem

ZIP-code insurance pricing is too coarse. Two blocks apart in Sarasota, FL
can have a **160× difference** in crash density (9.4 vs 1,520 crashes/km²).
This system estimates road risk at H3 res-9 hexagon level using Google Street
View imagery scored by CLIP, combined with OSM road features, POI counts,
and FDOT crash records — producing a ranked risk score for every 0.1 km²
cell across Sarasota and Tampa, FL.

---

## Live Demo

| Component     | URL |
|---------------|-----|
| Streamlit App | https://street-risk-mvp.streamlit.app |
| FastAPI Docs  | https://street-risk-mvp.onrender.com/docs |

---

## How It Works

```
Street View images        OSM road network        FDOT crash records
(3,214 images, S3)        (road points + POI)     (111,084 crashes)
        |                       |                       |
        v                       v                       v
  CLIP zero-shot          Road aggregation        Hex aggregation
  7 risk concepts         per H3 hexagon          crash_density
  (Silver layer)          + 6 POI features        (Silver layer)
        |                       |                       |
        +----------+------------+                       |
                   v                                    v
         Gold table: 806 hexagons, 23 features, TARGET=crash_density
                   |
                   v
             LightGBM regressor
             (geographic train/test split — both cities)
                   |
                   v
         Risk score per hexagon (0-1 normalised)
                   |
          +---------+---------+
          v                   v
       FastAPI             Streamlit
    /predict, /hex         Folium map
    /map-data, /stats      risk card + compare tab
    (city auto-detect)     (Sarasota + Tampa)
```

---

## Headline Result

> Trained on Sarasota. Tested on Tampa with zero Tampa training examples.
> **Spearman 0.692** — the visual risk signal generalizes across cities.

CLIP features dominate cross-city transfer. `traffic_signals_count` is the
strongest single predictor in the final model (permutation importance 31,415
vs 16,972 for `speed_limit_mean`) — the model learned road risk structure,
not Sarasota geography.

---

## Model Evolution

| Version | Train | Test | Spearman | R² | Key addition |
|---------|-------|------|----------|----|--------------|
| v1 | Sarasota | Sarasota (geo split) | 0.666 | −0.594 | Baseline CLIP + road |
| v2 | Sarasota | **Tampa (zero-shot)** | **0.692** | — | Cross-city transfer |
| v3 final | Sarasota+Tampa | Both (geo split) | 0.632 | **+0.540** | POI features, R² positive |

v2→v3 trades 0.03 Spearman for **positive R²**: the final model estimates
magnitude correctly (not just ranking) because it trains on both city scales.
The cross-city zero-shot result (v2) remains the headline generalization proof.

### Why Spearman is the primary metric
Insurance pricing requires correct *ranking*, not exact magnitude. A model
that ranks every hex correctly but predicts the wrong absolute density still
prices risk in the right order. Spearman 0.692 means the model correctly
ranks ~87% of all hex pairs by relative crash risk.

### Why R² was negative in v1
The geographic test cluster is OOD by design — its mean crash density
differs from the training set mean. Predicting the training mean everywhere
scores R²=0; negative R² means the OOD density shift outweighs within-cluster
signal. The final multicity model trains on both city scales, so the test
cluster mean aligns better with training, giving R²=+0.54.

---

### Visual Signal Analysis — CLIP vs Structural Features (v1)

| Feature set      | Spearman | RMSE   | MAE    | Interpretation                    |
|------------------|----------|--------|--------|-----------------------------------|
| CLIP only        | 0.360    | 222.21 | 137.82 | Pure visual signal (Street View)  |
| Structural only  | 0.443    | 114.27 | 77.79  | Pure road geometry + speed limits |
| **Full model**   | **0.666**| **80.28** | **59.63** | Combined — best ranking      |

**CLIP adds +0.223 Spearman lift over structural features alone.**

### Permutation Importance — Final Model, Top 10

Mean MSE increase when a single feature is shuffled (combined test set).

| Feature | MSE increase | Signal type |
|---------|-------------|-------------|
| `traffic_signals_count` | 31,415 | POI |
| `speed_limit_mean`      | 16,972 | Structural |
| `gas_stations_count`    |  5,929 | POI |
| `clip_clear_road`       |  5,160 | CLIP |
| `lanes_mean`            |  2,694 | Structural |
| `clip_parked_cars`      |  2,653 | CLIP |
| `clip_no_signals`       |  2,078 | CLIP |
| `clip_poor_lighting`    |    696 | CLIP |
| `hospitals_count`       |     13 | POI |
| `road_unclassified`     |      0 | Road type |

POI features mean importance (6,139) is **4× the non-POI mean** (1,439).

---

## Architecture — Medallion Pipeline

```
Bronze  (S3: street-risk-mvp/bronze/)
  images/sarasota/  1,550 Street View JPEGs (640x640, 4 headings)
  images/tampa/     1,664 Street View JPEGs (640x640, 4 headings)
  crash/            111,084 FDOT crash records (Parquet)
  roads/            road points with OSM attributes (Parquet)

Silver  (S3: street-risk-mvp/silver/)
  image_features/   CLIP scores x7 concepts per hexagon (Parquet)
  crash_hex/        crash_density aggregated to H3 hexagons (Parquet)
  poi_features/     6 OSM POI counts per hexagon (Parquet)

Gold    (S3: street-risk-mvp/gold/)
  training_table/   806 hexagons x 23 features, zero NaN (Parquet)
  final_model.pkl   Trained LightGBM (Sarasota + Tampa)
  final_feature_columns.json
  city_scale_factors.json
```

**Distributed cloud stages:**

1. **Image ingestion** — `pipeline/ingestion/fetch_images.py` writes raw
   Street View images directly to S3 Bronze (city-prefixed paths), checking
   the cache before every request.
2. **CLIP batch extraction** — `pipeline/features/extract_clip_features.py`
   reads Bronze images from S3, runs batch CLIP inference, writes Silver
   Parquet back to S3.

---

## Tech Stack

| Layer       | Technology |
|-------------|------------|
| Storage     | AWS S3 (boto3) |
| Road network| OSMnx + OpenStreetMap |
| POI features| OSMnx `features_from_place` (6 tag groups) |
| Images      | Google Street View Static API |
| Crash data  | FDOT Open Data Hub (ArcGIS REST, layer 2000) |
| Vision      | CLIP `openai/clip-vit-base-patch32` (HuggingFace) |
| Model       | LightGBM regressor |
| Tracking    | MLflow (SQLite backend, artifacts to S3) |
| API         | FastAPI + Uvicorn (deployed on Render) |
| Frontend    | Streamlit + Folium (deployed on Streamlit Cloud) |
| Spatial     | H3 (Uber) resolution 9, ~0.1 km² per cell |

---

## Data Sources

| City | Images | Hexagons | Crashes | Headings |
|------|--------|----------|---------|----------|
| Sarasota, FL | 1,550 | 390 | 19,824 | 4 (0°/90°/180°/270°) |
| Tampa, FL | 1,664 | 416 | 91,260 | 4 (0°/90°/180°/270°) |
| **Total** | **3,214** | **806** | **111,084** | |

All images cached in S3 Bronze. Crash data from FDOT Open Data Hub
(Sarasota County + Hillsborough County), all years.

---

## Validation

- **Geographic train/test split** via KMeans (n=6) on H3 centroid coordinates
- One Sarasota cluster held out (within-city OOD), one Tampa cluster held out (cross-city OOD)
- Prevents spatial leakage — neighbouring hexagons share visual features
- Cross-city test (v2): train exclusively on Sarasota, evaluate on Tampa — **Spearman 0.692**

---

## Limitations & Honest Assessment

- `traffic_signals_count` and `speed_limit_mean` are the two strongest predictors
  in the final model — structural and POI signals drive most of the lift
- CLIP contributes meaningfully (+0.22 Spearman lift over structural alone) but is
  constrained by ~4 images per hexagon at fixed headings
- Tampa crash density (230/km²) is 2× Sarasota's (116/km²); the model ranks
  correctly across cities but absolute magnitudes need per-city recalibration
- `speed_limit` and `lanes` OSM tags missing for ~88%/~71% of residential roads;
  imputed with city defaults (25 mph, 1 lane)

---

## Experiments

### Linear Probe vs Zero-Shot CLIP

Trained logistic regression on 104 manually labeled Street View images.
AUC 0.836 at image-level classification. Despite strong image-level accuracy,
collapsing 7 zero-shot dimensions into one probability reduced Spearman from
0.666 to 0.492. The 7-dimensional zero-shot representation preserves structure
that gradient boosting exploits more effectively than a compressed scalar.

---

### NASA Black Marble nighttime lights
Extracted VNP46A2 radiance per hexagon. Spearman correlation 
with crash_density = 0.39, but 79% of urban hexagons saturate 
at max intensity — insufficient within-city variance at 5km 
resolution vs H3 res-9 (~350m). Useful for cross-city 
differentiation but discarded for within-city model.
Script preserved at pipeline/features/extract_nightlight_features.py

---

## Future Work

- ~~Zero-shot transfer to Tampa~~ — Done (Spearman 0.692)
- **Orlando as third city** for cross-city validation — confirm generalization holds across urban morphologies
- **iRAP supervised visual risk features** — integrate International Road Assessment Programme labels for CLIP fine-tuning
- **City-specific recalibration layer** — isotonic regression per city to correct magnitude scaling
- **GNN over H3 hexagon graph** — exploit spatial autocorrelation between neighbouring hexagons
- **More images per hex** — 5+ images vs current ~4 average at varied headings

---

## Run Locally

```bash
# 1. Clone
git clone https://github.com/leonardo-schneider/street-risk-mvp.git
cd street-risk-mvp

# 2. Install
pip install -r requirements.txt

# 3. Environment variables
cp .env.example .env
# Fill in: GOOGLE_MAPS_API_KEY, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
#          AWS_DEFAULT_REGION=us-east-1, S3_BUCKET_NAME=street-risk-mvp,
#          MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# 4. Infrastructure (one-time S3 setup)
python infrastructure/s3_setup.py

# 5. Run the full pipeline (multicity)
python pipeline/ingestion/sample_roads.py --city sarasota
python pipeline/ingestion/sample_roads.py --city tampa
python pipeline/ingestion/fetch_images.py --city sarasota
python pipeline/ingestion/fetch_images.py --city tampa --limit 414
python pipeline/ingestion/fetch_crash_data.py --city sarasota
python pipeline/ingestion/fetch_crash_data.py --city tampa
python pipeline/features/extract_clip_features.py --city sarasota
python pipeline/features/extract_clip_features.py --city tampa
python pipeline/features/extract_poi_features.py --city sarasota
python pipeline/features/extract_poi_features.py --city tampa
python pipeline/gold/build_gold_table.py --multicity

# 6. Train final model
python model/train_final.py

# 7. Start API
uvicorn api.main:app --reload --port 8000

# 8. Start Streamlit (separate terminal)
streamlit run app/streamlit_app.py

# 9. Run e2e tests
python tests/e2e_test.py --local
```

---

## Repository Structure

```
street-risk-mvp/
├── pipeline/
│   ├── ingestion/
│   │   ├── sample_roads.py          # OSMnx -> S3 Bronze road points (--city)
│   │   ├── fetch_images.py          # Street View -> S3 Bronze images (--city, --limit)
│   │   └── fetch_crash_data.py      # FDOT ArcGIS -> S3 Bronze + Silver (--city)
│   ├── features/
│   │   ├── extract_clip_features.py # CLIP batch inference -> S3 Silver (--city)
│   │   └── extract_poi_features.py  # OSM POI counts -> S3 Silver (--city)
│   └── gold/
│       └── build_gold_table.py      # Join layers -> Gold table (--multicity)
├── model/
│   ├── train.py                     # LightGBM v1 (Sarasota only)
│   ├── train_multicity.py           # 3-scenario multicity training
│   ├── train_final.py               # Final production model (Sarasota+Tampa)
│   ├── train_experiments.py         # Ridge / RF / XGBoost comparison
│   ├── visual_contribution.py       # CLIP vs structural ablation
│   ├── final_model.pkl              # Production LightGBM
│   ├── final_feature_columns.json
│   └── city_scale_factors.json
├── api/
│   ├── main.py                      # FastAPI (5 endpoints, city auto-detect)
│   └── schemas.py                   # Pydantic models
├── app/
│   └── streamlit_app.py             # Folium map + risk card + compare tab
├── infrastructure/
│   └── s3_setup.py                  # S3 bucket + prefix init
├── tests/
│   └── e2e_test.py                  # 12/12 e2e tests (prod + local)
├── docs/
│   ├── writeup.md                   # One-page prose writeup
│   └── screenshots/
│       ├── feature_importance.png
│       ├── permutation_importance.png
│       └── final_permutation_importance.png
├── render.yaml                      # Render.com deploy config
├── requirements.txt
└── .env.example
```

---

## End-to-End Tests

```bash
python tests/e2e_test.py           # 12/12 tests against production
python tests/e2e_test.py --local   # 12/12 tests against localhost:8000
```

Tests cover: health check, stats (multicity), high/low risk hex lookup,
predict (Sarasota + Tampa), outside-coverage handling, GeoJSON map data
with city properties, local Gold table integrity (23 columns, both cities),
Tampa H3 prefix validation, and cross-city city auto-detection.

---

*MLflow experiment: `sarasota-risk-model`. Cities: Sarasota, FL + Tampa, FL.*

# Street Risk — Micro-Zone Road Risk Scorer

> Predict auto insurance risk at H3 hexagon resolution (~0.1 km²)
> using street-level imagery, road geometry, traffic volume,
> and crash history. Validated across three Florida cities.

---

## The Problem

ZIP-code insurance pricing is too coarse. Two blocks apart in Sarasota, FL
can have a **160× difference** in crash density (9.4 vs 1,520 crashes/km²).
This system estimates road risk at H3 res-9 hexagon level using Google Street
View imagery scored by CLIP, combined with OSM road features, POI counts,
FDOT AADT traffic volume, and crash records — producing a ranked risk score
for every 0.1 km² cell across Sarasota, Tampa, and Orlando, FL.

---

## Live Demo

| Component | URL |
|-----------|-----|
| Streamlit App | https://street-risk-mvp.streamlit.app |
| FastAPI Docs | https://street-risk-mvp.onrender.com/docs |

---

## How It Works

```
Street View images        OSM road network        FDOT crash records
(4,878 images, S3)        (road points + POI)     (203,351 crashes)
        |                       |                       |
        v                       v                       v
  CLIP zero-shot          Road aggregation        Hex aggregation
  7 risk concepts         per H3 hexagon          crash_density
  (Silver layer)          + 6 POI features        (Silver layer)
        |                       |                       |
        |               FDOT AADT traffic               |
        |               (ArcGIS REST layer 7)           |
        |                       |                       |
        +----------+------------+                       |
                   v                                    v
      Gold table: 1,220 hexagons, 27 features, TARGET=crash_density
                   |
                   v
             LightGBM regressor
             (train: Sarasota + Tampa / test: Orlando zero-shot)
                   |
                   v
         Risk score per hexagon (0-1 normalised)
                   |
          +---------+---------+
          v                   v
       FastAPI             Streamlit
    /predict, /hex         Top-bar UI, no sidebar
    /map-data, /stats      Folium map + risk card
    (city auto-detect)     + compare tab
    3 cities live          (Sarasota, Tampa, Orlando)
```

---

## Headline Result

> Trained on Sarasota + Tampa. Tested on Orlando with zero Orlando training
> examples. **Spearman 0.878** — the model correctly ranks risk in 93.9% of
> Orlando hex pairs without ever seeing the city.

AADT (Annual Average Daily Traffic) is the dominant feature. `aadt_max`
alone has permutation importance 70,025 on the Orlando test set — 2× the
next feature — because high-traffic roads concentrate crash risk universally
across cities.

---

## Model Evolution

| Version | Train | Test | Spearman | R² | Key Addition |
|---------|-------|------|----------|----|--------------|
| v1 | Sarasota | Sarasota geo-OOD | 0.666 | −0.594 | Baseline CLIP + road |
| v2 | Sarasota | Tampa (zero-shot) | 0.692 | — | Cross-city transfer |
| v3 | Sara+Tampa | Both geo-OOD | 0.632 | +0.540 | POI features, R² positive |
| v4 | Sara+Tampa | Tampa cross-city | 0.792 | +0.628 | AADT — largest lift |
| **v5** | **Sara+Tampa** | **Orlando (zero-shot)** | **0.878** | **+0.577** | **3-city generalization** |

### Why Spearman is the primary metric
Insurance pricing requires correct *ranking*, not exact magnitude. A model
that ranks every hex correctly but predicts the wrong absolute density still
prices risk in the right order. Spearman 0.878 means the model correctly
ranks ~93.9% of all Orlando hex pairs by relative crash risk — on a city it
has never seen.

### Why R² was negative in v1
The geographic test cluster is OOD by design — its mean crash density
differs from the training set mean. The multicity model trains on both city
scales, aligning the test distribution with training, giving R²=+0.54→+0.58.

---

## Feature Importance — v5 (Orlando zero-shot test)

Top features by permutation importance (mean MSE increase when shuffled):

| Feature | MSE increase | Group |
|---------|-------------|-------|
| `aadt_max` | 70,025 | AADT |
| `aadt_segment_count` | 32,930 | AADT |
| `traffic_signals_count` | 22,106 | POI |
| `fast_food_count` | 3,447 | POI |
| `clip_poor_lighting` | 542 | CLIP |

**AADT group mean: 34,973 — 8× the POI group mean (4,428).**

CLIP contributes meaningfully as a cross-city visual prior even when AADT
and POI dominate: `clip_poor_lighting` ranks 5th and CLIP-only ablation
shows +0.22 Spearman lift over structural features alone.

---

## Architecture — Medallion Pipeline

```
Bronze  (S3: street-risk-mvp/bronze/)
  images/sarasota/   1,550 Street View JPEGs (640×640, 4 headings)
  images/tampa/      1,664 Street View JPEGs (640×640, 4 headings)
  images/orlando/    1,664 Street View JPEGs (640×640, 4 headings)
  crash/             FDOT crash records by county (Parquet)
  roads/             road points with OSM attributes (Parquet)

Silver  (S3: street-risk-mvp/silver/)
  image_features/    CLIP scores × 7 concepts per hexagon (Parquet)
  crash_hex/         crash_density aggregated to H3 hexagons (Parquet)
  poi_features/      6 OSM POI counts per hexagon (Parquet)
  aadt/              FDOT AADT: mean, max, segment count per hexagon (Parquet)

Gold    (S3: street-risk-mvp/gold/)
  training_table/    multicity_gold_v3.parquet — 1,220 hexagons × 27 features
  final_model_v5.pkl           Trained LightGBM (Sarasota + Tampa → Orlando)
  final_feature_columns_v5.json
  city_scale_factors.json
```

**Distributed cloud stages:**

1. **Image ingestion** — `pipeline/ingestion/fetch_images.py` writes raw
   Street View images directly to S3 Bronze (city-prefixed paths), checking
   the cache before every API request.
2. **CLIP batch extraction** — `pipeline/features/extract_clip_features.py`
   reads Bronze images from S3, runs batch CLIP inference, writes Silver
   Parquet back to S3.

**Frontend architecture:**

The Streamlit app uses a top-bar layout (no sidebar) — city selector, address
input, and SCORE button sit in a single `st.columns` row above the tabs,
eliminating sidebar toggle fragility entirely.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Storage | AWS S3 (boto3) |
| Road network | OSMnx + OpenStreetMap |
| POI features | OSMnx `features_from_place` (6 tag groups) |
| Traffic volume | FDOT AADT via ArcGIS REST (layer 7, WGS84) |
| Images | Google Street View Static API |
| Crash data | FDOT Open Data Hub (ArcGIS REST, layer 2000) |
| Vision | CLIP `openai/clip-vit-base-patch32` (HuggingFace) |
| Model | LightGBM regressor |
| Tracking | MLflow (SQLite backend, artifacts to S3) |
| API | FastAPI + Uvicorn (deployed on Render) |
| Frontend | Streamlit + Folium (deployed on Streamlit Cloud) |
| Spatial | H3 (Uber) resolution 9, ~0.1 km² per cell |

---

## Data Sources

| City | Images | Hexagons | Crashes | Headings |
|------|--------|----------|---------|----------|
| Sarasota, FL | 1,550 | 390 | 19,824 | 4 (0°/90°/180°/270°) |
| Tampa, FL | 1,664 | 416 | 91,260 | 4 (0°/90°/180°/270°) |
| Orlando, FL | 1,664 | 414 | 91,675 | 4 (0°/90°/180°/270°) |
| **Total** | **4,878** | **1,220** | **203,759** | |

All images cached in S3 Bronze. Crash data from FDOT Open Data Hub
(Sarasota / Hillsborough / Orange counties), all years available.

---

## Validation

- **v1–v4**: Geographic train/test split via KMeans (n=6) on H3 centroid
  coordinates — one cluster per city held out, prevents spatial leakage
- **v5**: Hard city split — train on Sarasota + Tampa (806 hexagons),
  test on Orlando (414 hexagons), zero Orlando examples seen during training
- Spearman rank correlation is the primary metric (insurance pricing = ranking)

---

## Experiments

### Feature Ablation (Sarasota geo-OOD test)

| Feature set | Spearman | Interpretation |
|-------------|----------|----------------|
| CLIP only | 0.360 | Pure visual signal |
| Structural only | 0.443 | Road geometry + speed limits |
| Full model | 0.792 | Combined — best ranking |

CLIP adds +0.22 Spearman lift over structural features alone.

### Linear Probe vs Zero-Shot CLIP

Trained logistic regression on 104 manually labeled Street View images.
AUC 0.836 at image-level classification. Collapsing 7 zero-shot dimensions
into one probability reduced Spearman from 0.666 to 0.492. The 7-dimensional
zero-shot representation preserves structure that gradient boosting exploits
more effectively than a compressed scalar.

### NASA Black Marble Nighttime Lights

Spearman correlation with crash_density = 0.39, but 79% of urban hexagons
saturated at max intensity — 5 km resolution insufficient for within-city
H3 res-9 discrimination. Discarded for the within-city model; script
preserved at `pipeline/features/extract_nightlight_features.py` for
cross-city use cases.

---

## Limitations & Honest Assessment

- AADT and traffic signal count are the two strongest predictors — structural
  and road-network signals drive most of the lift over CLIP alone
- CLIP contributes meaningfully (+0.22 Spearman) but is constrained by ~4
  images per hexagon at fixed headings
- Orlando crash density (303/km²) is 2.6× Sarasota's (116/km²); the model
  ranks correctly zero-shot but absolute magnitudes would benefit from
  per-city isotonic recalibration for pricing use
- `speed_limit` and `lanes` OSM tags missing for ~88%/~71% of residential
  roads; imputed with city defaults (25 mph, 1 lane)
- FDOT AADT covers state-managed roads only; local residential streets have
  aadt_mean = 0 (correct for road type, not a data gap)

---

## Future Work

- ~~Zero-shot transfer to Tampa~~ ✅ Spearman 0.692
- ~~3-city validation (Orlando)~~ ✅ Spearman 0.878
- Visual-only mode: CLIP + OSM without AADT or crash data (cold-start cities)
- iRAP supervised visual risk features for CLIP fine-tuning
- Log-transform of target to reduce RMSE skew from high-density outliers
- GNN over H3 hexagon graph to exploit spatial autocorrelation
- Jacksonville / Miami expansion

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

# 5. Run the full pipeline (all three cities)
python pipeline/ingestion/sample_roads.py --city sarasota
python pipeline/ingestion/sample_roads.py --city tampa
python pipeline/ingestion/sample_roads.py --city orlando
python pipeline/ingestion/fetch_images.py --city sarasota
python pipeline/ingestion/fetch_images.py --city tampa
python pipeline/ingestion/fetch_images.py --city orlando
python pipeline/ingestion/fetch_crash_data.py --city sarasota
python pipeline/ingestion/fetch_crash_data.py --city tampa
python pipeline/ingestion/fetch_crash_data.py --city orlando
python pipeline/features/extract_clip_features.py --city sarasota
python pipeline/features/extract_clip_features.py --city tampa
python pipeline/features/extract_clip_features.py --city orlando
python pipeline/features/extract_poi_features.py --city sarasota
python pipeline/features/extract_poi_features.py --city tampa
python pipeline/features/extract_poi_features.py --city orlando
python pipeline/features/extract_aadt_features.py          # all cities
python pipeline/gold/build_gold_table.py --multicity --aadt --orlando

# 6. Train v5 model (Sara+Tampa train, Orlando zero-shot test)
python model/train_final_v5.py

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
│   │   ├── sample_roads.py               # OSMnx -> S3 Bronze (--city)
│   │   ├── fetch_images.py               # Street View -> S3 Bronze (--city)
│   │   └── fetch_crash_data.py           # FDOT ArcGIS -> S3 Bronze+Silver (--city)
│   ├── features/
│   │   ├── extract_clip_features.py      # CLIP batch inference -> S3 Silver (--city)
│   │   ├── extract_poi_features.py       # OSM POI counts -> S3 Silver (--city)
│   │   ├── extract_aadt_features.py      # FDOT AADT -> S3 Silver (--city)
│   │   └── extract_nightlight_features.py# NASA VIIRS nightlights (experimental)
│   └── gold/
│       └── build_gold_table.py           # Join layers -> Gold table (--multicity --aadt --orlando)
├── model/
│   ├── train_final_v5.py                 # v5: Sara+Tampa train, Orlando zero-shot test
│   ├── train_final_v4.py                 # v4: multicity + AADT, KMeans split
│   ├── train_final.py                    # v3: multicity + POI, KMeans split
│   ├── train_multicity.py                # v2: cross-city zero-shot experiments
│   ├── train.py                          # v1: Sarasota-only baseline
│   ├── train_experiments.py              # Ridge / RF / XGBoost comparison
│   ├── visual_contribution.py            # CLIP vs structural ablation
│   ├── predict.py                        # build_feature_matrix + compute_shap_values
│   ├── final_model_v5.pkl                # Production LightGBM
│   ├── final_feature_columns_v5.json
│   └── city_scale_factors.json
├── api/
│   ├── main.py                           # FastAPI v5 (5 endpoints, 3-city auto-detect)
│   └── schemas.py                        # Pydantic models
├── app/
│   └── streamlit_app.py                  # Top-bar UI, Folium map, risk card, compare tab
├── infrastructure/
│   └── s3_setup.py                       # S3 bucket + prefix init
├── tests/
│   └── e2e_test.py                       # 12/12 e2e tests (prod + local)
├── docs/
│   ├── writeup.md
│   └── screenshots/
│       ├── feature_importance.png
│       ├── permutation_importance.png
│       ├── final_permutation_importance.png
│       └── v5_permutation_importance.png
├── render.yaml                           # Render.com deploy config
├── requirements.txt
└── .env.example
```

---

## End-to-End Tests

```bash
python tests/e2e_test.py           # 12/12 tests against production
python tests/e2e_test.py --local   # 12/12 tests against localhost:8000
```

Tests cover: health check (`model=v5`, 3 cities), stats (multicity breakdown),
high/low risk hex lookup, predict (Sarasota + Tampa + Orlando), outside-coverage
handling, GeoJSON map data with city properties, local Gold table integrity
(27 columns, all three cities), H3 prefix validation, and cross-city
city auto-detection.

---

*MLflow experiment: `sarasota-risk-model` · Cities: Sarasota FL, Tampa FL, Orlando FL · Model v5 in production*

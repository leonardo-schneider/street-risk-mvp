# Street Risk — Micro-Zone Road Risk Scorer

## Overview

Street Risk is a distributed ML pipeline that predicts auto insurance crash risk
at H3 hexagon resolution (~0.1 km²) for Sarasota, Florida. The motivation is
straightforward: ZIP-code pricing is too coarse. Two blocks apart in Sarasota
can have a 160× difference in crash density — 9.4 versus 1,520 crashes per km².
By operating at hexagon level, the system can distinguish a quiet residential
street from a high-speed arterial two blocks away.

## Pipeline

The system follows a Bronze-Silver-Gold medallion architecture, with all data
stored in AWS S3. The Bronze layer contains 770 Street View images (Google
Static API, 640×640), 6,655 OSM road points sampled from the Sarasota drive
network, and 19,824 FDOT crash records pulled from the state's ArcGIS REST
endpoint. Two distributed cloud stages produce the Silver layer: an image
ingestion job writes raw JPEGs directly to S3, and a CLIP batch extraction
job reads those images, scores them against seven road-risk text concepts using
`openai/clip-vit-base-patch32`, and writes feature Parquet files back to S3.
The Gold table joins CLIP features, road aggregates, and crash density into
385 hexagons with 16 features and zero missing values.

## Model & Results

A LightGBM regressor trained on the Gold table achieves a Spearman rank
correlation of 0.666 on a geographically held-out test cluster — meaning the
model correctly ranks 83% of all hexagon pairs by relative crash risk. The
negative R² (−0.594) is expected: the test set is a spatially distinct cluster
by design, so absolute magnitudes shift while relative rankings hold. Among four
models evaluated (LightGBM, XGBoost, Random Forest, Ridge), LightGBM performs
best on every metric.

Permutation importance analysis shows that structural features (speed limit,
lane count) contribute 64% of the model's Spearman signal, while CLIP visual
features contribute the remaining 36%. Crucially, an ablation study confirms
that neither signal alone is sufficient: CLIP-only achieves Spearman 0.360,
structural-only 0.443, and the combined model 0.666. CLIP adds +0.22 Spearman
lift over road geometry alone — a meaningful contribution given only ~1.3 images
per hexagon on average.

## Deployment

The trained model is served via a FastAPI application deployed on Render, with
four endpoints: `/health`, `/hex/{h3_index}`, `/predict` (lat/lon to risk score),
and `/map-data` (GeoJSON FeatureCollection of all 385 hexagons). A Streamlit
frontend renders a Folium choropleth map colored by risk tier and a risk card
showing crash density, percentile rank, top CLIP factors, and the Street View
image at the scored location. An eight-test end-to-end suite validates the full
stack against the production API.

## Limitations

The model's primary weakness is OSM data quality: speed limit and lane count
tags are missing for 88% and 71% of residential road points respectively,
requiring default imputation. CLIP features have low variance across similar
suburban streets, and would benefit from 5+ images per hexagon rather than the
current ~1.3. Future work includes zero-shot transfer to Tampa (applying the
Sarasota model with no retraining), CLIP fine-tuning on road-safety imagery,
and adding AADT traffic counts as a temporal feature.

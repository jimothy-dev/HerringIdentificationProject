# Herring Spawn Labeler

A local app for building a training dataset of Pacific herring spawns as seen from satellite. Spawn shows up as milky turquoise water along the coast for a few days around each event. The app takes DFO's spawn index records (2016+, ~3,000 usable), pulls Sentinel-2 and Landsat 8/9 scenes from Google Earth Engine for the days around each spawn, and lets you label them positive or negative. Labels go to `data/labels.csv`; labeled scenes also download a multiband GeoTIFF chip to `data/chips/` for training.

## Running it

```
run.bat
```

Then open http://127.0.0.1:8137. Without Earth Engine credentials the app still runs in a mock mode with placeholder imagery (labels made there are practice only).

## Earth Engine setup (one time)

1. Register a free non-commercial project at https://code.earthengine.google.com/register
2. Run `.venv\Scripts\earthengine authenticate` from the project folder
3. Put your cloud project id in `config.json` under `ee_project`, then restart

## Settings (config.json)

The defaults are sensible. The ones you might change:

- `pre_days` / `post_days` (3 / 10) — search window around the spawn dates
- `max_cloud_pct` (70) — scenes cloudier than this over the site are dropped
- `thumb_px` (1120) — thumbnail resolution
- `download_chips` (true) — save a GeoTIFF when you label positive/negative

## Labeling

Click a record in the sidebar, then work through its scenes: `P` positive, `N` negative, `U` unsure, `X` unusable, `C` clear. Arrow keys switch scenes, `J`/`K` switch records, `F` toggles false color, `M` toggles off-season mode (Aug–Sep, for easy negatives). Labeling auto-advances.

Because of the cloud ceiling, some records have no usable scenes. "Hide empty" in the sidebar hides those (never ones you've labeled), and "Scan for scenes" checks the current filter's records in the background so the list thins out ahead of you. If a record's scenes were all dropped for cloud, the viewer says so and offers "Show cloudy scenes".

## Segment mode

Press `S`, then click a feature in the image — SAM 3 outlines it and a panel shows its area plus a spawn score. Shift+click adds exclude points to push the mask off land; `Esc` clears. The first click on a scene takes ~5–30 s to encode; later clicks are instant.

The score is currently a spectral heuristic (milt is bright in visible, dark in NIR), not a trained model — treat it as a hint. When a classifier is trained, drop it at `models/spawn_classifier.pt` (interface in `models/README.txt`) and it takes over automatically. SAM 3 weights are gated on Hugging Face; without an accepted license and `HF_TOKEN` the app falls back to SAM 2.1.

## Data

- `data/labels.csv` — one row per labeled scene, with record/scene ids, coordinates, dates, label, and chip path
- `data/chips/positive|negative/*.tif` — GeoTIFF chips: S2 at 10 m (B2,B3,B4,B8,B11,B12), Landsat at 30 m, TOA reflectance
- `data/scene_availability.json` — cache behind hide-empty/scan (regenerates itself)

## Roadmap

1. Label chips with this app — done, ongoing
2. Train a spawn/not-spawn classifier on the chips
3. Plug the classifier into segment mode (slot already wired)

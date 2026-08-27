# Herring Spawn Labeler

A local mini app for building a **training dataset of Pacific herring spawn events as seen from satellite**.

Herring spawn shows up as **milky turquoise water (milt) hugging the coastline**, typically visible for
about 1-5 days around the recorded spawn date. This app takes DFO's Pacific herring spawn index
(31k records, 1951-2025; the Sentinel-2 era subset from 2016 onward is used — 3,066 usable records),
finds satellite scenes (Sentinel-2, Landsat 8/9) around each confirmed spawn, and lets you rapidly label
image chips as **positive / negative / unsure / unusable**. Labels are saved to a CSV and, once Earth
Engine is connected, GeoTIFF chips are downloaded in the background for later model training.

## Quickstart

```
run.bat
```

or equivalently, from the project folder:

```
.venv\Scripts\python.exe -m app.main
```

Then open **http://127.0.0.1:8137** in your browser.

The app works immediately, even with no Earth Engine credentials — see *Mock mode* below.

## Earth Engine setup (to get real satellite imagery)

The app uses Google Earth Engine for scene search, thumbnails, and chip downloads. One-time setup:

1. **Register a (free, non-commercial) Earth Engine cloud project** at
   https://code.earthengine.google.com/register — sign in with your Google account, choose
   "Unpaid usage" / non-commercial, and create or pick a Google Cloud project. Note the project id
   (something like `ee-yourname`).
2. **Authenticate on this machine.** From the project folder run:

   ```
   .venv\Scripts\earthengine authenticate
   ```

   A browser window opens; approve access. Credentials are stored in your user profile.
3. **Tell the app which cloud project to use.** Edit `config.json` and set:

   ```json
   "ee_project": "ee-yourname"
   ```
4. **Restart the app.** The sidebar status strip should switch from the amber "MOCK MODE" banner to
   "Earth Engine ready".

## Mock mode

If Earth Engine initialization fails (no credentials yet, no project id, no network), the app does **not**
break — it runs in **mock mode**:

- `/api/status` reports `ee_mock: true` plus an `ee_error` message explaining how to fix it, and the UI
  shows an amber **MOCK MODE** banner.
- Scene lists are **fabricated** (6-10 plausible scenes per query window, deterministic per record, with
  fake cloud percentages) and thumbnails are locally generated placeholder images watermarked **MOCK**.
- The full labeling workflow still works end to end, so you can learn the UI and keyboard flow.

**Note:** labels saved in mock mode use fake scene ids (`MOCK_S2_20240109` etc.) that will *not* match
real Earth Engine scene ids after you authenticate. Treat mock-era labels as throwaway practice, and
clear `data/labels.csv` before starting real labeling.

## config.json reference

| Key | Default | Meaning |
| --- | --- | --- |
| `ee_project` | `""` | Google Cloud project id for Earth Engine. Empty = not configured (mock mode unless credentials alone suffice). |
| `pre_days` | `3` | Days before the record's `StartDate` included in the spawn-window scene search. |
| `post_days` | `10` | Days after the `EndDate` (or `StartDate` if no end date) included in the spawn-window search. |
| `max_scenes` | `16` | Cap on scenes returned per query. When more exist, the least-cloudy 16 are kept, then shown date-ascending. |
| `max_cloud_pct` | `70` | Scenes with regional cloud cover above this percentage are dropped from results (unknown cloud is kept). |
| `chip_min_half_m` | `600` | Minimum chip half-size in metres. Chip half-size is `clamp(0.35 * spawn_length + 300, min, max)`; 1200 m when the record has no length. |
| `chip_max_half_m` | `2500` | Maximum chip half-size in metres. |
| `download_chips` | `true` | Download a GeoTIFF chip in the background when a scene is labeled positive/negative (real EE mode only). |
| `segment_lru_scenes` | `3` | How many scenes' SAM image embeddings stay cached on the GPU in segment mode (~50 MB each). |
| `sensors` | `["s2","l8","l9"]` | Which sensors to search: Sentinel-2, Landsat 8, Landsat 9. |
| `port` | `8137` | HTTP port the app listens on. |

## Data layout

```
data/
  Pacific_herring_spawn_index_data_2025_EN.csv   # source spawn records (DFO)
  labels.csv                                     # your labels (the training dataset)
  chips/                                         # downloaded GeoTIFF chips, one folder per label
    positive/
    negative/
```

`labels.csv` columns (one row per labeled record+scene, upserted; deleting a label removes the row):

| Column | Meaning |
| --- | --- |
| `record_id` | Spawn record id: `{Year}_{LocationCode}_{SpawnNumber}` |
| `scene_id` | Satellite scene id (Earth Engine image id, or `MOCK_...` in mock mode) |
| `sensor` | `S2`, `L8`, or `L9` |
| `scene_date` | Scene acquisition date `YYYY-MM-DD` |
| `label` | `positive`, `negative`, `unsure`, or `unusable` |
| `notes` | Optional free-text note entered in the UI |
| `region`, `year`, `location_name`, `lon`, `lat`, `start_date` | Denormalized from the spawn record for convenience |
| `chip_path` | Path of the downloaded GeoTIFF chip, once the background download finishes |
| `labeled_at_utc` | UTC timestamp of the (last) label write |

## Keyboard shortcuts

| Key | Action |
| --- | --- |
| `P` | Label current scene **positive** |
| `N` | Label current scene **negative** |
| `U` | Label current scene **unsure** |
| `X` | Label current scene **unusable** |
| `C` | Clear (undo) the label on the current scene |
| `←` / `→` | Previous / next scene |
| `J` / `K` | Next / previous record (crosses page boundaries) |
| `F` | Toggle true-color / false-color view |
| `M` | Toggle Spawn window / Off-season mode |
| `S` | Toggle **Segment mode** (see below) |
| `Esc` | In segment mode: clear the current points + mask |

Labeling auto-advances to the next scene, and after the last scene, to the next record.
The **Off-season** mode queries Aug 1 - Sep 15 of the same year (herring do not spawn then), which is a
fast way to harvest guaranteed-negative examples of the same coastline.

## Segment mode (SAM)

Press **`S`** (or click the **Segment** button in the viewer bar) to enter segment mode. A status badge
appears next to the button: pulsing amber *"loading SAM…"* while the model loads, green *"SAM2 · cuda"*
when ready, red on error (hover for details).

- **Click** any feature in the image — SAM outlines it and a panel below the viewer shows its area,
  coverage, SAM IoU, timing, and a **spawn score**.
- **Shift+click** adds a *background* (exclude) point; all accumulated points refine the same mask, so
  use it to push the mask off land or out of a channel.
- **`Esc`** (or the Clear button) drops the points and mask. Switching scene or record clears too.

**Latency:** the first click on a new scene downloads + encodes the image (~5-30 s; slower if the thumb
is not yet cached under `data/scene_cache/`). Every later click on that scene reuses the cached
embedding and takes well under a second (~0.3-0.6 s measured on the RTX 3060). The last
`segment_lru_scenes` (default 3) scenes stay encoded.

**Model backends.** The engine tries **SAM 3** (`facebook/sam3`) first, then falls back to
**SAM 2.1 small** (`facebook/sam2.1-hiera-small`, bf16 on CUDA, ~1.6 GB VRAM in use). SAM 3 is a
*gated* model: to enable it, accept the license at https://huggingface.co/facebook/sam3, then set
`HF_TOKEN` (or run `hf auth login`) and restart the app — until then `/api/segment/status` reports the
SAM 2.1 fallback with exactly that hint. No CUDA GPU? SAM 2.1 runs on CPU (slower first click).

**Spawn scoring — heuristic vs trained.** No classifier has been trained yet (that needs the labels
this app collects), so the score is a **clearly-labeled spectral heuristic**, not a model: it compares
the mask's interior against a 15 px ring just outside it on three lifts — visible brightness, NIR
suppression (bright-visible-but-dark-NIR separates milt from whitecaps/cloud), and cyan-green
dominance — squashed through a hand-tuned sigmoid. The panel marks it with a `heuristic` pill and an
explanatory note; treat it as a rough hint only.

Once a trained model exists, drop it at **`models/spawn_classifier.pt`** and restart — the engine
auto-loads it and the pill switches to `model`. Interface (also documented in `models/README.txt`):
a `torch.load`-able callable taking a float32 `(1, 4, 128, 128)` tensor — a mask-cropped
`[R, G, B, NIR]` chip in 0-1 with out-of-mask pixels zeroed — and returning a single spawn logit.
A broken `.pt` file falls back to the heuristic instead of breaking the app.

## Roadmap

- **Phase 1 (this app):** label satellite chips around confirmed spawn records to build a supervised
  training dataset (`labels.csv` + `data/chips/`).
- **Phase 2:** train a CNN classifier on the labeled chips to score "spawn / not spawn" on unseen imagery.
- **Phase 3 (partially done):** click-to-segment is **live** — Segment mode (`S`) segments any clicked
  feature with SAM (2.1 today, 3 once the gated license is accepted) and scores it with the spectral
  heuristic. **Remaining:** train the phase-2 classifier and drop it in as
  `models/spawn_classifier.pt`, which replaces the heuristic automatically.

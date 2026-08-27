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
| `chip_min_half_m` | `600` | Minimum chip half-size in metres. Chip half-size is `clamp(0.35 * spawn_length + 300, min, max)`; 1200 m when the record has no length. |
| `chip_max_half_m` | `2500` | Maximum chip half-size in metres. |
| `download_chips` | `true` | Download a GeoTIFF chip in the background when a scene is labeled positive/negative (real EE mode only). |
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

Labeling auto-advances to the next scene, and after the last scene, to the next record.
The **Off-season** mode queries Aug 1 - Sep 15 of the same year (herring do not spawn then), which is a
fast way to harvest guaranteed-negative examples of the same coastline.

## Roadmap

- **Phase 1 (this app):** label satellite chips around confirmed spawn records to build a supervised
  training dataset (`labels.csv` + `data/chips/`).
- **Phase 2:** train a CNN classifier on the labeled chips to score "spawn / not spawn" on unseen imagery.
- **Phase 3:** a SAM 3 click-to-segment app — the user clicks a suspected feature on a satellite image,
  SAM 3 segments it, and the phase-2 classifier scores the segment as spawn / not-spawn.

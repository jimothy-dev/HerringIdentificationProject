"""One-click training-data export.

Zips labels.csv, every downloaded GeoTIFF chip, and a README describing the
format into data/exports/, so the user can hand the dataset to someone else
without any manual packaging.
"""

from __future__ import annotations

import csv
import datetime as dt
import re
import zipfile
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EXPORTS_DIR = DATA_DIR / "exports"
LABELS_CSV = DATA_DIR / "labels.csv"
CHIPS_DIR = DATA_DIR / "chips"

# Export filenames are server-generated; the download route only serves names
# matching this, so a crafted path can never escape data/exports/.
_NAME_RE = re.compile(r"^herring_training_data_[0-9_\-]+\.zip$")

_README = """Pacific Herring Spawn - satellite training chips
=================================================
Exported {stamp} by the Herring Spawn Labeler.
{n_labels} labeled scenes, {n_chips} GeoTIFF chips.

Files
-----
labels.csv    One row per labeled scene. Key columns:
              record_id, scene_id, sensor (S2|L8|L9), scene_date,
              label (positive|negative|unsure|unusable), lon, lat,
              start_date (spawn start), chip_path, location_name.
              Rows labeled positive/negative have a chip.
chips/positive/*.tif   Spawn visible (milky milt water)
chips/negative/*.tif   No spawn (same coastline types, incl. off-season)

Chip format
-----------
GeoTIFF, WGS84, named {{record_id}}__{{scene_id}}.tif
Sentinel-2 (S2):  bands B2,B3,B4,B8,B11,B12 at 10 m, TOA reflectance x10000 (int)
Landsat 8/9:      bands B2,B3,B4,B5,B6,B7  at 30 m, TOA reflectance (float 0-1)
Chip footprint is sized to the recorded spawn extent
(half-width = clamp(0.35*spawn_length_m + 300, 600, 2500) m).

Notes
-----
- Milt signature: bright in visible (esp. green/blue), dark in NIR.
- TOA (not surface reflectance) on purpose: SR products are unreliable
  over dark coastal water. Normalize per-chip for training.
"""


def build_export() -> dict[str, Any]:
    """Build a dated zip in data/exports; returns filename and counts."""
    if not LABELS_CSV.exists():
        raise RuntimeError("No labels yet - label some scenes first.")

    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        n_labels = sum(1 for _ in csv.DictReader(f))
    if n_labels == 0:
        raise RuntimeError("No labels yet - label some scenes first.")

    chips = sorted(CHIPS_DIR.rglob("*.tif")) if CHIPS_DIR.exists() else []

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = f"herring_training_data_{stamp}.zip"
    out_path = EXPORTS_DIR / filename

    readme = _README.format(
        stamp=dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        n_labels=n_labels,
        n_chips=len(chips),
    )
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(LABELS_CSV, "labels.csv")
        z.writestr("README_dataset.txt", readme)
        for chip in chips:
            z.write(chip, str(chip.relative_to(DATA_DIR)))

    return {
        "filename": filename,
        "n_labels": n_labels,
        "n_chips": len(chips),
        "size_bytes": out_path.stat().st_size,
    }


def safe_export_path(filename: str) -> Path | None:
    """Resolve a download filename to a real export file, or None."""
    if not _NAME_RE.fullmatch(filename):
        return None
    p = EXPORTS_DIR / filename
    return p if p.is_file() else None

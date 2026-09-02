"""All Earth Engine logic: init, config, scene search, thumbnails, chips, mock mode.

Design rules honoured here:
- Per-scene metadata (id / date / regional cloud) is computed server-side by
  mapping the joined collections into FeatureCollections, so there is exactly
  ONE .getInfo() round trip per sensor family (one for S2, one for L8+L9).
- Thumbnails are only requested AFTER dedupe/cap, in a ThreadPoolExecutor
  (each getThumbURL is its own network round trip).
- Every GEE call is wrapped so a failure degrades to an error string or mock
  mode instead of a 500.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

log = logging.getLogger("herring.gee")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.json"
CHIPS_DIR = PROJECT_ROOT / "data" / "chips"

DEFAULT_CONFIG: dict[str, Any] = {
    "ee_project": "",
    "pre_days": 3,
    "post_days": 10,
    "max_scenes": 16,
    "max_cloud_pct": 70,
    "chip_min_half_m": 600,
    "chip_max_half_m": 2500,
    "download_chips": True,
    "sensors": ["s2", "l8", "l9"],
    "thumb_px": 1120,
    "segment_lru_scenes": 3,
    "port": 8137,
}

# Asset prefixes used to rebuild an ee.Image from a stored scene index.
_ASSET_PREFIX = {
    "S2": "COPERNICUS/S2_HARMONIZED",  # TOA -- deliberately NOT the SR collection
    "L8": "LANDSAT/LC08/C02/T1_TOA",
    "L9": "LANDSAT/LC09/C02/T1_TOA",
}

_CHIP_BANDS = {
    "S2": ["B2", "B3", "B4", "B8", "B11", "B12"],
    "L8": ["B2", "B3", "B4", "B5", "B6", "B7"],
    "L9": ["B2", "B3", "B4", "B5", "B6", "B7"],
}

_CHIP_SCALE = {"S2": 10, "L8": 30, "L9": 30}

# Visualization parameters for thumbnails (true / false color per sensor).
_VIS = {
    ("S2", "true"): {"bands": ["B4", "B3", "B2"], "min": 0, "max": 2600, "gamma": 1.1},
    ("S2", "false"): {"bands": ["B8", "B4", "B3"], "min": 0, "max": 3600},
    ("L8", "true"): {"bands": ["B4", "B3", "B2"], "min": 0, "max": 0.26},
    ("L8", "false"): {"bands": ["B5", "B4", "B3"], "min": 0, "max": 0.36},
    ("L9", "true"): {"bands": ["B4", "B3", "B2"], "min": 0, "max": 0.26},
    ("L9", "false"): {"bands": ["B5", "B4", "B3"], "min": 0, "max": 0.36},
}


class GeeError(Exception):
    """Raised when an Earth Engine operation fails; message is user-facing."""


# --------------------------------------------------------------------- config

_config: dict[str, Any] | None = None


def get_config() -> dict[str, Any]:
    """Load config.json (creating it from defaults if absent), merged over defaults."""
    global _config
    if _config is not None:
        return _config
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                cfg.update(loaded)
            else:
                log.warning(
                    "config.json is not a JSON object (got %s); using defaults",
                    type(loaded).__name__,
                )
        # ValueError covers both JSONDecodeError and UnicodeDecodeError (e.g. a
        # UTF-16 file written by PowerShell redirection) -- the app must always
        # come up, worst case with defaults.
        except (OSError, ValueError) as exc:
            log.warning("config.json unreadable (%s); using defaults", exc)
    else:
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
        except OSError as exc:
            log.warning("could not write default config.json: %s", exc)
    _config = cfg
    return cfg


# ------------------------------------------------------------------ EE state

class _EEState:
    def __init__(self) -> None:
        self.ready: bool = False
        self.mock: bool = True
        self.error: str | None = "Earth Engine not initialized yet."
        self.project: str | None = None


_state = _EEState()


def state() -> _EEState:
    return _state


def init_ee() -> None:
    """Try to initialize Earth Engine; fall back to mock mode on any failure."""
    cfg = get_config()
    project = (cfg.get("ee_project") or "").strip() or None
    _state.project = project
    try:
        import ee

        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        # Force a real round trip so bad/expired credentials fail here, not later.
        ee.Number(1).getInfo()
        _state.ready = True
        _state.mock = False
        _state.error = None
        log.info("Earth Engine initialized (project=%s)", project or "<default>")
    except Exception as exc:  # noqa: BLE001 - any EE failure means mock mode
        _state.ready = False
        _state.mock = True
        _state.error = (
            f"Earth Engine init failed: {exc}. Running in MOCK mode with fabricated "
            "scenes. To fix: run '.venv\\Scripts\\earthengine authenticate' in the "
            "project folder, put your Google Cloud project id in \"ee_project\" in "
            "config.json, then restart the app."
        )
        log.warning("EE init failed, running in mock mode: %s", exc)


# ----------------------------------------------------------------- geometry

def chip_half_size_m(length_m: float | None) -> float:
    """Half-size of the square chip around the spawn point, in metres."""
    cfg = get_config()
    if length_m is None:
        return 1200.0
    half = 0.35 * length_m + 300.0
    return max(float(cfg["chip_min_half_m"]), min(float(cfg["chip_max_half_m"]), half))


def _chip_region(lon: float, lat: float, length_m: float | None):
    import ee

    half = chip_half_size_m(length_m)
    return ee.Geometry.Point([lon, lat]).buffer(half).bounds()


# ------------------------------------------------------------- scene search

def effective_max_cloud(override: float | None = None) -> float:
    """The regional-cloud ceiling to apply: the override when given, else config."""
    if override is not None:
        return float(override)
    return float(get_config().get("max_cloud_pct", 70))


def search_scenes(
    lon: float,
    lat: float,
    length_m: float | None,
    window_start: dt.date,
    window_end: dt.date,
    max_cloud_override: float | None = None,
    with_thumbs: bool = True,
    sensors_override: list[str] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Return (scenes, n_cloud_filtered) for the window.

    Scenes are dicts (scene_id, sensor, date, cloud_region_pct, and thumb urls
    when with_thumbs), sorted by date asc, one per (sensor, date), capped at
    max_scenes. n_cloud_filtered counts DEDUPED scenes dropped by the cloud
    ceiling (config max_cloud_pct, or max_cloud_override for this call only),
    i.e. distinct scene slots the user would otherwise have seen.
    Raises GeeError with a user-facing message on failure.
    """
    cfg = get_config()
    source = sensors_override if sensors_override else cfg.get("sensors", ["s2", "l8", "l9"])
    sensors = [s.lower() for s in source]
    try:
        raw = _fetch_scene_metadata(lon, lat, length_m, window_start, window_end, sensors)
    except GeeError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise GeeError(f"Scene search failed: {exc}") from exc

    # Dedupe FIRST so n_cloud_filtered counts distinct scene slots, then drop
    # scenes too cloudy over the spawn site to be labelable; scenes whose
    # regional cloud could not be computed (None) are kept — unknown != bad.
    # Dedupe is ceiling-aware: within a (sensor, date) slot a scene that
    # passes the ceiling always survives over one that fails it, so a known
    # over-ceiling duplicate can never knock out a labelable scene. A slot is
    # counted in n_cloud_filtered exactly when NO member passes the ceiling.
    max_cloud = effective_max_cloud(max_cloud_override)
    deduped = _dedupe(raw, max_cloud)
    kept = [s for s in deduped if _passes_ceiling(s, max_cloud)]
    n_cloud_filtered = len(deduped) - len(kept)
    scenes = _cap_and_sort(kept, int(cfg["max_scenes"]))
    if with_thumbs:
        _attach_thumbs(scenes, lon, lat, length_m)
    return scenes, n_cloud_filtered


def count_usable_scenes(
    lon: float,
    lat: float,
    length_m: float | None,
    window_start: dt.date,
    window_end: dt.date,
) -> int:
    """Metadata-only availability probe: usable-scene count at the config ceiling.

    Skips thumbnail generation entirely (the expensive part of search_scenes),
    which makes this the path the background availability sweep uses.
    """
    scenes, _n_cloud_filtered = search_scenes(
        lon, lat, length_m, window_start, window_end, with_thumbs=False
    )
    return len(scenes)


def _fetch_scene_metadata(
    lon: float,
    lat: float,
    length_m: float | None,
    window_start: dt.date,
    window_end: dt.date,
    sensors: list[str],
) -> list[dict[str, Any]]:
    import ee

    point = ee.Geometry.Point([lon, lat])
    region = _chip_region(lon, lat, length_m)
    start = window_start.isoformat()
    end_excl = (window_end + dt.timedelta(days=1)).isoformat()

    out: list[dict[str, Any]] = []
    if "s2" in sensors:
        out.extend(_fetch_s2(ee, point, region, start, end_excl))
    landsat = [s for s in sensors if s in ("l8", "l9")]
    if landsat:
        out.extend(_fetch_landsat(ee, point, region, start, end_excl, landsat))
    return out


def _fetch_s2(ee, point, region, start: str, end_excl: str) -> list[dict[str, Any]]:
    """S2 TOA joined to S2_CLOUD_PROBABILITY; ONE getInfo for all scenes."""
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterBounds(point)
        .filterDate(start, end_excl)
    )
    prob = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterBounds(point)
        .filterDate(start, end_excl)
    )
    joined = ee.ImageCollection(
        ee.Join.saveFirst("cloud_prob").apply(
            primary=s2,
            secondary=prob,
            condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
        )
    )

    def to_feature(img):
        img = ee.Image(img)
        cloud = (
            ee.Image(img.get("cloud_prob"))
            .select("probability")
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=120,
                bestEffort=True,
                maxPixels=10_000_000,
            )
            .get("probability")
        )
        return ee.Feature(
            None,
            {
                "index": img.get("system:index"),
                "time_start": img.get("system:time_start"),
                "cloud": cloud,  # already 0-100
            },
        )

    feats = ee.FeatureCollection(joined.map(to_feature)).getInfo()
    return _features_to_scenes(feats, sensor="S2", cloud_scale=1.0)


def _fetch_landsat(
    ee, point, region, start: str, end_excl: str, landsat: list[str]
) -> list[dict[str, Any]]:
    """L8 + L9 TOA merged into one FeatureCollection; ONE getInfo for the family."""

    def sensor_fc(collection_id: str, sensor: str):
        col = (
            ee.ImageCollection(collection_id)
            .filterBounds(point)
            .filterDate(start, end_excl)
        )

        def to_feature(img):
            img = ee.Image(img)
            qa = img.select("QA_PIXEL")
            # QA_PIXEL bit 3 = cloud, bit 4 = cloud shadow.
            cloudy = (
                qa.rightShift(3).bitwiseAnd(1).Or(qa.rightShift(4).bitwiseAnd(1))
            ).rename("cloudy")
            frac = cloudy.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=90,
                bestEffort=True,
                maxPixels=10_000_000,
            ).get("cloudy")
            return ee.Feature(
                None,
                {
                    "index": img.get("system:index"),
                    "time_start": img.get("system:time_start"),
                    "cloud": frac,  # fraction 0-1, scaled to % client-side
                    "sensor": sensor,
                },
            )

        return ee.FeatureCollection(col.map(to_feature))

    fc = None
    if "l8" in landsat:
        fc = sensor_fc("LANDSAT/LC08/C02/T1_TOA", "L8")
    if "l9" in landsat:
        l9 = sensor_fc("LANDSAT/LC09/C02/T1_TOA", "L9")
        fc = l9 if fc is None else fc.merge(l9)

    feats = fc.getInfo()
    return _features_to_scenes(feats, sensor=None, cloud_scale=100.0)


def _features_to_scenes(
    feats: dict, sensor: str | None, cloud_scale: float
) -> list[dict[str, Any]]:
    scenes: list[dict[str, Any]] = []
    for feat in feats.get("features", []):
        props = feat.get("properties", {})
        index = props.get("index")
        time_start = props.get("time_start")
        if not index or time_start is None:
            continue
        date = dt.datetime.fromtimestamp(time_start / 1000.0, tz=dt.timezone.utc).date()
        cloud = props.get("cloud")
        if cloud is not None:
            cloud = round(float(cloud) * cloud_scale, 1)
        scenes.append(
            {
                "scene_id": str(index),
                "sensor": sensor or str(props.get("sensor", "L8")),
                "date": date.isoformat(),
                "cloud_region_pct": cloud,
            }
        )
    return scenes


def _passes_ceiling(scene: dict[str, Any], max_cloud: float) -> bool:
    """Whether a scene survives the regional-cloud ceiling (None = unknown passes)."""
    cloud = scene.get("cloud_region_pct")
    return cloud is None or float(cloud) <= float(max_cloud)


def _dedupe(scenes: list[dict[str, Any]], max_cloud: float) -> list[dict[str, Any]]:
    """One scene per (sensor, date): prefer scenes that pass the cloud ceiling
    (unknown cloud counts as passing), then the lowest regional cloud.

    Ceiling-awareness matters: with a plain lowest-cloud rule, a known 80%
    duplicate would beat an unknown-cloud scene (999 sort key) and the whole
    slot would then be dropped by the ceiling — hiding a labelable scene.
    """
    best: dict[tuple[str, str], dict[str, Any]] = {}

    def rank(sc: dict[str, Any]) -> tuple[bool, float]:
        return (not _passes_ceiling(sc, max_cloud), _cloud_sort_key(sc))

    for sc in scenes:
        key = (sc["sensor"], sc["date"])
        prev = best.get(key)
        if prev is None or rank(sc) < rank(prev):
            best[key] = sc
    return list(best.values())


def _cap_and_sort(scenes: list[dict[str, Any]], max_scenes: int) -> list[dict[str, Any]]:
    """Cap at max_scenes (keeping the least cloudy), then sort by date asc."""
    result = list(scenes)
    if len(result) > max_scenes:
        # Keep the max_scenes least-cloudy scenes across all sensors...
        result.sort(key=_cloud_sort_key)
        result = result[:max_scenes]
    # ...and always present them sorted by date asc.
    result.sort(key=lambda sc: (sc["date"], sc["sensor"]))
    return result


def _cloud_sort_key(scene: dict[str, Any]) -> float:
    cloud = scene.get("cloud_region_pct")
    return 999.0 if cloud is None else float(cloud)


# ------------------------------------------------------------------- thumbs

def _attach_thumbs(
    scenes: list[dict[str, Any]], lon: float, lat: float, length_m: float | None
) -> None:
    """Fill thumb_true / thumb_false for each scene, in parallel (post-dedupe only)."""
    if not scenes:
        return

    def one(scene: dict[str, Any]) -> None:
        try:
            scene["thumb_true"] = _thumb_url(scene["sensor"], scene["scene_id"], "true", lon, lat, length_m)
            scene["thumb_false"] = _thumb_url(scene["sensor"], scene["scene_id"], "false", lon, lat, length_m)
        except Exception as exc:  # noqa: BLE001 - a broken thumb must not kill the list
            log.warning("thumbnail failed for %s: %s", scene["scene_id"], exc)
            scene["thumb_true"] = scene.get("thumb_true") or ""
            scene["thumb_false"] = ""

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(one, scenes))


def _thumb_url(
    sensor: str, scene_index: str, kind: str, lon: float, lat: float, length_m: float | None
) -> str:
    import ee

    img = ee.Image(f"{_ASSET_PREFIX[sensor]}/{scene_index}")
    vis = dict(_VIS[(sensor, kind)])
    params: dict[str, Any] = {
        "dimensions": int(get_config().get("thumb_px", 1120)),
        "region": _chip_region(lon, lat, length_m),
        "format": "png",
        **vis,
    }
    return img.getThumbURL(params)


# -------------------------------------------------------------------- chips

def download_chip(
    record_id: str,
    scene_id: str,
    sensor: str,
    label: str,
    lon: float,
    lat: float,
    length_m: float | None,
) -> str | None:
    """Download a multiband GeoTIFF chip; returns the saved path or None.

    Runs in a background thread -- must never raise.
    """
    try:
        import ee
        import requests

        if _state.mock or not _state.ready:
            return None
        img = ee.Image(f"{_ASSET_PREFIX[sensor]}/{scene_id}").select(_CHIP_BANDS[sensor])
        url = img.getDownloadURL(
            {
                "scale": _CHIP_SCALE[sensor],
                "region": _chip_region(lon, lat, length_m),
                "format": "GEO_TIFF",
            }
        )
        resp = requests.get(url, timeout=600)
        resp.raise_for_status()

        safe_scene = "".join(c if (c.isalnum() or c in "_-") else "_" for c in scene_id)
        out_dir = CHIPS_DIR / label
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{record_id}__{safe_scene}.tif"
        tmp_path = out_path.with_suffix(".tif.part")
        with open(tmp_path, "wb") as f:
            f.write(resp.content)
        tmp_path.replace(out_path)
        log.info("chip saved: %s (%d bytes)", out_path, len(resp.content))
        return str(out_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("chip download failed for %s / %s: %s", record_id, scene_id, exc)
        return None


# ---------------------------------------------------------------- mock mode

def mock_scenes(
    record_id: str,
    mode: str,
    window_start: dt.date,
    window_end: dt.date,
    max_cloud_override: float | None = None,
    sensors_override: list[str] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Deterministic fabricated scenes so the UI is fully usable before EE auth.

    Returns (scenes, n_cloud_filtered), the same shape as search_scenes, with
    the same cloud ceiling / override applied.

    6-10 scenes spread across the window (every ~3-5 days for typical windows),
    alternating sensors, seeded from the record id so results are stable across
    restarts (which keeps saved mock labels attached to the same scene ids).
    """
    cfg = get_config()
    name_map = {"s2": "S2", "l8": "L8", "l9": "L9"}
    sensor_source = sensors_override if sensors_override else cfg.get("sensors", ["s2", "l8", "l9"])
    sensor_cycle = [
        name_map[s.lower()] for s in sensor_source if s.lower() in name_map
    ] or ["S2"]

    seed_int = int(hashlib.md5(f"{record_id}:{mode}".encode()).hexdigest()[:12], 16)
    rng = random.Random(seed_int)

    span_days = max(0, (window_end - window_start).days)
    n_want = min(rng.randint(6, 10), int(cfg["max_scenes"]), span_days + 1)
    n_want = max(1, n_want)
    offsets = sorted(rng.sample(range(span_days + 1), n_want))

    scenes: list[dict[str, Any]] = []
    for i, off in enumerate(offsets):
        date = window_start + dt.timedelta(days=off)
        sensor = sensor_cycle[i % len(sensor_cycle)]
        scene_id = f"MOCK_{sensor}_{date.strftime('%Y%m%d')}"
        scenes.append(
            {
                "scene_id": scene_id,
                "sensor": sensor,
                "date": date.isoformat(),
                "cloud_region_pct": round(rng.uniform(5.0, 95.0), 1),
                "thumb_true": f"/api/mock_thumb/{mock_thumb_seed(record_id, scene_id, 'true')}.png",
                "thumb_false": f"/api/mock_thumb/{mock_thumb_seed(record_id, scene_id, 'false')}.png",
            }
        )
    # Same cloud ceiling (and override) as real scenes, so mock mode previews
    # the filter too. The fabricated list is already one scene per (sensor,
    # date), i.e. deduped, so the filtered count matches distinct scene slots.
    max_cloud = effective_max_cloud(max_cloud_override)
    kept = [s for s in scenes if s["cloud_region_pct"] <= max_cloud]
    return kept, len(scenes) - len(kept)


def mock_thumb_seed(record_id: str, scene_id: str, kind: str) -> str:
    return hashlib.md5(f"{record_id}|{scene_id}|{kind}".encode()).hexdigest()[:12]

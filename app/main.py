"""FastAPI app for the herring spawn labeling tool.

Implements exactly the API contract shared with the frontend. Works fully in
mock mode when Earth Engine credentials are not set up yet.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import logging
import random
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import gee, segment
from app.labels import VALID_LABELS, LabelStore
from app.records import RecordStore, SpawnRecord

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("herring.main")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "app" / "static"

# Module-level singletons (config, data, labels, EE state, chip worker pool).
CFG = gee.get_config()
records_store = RecordStore()
label_store = LabelStore()
gee.init_ee()
_chip_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="chip")

app = FastAPI(title="Herring Spawn Labeler")


# ------------------------------------------------------------------- status

@app.get("/api/status")
def api_status() -> dict:
    st = gee.state()
    counts = label_store.counts()
    return {
        "ee_ready": st.ready,
        "ee_mock": st.mock,
        "ee_error": st.error,
        "ee_project": st.project,
        "n_records": len(records_store.records),
        "n_labeled": counts["n_labeled"],
        "n_positive": counts["n_positive"],
        "n_negative": counts["n_negative"],
    }


# ------------------------------------------------------------------ records

@app.get("/api/records")
def api_records(
    year: int | None = Query(default=None),
    region: str | None = Query(default=None),
    labeled: Literal["all", "unlabeled", "labeled"] = Query(default="all"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
) -> dict:
    counts_by_record = label_store.record_label_counts()
    total, page_rows = records_store.query(
        year=year,
        region=region,
        labeled=labeled,
        page=page,
        page_size=page_size,
        n_labels_fn=lambda rid: counts_by_record.get(rid, 0),
    )
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "years": records_store.years,
        "regions": records_store.regions,
        "records": [rec.to_api(n) for rec, n in page_rows],
    }


# ------------------------------------------------------------------- scenes

def _scene_window(record: SpawnRecord, mode: str) -> tuple[dt.date, dt.date]:
    if mode == "offseason":
        return dt.date(record.year, 8, 1), dt.date(record.year, 9, 15)
    pre = int(CFG["pre_days"])
    post = int(CFG["post_days"])
    end_anchor = record.end_date or record.start_date
    return (
        record.start_date - dt.timedelta(days=pre),
        end_anchor + dt.timedelta(days=post),
    )


@app.get("/api/records/{record_id}/scenes")
def api_scenes(
    record_id: str,
    mode: Literal["spawn", "offseason"] = Query(default="spawn"),
) -> dict:
    record = records_store.get(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown record id: {record_id}")

    window_start, window_end = _scene_window(record, mode)
    response: dict = {
        "record_id": record_id,
        "mode": mode,
        "window": {"start": window_start.isoformat(), "end": window_end.isoformat()},
        "scenes": [],
        "error": None,
    }

    st = gee.state()
    if st.mock:
        scenes = gee.mock_scenes(record_id, mode, window_start, window_end)
    else:
        try:
            scenes = gee.search_scenes(
                record.lon, record.lat, record.length_m, window_start, window_end
            )
        except gee.GeeError as exc:
            response["error"] = str(exc)
            return response

    for sc in scenes:
        if mode == "spawn":
            scene_date = dt.date.fromisoformat(sc["date"])
            sc["days_from_start"] = (scene_date - record.start_date).days
        else:
            sc["days_from_start"] = None
        sc["label"] = label_store.label_for(record_id, sc["scene_id"])

    response["scenes"] = scenes
    return response


# ------------------------------------------------------------------- labels

class LabelIn(BaseModel):
    record_id: str
    scene_id: str
    sensor: str
    scene_date: str
    label: Literal["positive", "negative", "unsure", "unusable"]
    # None (field absent) = keep any previously saved notes; "" = clear them.
    notes: str | None = Field(default=None)


def _queue_chip(record: SpawnRecord, body: LabelIn) -> None:
    def job() -> None:
        path = gee.download_chip(
            record_id=body.record_id,
            scene_id=body.scene_id,
            sensor=body.sensor,
            label=body.label,
            lon=record.lon,
            lat=record.lat,
            length_m=record.length_m,
        )
        if path:
            label_store.set_chip_path(body.record_id, body.scene_id, path)

    _chip_pool.submit(job)


def _sync_chip_location(record_id: str, scene_id: str, chip_path: str, new_label: str) -> str:
    """Move an existing chip into data/chips/{new_label}/ if it lives elsewhere.

    Keeps the per-label chip layout truthful when a label is corrected, so a
    later training set assembled from data/chips/{label}/ never picks up a
    chip whose label has changed. Returns the (possibly new) chip path.
    """
    src = Path(chip_path)
    if src.parent.name == new_label:
        return chip_path
    dest_dir = gee.CHIPS_DIR / new_label
    dest = dest_dir / src.name
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        src.replace(dest)
    except OSError as exc:
        log.warning("could not move chip %s -> %s: %s", src, dest, exc)
        return chip_path
    label_store.set_chip_path(record_id, scene_id, str(dest))
    log.info("chip moved for relabel: %s -> %s", src, dest)
    return str(dest)


@app.post("/api/labels")
def api_save_label(body: LabelIn) -> dict:
    record = records_store.get(body.record_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown record id: {body.record_id}")
    if body.label not in VALID_LABELS:
        raise HTTPException(status_code=422, detail=f"Invalid label: {body.label}")

    existing = label_store.get(body.record_id, body.scene_id)
    notes = body.notes
    if notes is None:  # field absent -> preserve saved notes (like chip_path)
        notes = existing.get("notes", "") if existing else ""
    label_store.upsert(
        {
            "record_id": body.record_id,
            "scene_id": body.scene_id,
            "sensor": body.sensor,
            "scene_date": body.scene_date,
            "label": body.label,
            "notes": notes,
            "region": record.region,
            "year": record.year,
            "location_name": record.location_name,
            "lon": record.lon,
            "lat": record.lat,
            "start_date": record.start_date.isoformat(),
        }
    )

    # If a chip was already downloaded under a different label's folder, move it
    # so data/chips/{label}/ always reflects the current label.
    already = existing.get("chip_path") if existing else ""
    if already and Path(already).exists():
        already = _sync_chip_location(body.record_id, body.scene_id, already, body.label)

    chip = "skipped"
    if body.label in ("positive", "negative"):
        if already and Path(already).exists():
            chip = "exists"
        elif gee.state().mock or not CFG.get("download_chips", True):
            chip = "skipped"
        else:
            _queue_chip(record, body)
            chip = "queued"
    return {"ok": True, "chip": chip}


@app.delete("/api/labels")
def api_delete_label(record_id: str = Query(...), scene_id: str = Query(...)) -> dict:
    existing = label_store.get(record_id, scene_id)
    label_store.delete(record_id, scene_id)
    # Remove the chip belonging to the deleted label row so no orphaned GeoTIFF
    # lingers in a data/chips/{label}/ folder (labels.csv is the manifest).
    chip_path = (existing or {}).get("chip_path") or ""
    if chip_path:
        try:
            Path(chip_path).unlink(missing_ok=True)
        except OSError as exc:
            log.warning("could not delete chip %s: %s", chip_path, exc)
    return {"ok": True}


@app.get("/api/labels")
def api_get_labels(record_id: str | None = Query(default=None)) -> dict:
    return {"labels": label_store.all_rows(record_id)}


# ------------------------------------------------------------------ segment

class SegmentPoint(BaseModel):
    x: float  # normalized 0-1 in true-color thumb image space
    y: float
    label: Literal[0, 1]  # 1 = foreground, 0 = background/exclude


class SegmentIn(BaseModel):
    record_id: str
    scene_id: str
    sensor: str
    points: list[SegmentPoint]


@app.get("/api/segment/status")
def api_segment_status() -> dict:
    return segment.engine().status()


@app.post("/api/segment/warmup")
def api_segment_warmup() -> dict:
    return segment.engine().warmup()


@app.post("/api/segment")
def api_segment(body: SegmentIn) -> dict:
    # Contract: HTTP 200 always; failures are {ok: false, state, error?, hint?}.
    record = records_store.get(body.record_id)
    if record is None:
        # Contract: failure states are "loading"|"cold"|"error" only -- a warm
        # engine's "ready" must not leak into an ok:false payload.
        engine_state = segment.engine().status()["state"]
        if engine_state not in ("cold", "loading"):
            engine_state = "error"
        return {
            "ok": False,
            "state": engine_state,
            "error": f"Unknown record id: {body.record_id}",
            "hint": None,
        }
    return segment.engine().segment(
        record=record,
        scene_id=body.scene_id,
        sensor=body.sensor,
        points=[(p.x, p.y, p.label) for p in body.points],
    )


# --------------------------------------------------------------- mock thumb

_SEED_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@lru_cache(maxsize=512)
def _mock_thumb_png(seed: str) -> bytes:
    """A 560px muted blue-grey coastal gradient with a MOCK watermark."""
    from PIL import Image, ImageDraw, ImageFont

    size = 560
    rng = random.Random(int(hashlib.md5(seed.encode()).hexdigest()[:12], 16))

    # Vertical sea gradient, slightly different per seed.
    top = (86 + rng.randint(-12, 12), 108 + rng.randint(-12, 12), 124 + rng.randint(-10, 14))
    bottom = (44 + rng.randint(-8, 8), 62 + rng.randint(-8, 8), 78 + rng.randint(-8, 10))
    img = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(img)
    for y in range(size):
        t = y / (size - 1)
        color = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
        draw.line([(0, y), (size, y)], fill=color)

    # A rough "coastline": land wedge along one edge.
    land = (96 + rng.randint(-8, 8), 99 + rng.randint(-6, 10), 88 + rng.randint(-8, 8))
    edge_x = size - rng.randint(90, 200)
    pts = [(size, 0)]
    x = edge_x
    for y in range(0, size + 1, 40):
        x = min(size - 20, max(edge_x - 60, x + rng.randint(-35, 35)))
        pts.append((x, y))
    pts.append((size, size))
    draw.polygon(pts, fill=land)

    # Faint turquoise blotch near the coast on some seeds (fake "milt").
    if rng.random() < 0.5:
        overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)
        cx = edge_x - rng.randint(30, 120)
        cy = rng.randint(120, 440)
        rx, ry = rng.randint(60, 140), rng.randint(30, 80)
        odraw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(120, 200, 205, 70))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(img)

    # Watermark + seed text.
    overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    try:
        big = ImageFont.load_default(size=96)
        small = ImageFont.load_default(size=18)
    except TypeError:  # very old Pillow fallback
        big = small = ImageFont.load_default()
    odraw.text((size // 2, size // 2), "MOCK", font=big, anchor="mm", fill=(255, 255, 255, 46))
    odraw.text((14, size - 30), seed, font=small, fill=(235, 240, 244, 150))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@app.get("/api/mock_thumb/{seed}.png")
def api_mock_thumb(seed: str) -> Response:
    if not _SEED_RE.match(seed):
        raise HTTPException(status_code=404, detail="bad seed")
    return Response(content=_mock_thumb_png(seed), media_type="image/png")


# ------------------------------------------------------------------- static

@app.get("/", response_class=HTMLResponse)
def index():
    index_html = STATIC_DIR / "index.html"
    if index_html.exists():
        return FileResponse(index_html)
    return HTMLResponse(
        "<h1>Herring Spawn Labeler</h1><p>Frontend not built yet "
        "(app/static/index.html missing). The API is live at /api/status.</p>"
    )


STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=int(CFG["port"]), reload=False)

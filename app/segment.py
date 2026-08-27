"""Segment mode: SAM point-prompt segmentation + spawn-likelihood scoring.

Backend order: SAM 3 tracker (facebook/sam3, gated) -> SAM 2.1 small
(facebook/sam2.1-hiera-small, ungated apache-2.0) -> error state. bfloat16 on
CUDA (Ampere-safe), float32 on CPU; SAM 3 is skipped on CPU because its
encoder takes tens of seconds per scene there.

Design rules honoured here:
- torch/transformers are imported lazily inside the loader thread so that
  importing app.segment (and starting the server) stays fast.
- ONE model instance and ONE embedding LRU, guarded by a single inference
  lock: a shared 6GB GPU cannot serve concurrent SAM calls safely.
- Scene thumbs are fetched server-side once and cached under
  data/scene_cache/ so repeat clicks and the spectral scorer reuse them.
- Encoder outputs are LRU-cached per (record_id, scene_id) -- config key
  "segment_lru_scenes", default 3 -- so clicks after the first take ~10-40ms
  instead of re-running the image encoder.
- Scoring is a clearly-labeled spectral heuristic until
  models/spawn_classifier.pt exists (interface documented in
  models/README.txt); a broken classifier file falls back to the heuristic.
"""

from __future__ import annotations

import base64
import gc
import io
import logging
import math
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

from app import gee
from app.records import SpawnRecord

log = logging.getLogger("herring.segment")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENE_CACHE_DIR = PROJECT_ROOT / "data" / "scene_cache"
MODELS_DIR = PROJECT_ROOT / "models"
CLASSIFIER_PATH = MODELS_DIR / "spawn_classifier.pt"

SAM3_REPO = "facebook/sam3"
SAM2_REPO = "facebook/sam2.1-hiera-small"

GATED_HINT = (
    "SAM 3 is gated: accept the license at https://huggingface.co/facebook/sam3, "
    "then set HF_TOKEN (or run 'hf auth login') and restart the app. "
    "Using the SAM 2.1 fallback until then."
)
COLD_HINT = (
    "Model not loaded yet -- POST /api/segment/warmup to load it "
    "(the first run downloads weights)."
)
LOADING_HINT = "Model is loading in the background; poll /api/segment/status."

# Overlay styling per the API contract.
_MASK_RGB = (0, 229, 255)
_FILL_ALPHA = 110
_EDGE_ALPHA = 255
_EDGE_PX = 2

# Spectral heuristic: comparison-ring width (px) and sigmoid weights/bias.
# Hand-tuned on 0-1 thumb values, NOT trained -- see _heuristic_score.
_RING_PX = 15
_W_BRIGHT = 4.0   # visible brightness lift of mask vs ring
_W_NIRSUP = 10.0  # brightness lift NOT mirrored in NIR (milt, not whitecaps)
_W_CYAN = 12.0    # cyan-green dominance lift (milky turquoise water)
_BIAS = -1.5      # prior: most clicked features are not spawn


# ------------------------------------------------------------ small helpers

def _safe_name(part: str) -> str:
    return "".join(c if (c.isalnum() or c in "_-") else "_" for c in part)


def _erode(mask: np.ndarray) -> np.ndarray:
    """3x3 cross erosion; outside the image counts as background."""
    p = np.pad(mask, 1, constant_values=False)
    return p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]


def _dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    m = mask
    for _ in range(iterations):
        p = np.pad(m, 1, constant_values=False)
        m = p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return m


def _rgb_array(img: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """Image as float32 HxWx3 in 0-1, resized to `size` (W, H) if needed."""
    img = img.convert("RGB")
    if img.size != size:
        img = img.resize(size, Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def _mask_data_uri(mask: np.ndarray) -> str:
    """Boolean mask -> RGBA overlay data URI per the API contract.

    Cyan fill at alpha 110, ~2px boundary at alpha 255, transparent elsewhere;
    dimensions exactly match the mask (== the true-color thumb).
    """
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask] = (*_MASK_RGB, _FILL_ALPHA)
    inner = mask
    for _ in range(_EDGE_PX):
        inner = _erode(inner)
    rgba[mask & ~inner] = (*_MASK_RGB, _EDGE_ALPHA)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _looks_gated(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(tok in text for tok in ("gated", "401", "403", "authorized", "restricted"))


# Keep data/scene_cache/ bounded: evict oldest-by-mtime once past this cap.
_SCENE_CACHE_MAX_BYTES = 256 * 1024 * 1024


def _prune_scene_cache() -> None:
    """Best-effort cache bound; called after each write, must never raise.

    Drops stale MOCK_* thumbs once Earth Engine is real (real scene ids never
    collide with mock ones), then evicts oldest files over the byte cap.
    """
    try:
        files = [p for p in SCENE_CACHE_DIR.iterdir() if p.is_file()]
        if not gee.state().mock:
            for p in [f for f in files if "__MOCK_" in f.name]:
                try:
                    p.unlink()
                    files.remove(p)
                except OSError:
                    pass
        entries = []
        total = 0
        for p in files:
            try:
                st = p.stat()
            except OSError:
                continue
            entries.append((st.st_mtime, st.st_size, p))
            total += st.st_size
        entries.sort()  # oldest first
        for _mtime, size, p in entries:
            if total <= _SCENE_CACHE_MAX_BYTES:
                break
            try:
                p.unlink()
                total -= size
            except OSError:
                pass
    except Exception:  # noqa: BLE001 - pruning must never break a request
        log.debug("scene cache prune failed", exc_info=True)


def _scene_image(record: SpawnRecord, scene_id: str, sensor: str, kind: str) -> Image.Image:
    """True/false color thumb PNG for one scene, cached under data/scene_cache/.

    The cache key includes the configured thumb_px so a config change cannot
    serve a stale-sized thumb (the contract requires the mask to match the
    displayed thumb's WxH exactly).
    """
    thumb_px = int(gee.get_config().get("thumb_px", 1120))
    path = (
        SCENE_CACHE_DIR
        / f"{_safe_name(record.id)}__{_safe_name(scene_id)}__{kind}__{thumb_px}.png"
    )
    if path.exists():
        try:
            img = Image.open(path)
            img.load()
            return img
        except Exception as exc:  # noqa: BLE001 - corrupt cache entry -> refetch
            log.warning("corrupt cached thumb %s (%s); deleting and refetching", path, exc)
            try:
                path.unlink()
            except OSError:
                pass

    if gee.state().mock:
        # Lazy import: app.main imports this module, so pull the mock thumb
        # renderer only at call time to avoid a circular import at startup.
        from app.main import _mock_thumb_png

        data = _mock_thumb_png(gee.mock_thumb_seed(record.id, scene_id, kind))
    else:
        url = gee._thumb_url(sensor, scene_id, kind, record.lon, record.lat, record.length_m)
        resp = requests.get(url, timeout=180)
        resp.raise_for_status()
        data = resp.content

    # Decode BEFORE committing to the cache: an HTTP-200 body that is not a
    # valid image must fail this one request, not poison the cache on disk.
    img = Image.open(io.BytesIO(data))
    img.load()

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".png.part")
    with open(tmp, "wb") as fh:
        fh.write(data)
    tmp.replace(path)
    _prune_scene_cache()
    return img


def _classifier_input(true_img: Image.Image, false_img: Image.Image, mask: np.ndarray):
    """Build the (1, 4, 128, 128) float32 tensor documented in models/README.txt.

    Channels are [R, G, B, NIR] in 0-1 (NIR = red channel of the false-color
    thumb), cropped to the mask's bounding box (+8px margin) with pixels
    outside the SAM mask zeroed, bilinearly resized to 128x128.
    """
    import torch

    size = true_img.size
    t = _rgb_array(true_img, size)
    f = _rgb_array(false_img, size)
    chip = np.concatenate([t, f[..., :1]], axis=2)  # (H, W, 4)
    chip[~mask] = 0.0
    ys, xs = np.nonzero(mask)
    pad = 8
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, mask.shape[0])
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, mask.shape[1])
    crop = np.ascontiguousarray(chip[y0:y1, x0:x1])
    tens = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0)
    return torch.nn.functional.interpolate(tens, size=(128, 128), mode="bilinear", align_corners=False)


# ------------------------------------------------------------------- engine

class SegmentEngine:
    """Singleton wrapping one SAM model, an embedding LRU, and the scorer."""

    def __init__(self) -> None:
        cfg = gee.get_config()
        self._lru_max = max(1, int(cfg.get("segment_lru_scenes", 3)))
        self._state = "cold"  # cold | loading | ready | error
        self._state_lock = threading.Lock()  # guards state/backend/device/hint fields
        self._infer_lock = threading.Lock()  # serializes ALL model use + emb cache
        self._torch: Any = None
        self.model: Any = None
        self.processor: Any = None
        self.dtype: Any = None
        self.backend: str | None = None  # "sam3" | "sam2"
        self.device: str | None = None  # "cuda" | "cpu"
        self.classifier = "heuristic"  # "heuristic" | "trained"
        self._clf: Any = None
        self.error: str | None = None
        self.hint: str | None = COLD_HINT
        # (record_id, scene_id) -> {"embeddings", "original_sizes", "size", "true_img"};
        # embeddings live on self.device (~44-56MB each in bf16, fine for 3 scenes).
        self._emb_cache: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()

    # ---------------------------------------------------------- state / API

    def status(self) -> dict:
        with self._state_lock:
            return {
                "state": self._state,
                "backend": self.backend,
                "device": self.device,
                "classifier": self.classifier,
                "error": self.error,
                "hint": self.hint,
            }

    def warmup(self) -> dict:
        """Start the background load if needed; idempotent, returns immediately."""
        with self._state_lock:
            if self._state in ("loading", "ready"):
                return {"ok": True, "state": self._state}
            # cold, or error -> allow a retry after the user fixes the cause.
            self._state = "loading"
            self.error = None
            self.hint = LOADING_HINT
        threading.Thread(target=self._load, name="segment-load", daemon=True).start()
        return {"ok": True, "state": "loading"}

    # -------------------------------------------------------------- loading

    def _load(self) -> None:
        try:
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            import torch

            self._torch = torch

            try:
                cuda_ok = torch.cuda.is_available()
            except Exception as exc:  # noqa: BLE001 - broken CUDA init -> CPU
                log.warning("CUDA probe failed (%s); using CPU", exc)
                cuda_ok = False

            # Release any previous engine (e.g. warmup retry after an OOM
            # error) BEFORE loading a new one, so the reload never needs
            # double the model VRAM. gc.collect() before empty_cache because
            # exception/traceback/frame cycles commonly keep dead tensors
            # alive past plain refcounting.
            with self._infer_lock:
                with self._state_lock:
                    self.model = None
                    self.processor = None
                    self.backend = None
                    self.device = None
                    self.dtype = None
                self._emb_cache.clear()
            gc.collect()
            if cuda_ok:
                try:
                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass

            attempts: list[tuple[str, str, Any]] = []
            if cuda_ok:
                attempts.append(("sam3", "cuda", torch.bfloat16))
                attempts.append(("sam2", "cuda", torch.bfloat16))
            else:
                log.info("no CUDA: skipping SAM 3 (CPU encode is far too slow), trying SAM 2.1 on CPU")
            attempts.append(("sam2", "cpu", torch.float32))

            sam3_hint: str | None = None
            last_err: str | None = None  # message only: keeping the exception
            # object would pin the failed attempt's model via its traceback
            loaded = False
            for backend, device, dtype in attempts:
                failed = False
                try:
                    model, processor = self._load_backend(backend, device, dtype)
                except Exception as exc:  # noqa: BLE001 - any failure -> next fallback
                    failed = True
                    last_err = f"{type(exc).__name__}: {exc}"
                    if backend == "sam3":
                        sam3_hint = GATED_HINT if _looks_gated(exc) else (
                            f"SAM 3 load failed ({exc}); using the SAM 2.1 fallback."
                        )
                    log.warning("segment backend %s on %s failed: %s", backend, device, exc)
                if failed:
                    # Cleanup OUTSIDE the except block: while the handler is
                    # active, the exception's traceback pins the frames that
                    # hold the (partially) loaded model, so empty_cache there
                    # could not reclaim its VRAM.
                    gc.collect()
                    if cuda_ok:
                        try:
                            torch.cuda.empty_cache()
                        except Exception:  # noqa: BLE001
                            pass
                    continue
                # Install under _infer_lock so no queued inference can observe
                # a torn model/dtype/device combination or reuse embeddings
                # computed by a previous backend.
                with self._infer_lock:
                    self._emb_cache.clear()
                    with self._state_lock:
                        self.model = model
                        self.processor = processor
                        self.backend = backend
                        self.device = device
                        self.dtype = dtype
                        self._state = "ready"
                        self.error = None
                        self.hint = sam3_hint  # gated/fallback instructions, or None
                log.info("segment engine ready: %s on %s (%s)", backend, device, dtype)
                loaded = True
                break

            if not loaded:
                with self._state_lock:
                    self._state = "error"
                    self.error = f"Model load failed: {last_err}"
                    self.hint = sam3_hint or (
                        "Check network connectivity and disk space, then POST "
                        "/api/segment/warmup to retry."
                    )
                return

            self._load_classifier()
        except Exception as exc:  # noqa: BLE001 - the loader thread must never die silently
            log.exception("segment engine load crashed")
            with self._state_lock:
                self._state = "error"
                self.error = f"Model load crashed: {exc}"
                self.hint = None

    def _load_backend(self, backend: str, device: str, dtype: Any):
        """Load model+processor for one backend and run a tiny dummy inference."""
        if backend == "sam3":
            # Sam3TrackerModel/Processor = the interactive point-prompt path
            # (Sam3Model is text/box concept segmentation and takes no points).
            # Expect "some weights ... were not used" warnings: the checkpoint
            # also holds detector/text weights the tracker deliberately skips.
            from transformers import Sam3TrackerModel, Sam3TrackerProcessor

            model = Sam3TrackerModel.from_pretrained(SAM3_REPO, dtype=dtype).to(device).eval()
            processor = Sam3TrackerProcessor.from_pretrained(SAM3_REPO)
        else:
            from transformers import Sam2Model, Sam2Processor

            model = Sam2Model.from_pretrained(SAM2_REPO, dtype=dtype).to(device).eval()
            processor = Sam2Processor.from_pretrained(SAM2_REPO)
        self._dummy_forward(model, processor, device, dtype)
        return model, processor

    def _dummy_forward(self, model, processor, device: str, dtype: Any) -> None:
        """Absorb CUDA context + first-call kernel autotune during warmup."""
        torch = self._torch
        img = Image.new("RGB", (128, 128), (12, 40, 60))
        inputs = processor(images=img, return_tensors="pt")
        with torch.inference_mode():
            emb = model.get_image_embeddings(inputs.pixel_values.to(device, dtype))
            prompt = processor(
                input_points=[[[[64.0, 64.0]]]],
                input_labels=[[[1]]],
                original_sizes=inputs["original_sizes"],
                return_tensors="pt",
            ).to(device)
            model(
                input_points=prompt["input_points"].to(dtype),
                input_labels=prompt["input_labels"],
                image_embeddings=emb,
                multimask_output=True,
            )

    def _load_classifier(self) -> None:
        """Use models/spawn_classifier.pt when present; else keep the heuristic."""
        if not CLASSIFIER_PATH.exists():
            return
        torch = self._torch
        try:
            obj = torch.load(CLASSIFIER_PATH, map_location=self.device or "cpu", weights_only=False)
            if hasattr(obj, "eval"):
                obj.eval()
            if not callable(obj):
                raise TypeError("file did not deserialize to a callable")
            with self._state_lock:
                self._clf = obj
                self.classifier = "trained"
            log.info("trained spawn classifier loaded from %s", CLASSIFIER_PATH)
        except Exception as exc:  # noqa: BLE001 - a broken .pt must not kill the engine
            log.warning(
                "could not load %s (%s); using the spectral heuristic", CLASSIFIER_PATH, exc
            )

    # ---------------------------------------------------------- segmentation

    def segment(self, record: SpawnRecord, scene_id: str, sensor: str, points: list[tuple[float, float, int]]) -> dict:
        """One click session -> mask + score. Always returns a JSON-able dict.

        `points` are (x, y, label) with x/y normalized 0-1 in true-thumb space
        and label 1=foreground / 0=background, per the API contract.
        """
        with self._state_lock:
            state = self._state
        if state == "cold":
            self.warmup()  # first /api/segment triggers the load
            with self._state_lock:
                state = self._state
        if state != "ready":
            with self._state_lock:
                return {"ok": False, "state": self._state, "error": self.error, "hint": self.hint}

        if not gee.state().mock and sensor not in gee._ASSET_PREFIX:
            return {"ok": False, "state": "error", "error": f"Unknown sensor: {sensor}", "hint": None}
        pts = [(float(x), float(y), int(lbl)) for x, y, lbl in points]
        if not any(lbl == 1 for _, _, lbl in pts):
            return {
                "ok": False,
                "state": "error",
                "error": "At least one foreground point (label=1) is required.",
                "hint": None,
            }

        torch = self._torch
        t0 = time.perf_counter()
        try:
            with self._infer_lock:
                # Re-check under the inference lock: a request queued here may
                # have crossed a state transition (OOM -> error, or a reload in
                # progress) and must not run against a torn-down/mid-swap engine.
                with self._state_lock:
                    if self._state != "ready":
                        return {
                            "ok": False,
                            "state": self._state,
                            "error": self.error,
                            "hint": self.hint,
                        }

                # OOM retry protocol: the retry must happen OUTSIDE the except
                # block. While the handler is active, the exception's traceback
                # pins every unwound frame's locals (pixel_values, encoder
                # activations, decoder intermediates), so empty_cache inside
                # the handler cannot reclaim the very memory that OOM'd.
                oom = False
                try:
                    result = self._segment_locked(record, scene_id, sensor, pts)
                except torch.cuda.OutOfMemoryError:
                    oom = True
                    self._emb_cache.clear()
                if oom:
                    log.warning("CUDA OOM during segmentation; clearing caches, retrying once")
                    gc.collect()  # break exception/traceback/frame cycles first
                    torch.cuda.empty_cache()
                    oom_again: str | None = None
                    try:
                        result = self._segment_locked(record, scene_id, sensor, pts)
                    except torch.cuda.OutOfMemoryError as exc:
                        oom_again = str(exc)
                        # Drop the engine so the error state does not keep the
                        # model's VRAM pinned while the hint asks the user to
                        # free GPU memory; warmup reloads from scratch.
                        self.model = None
                        self.processor = None
                        self._emb_cache.clear()
                    if oom_again is not None:
                        gc.collect()
                        torch.cuda.empty_cache()
                        hint = (
                            "GPU out of memory even after a retry -- close other GPU-heavy "
                            "apps, then POST /api/segment/warmup to reload."
                        )
                        with self._state_lock:
                            self._state = "error"
                            self.error = f"CUDA out of memory: {oom_again}"
                            self.hint = hint
                        return {"ok": False, "state": "error", "error": self.error, "hint": hint}
        except Exception as exc:  # noqa: BLE001 - contract: HTTP 200 with ok:false
            log.exception("segmentation failed for %s / %s", record.id, scene_id)
            return {"ok": False, "state": "error", "error": f"Segmentation failed: {exc}", "hint": None}

        result["timing_ms"] = int((time.perf_counter() - t0) * 1000)
        return result

    def _segment_locked(self, record: SpawnRecord, scene_id: str, sensor: str, pts: list[tuple[float, float, int]]) -> dict:
        entry = self._embeddings_for(record, scene_id, sensor)
        w, h = entry["size"]
        # Normalized -> pixel coords of the true thumb; (x=column, y=row).
        points_px = [[x * w, y * h] for x, y, _ in pts]
        labels = [lbl for _, _, lbl in pts]
        mask, iou = self._decode(entry, points_px, labels)

        area_px = int(mask.sum())
        total_px = int(mask.size)
        coverage_pct = 100.0 * area_px / total_px if total_px else 0.0
        # Ground resolution from the chip box: BOTH axes span ~2*half_m on the
        # ground, but EE renders thumbs in plate-carree (no crs param), so at
        # BC latitudes the PNG is non-square (w > h) and the per-axis scales
        # differ (~1/cos(lat)). Multiply per-axis scales: identical to the
        # contract's (m_per_px)^2 when the thumb is square, correct when not.
        half_m = gee.chip_half_size_m(record.length_m)
        m_per_px_x = 2.0 * half_m / float(w)
        m_per_px_y = 2.0 * half_m / float(h)
        area_m2 = float(area_px) * m_per_px_x * m_per_px_y

        if area_px == 0:
            spawn = {
                "kind": "heuristic",
                "prob": 0.0,
                "note": "Empty mask -- nothing segmented at that point.",
            }
        else:
            spawn = self._score(record, scene_id, sensor, entry, mask)

        return {
            "ok": True,
            "mask_png": _mask_data_uri(mask),
            "area_px": area_px,
            "area_m2": round(area_m2, 1),
            "coverage_pct": round(coverage_pct, 3),
            "sam_iou": round(iou, 4),
            "spawn_score": spawn,
            "backend": self.backend,
            "device": self.device,
        }

    def _embeddings_for(self, record: SpawnRecord, scene_id: str, sensor: str) -> dict[str, Any]:
        """Cached encoder outputs for one scene (call with _infer_lock held)."""
        key = (record.id, scene_id)
        entry = self._emb_cache.get(key)
        if entry is not None:
            self._emb_cache.move_to_end(key)
            return entry

        torch = self._torch
        true_img = _scene_image(record, scene_id, sensor, "true")
        inputs = self.processor(images=true_img.convert("RGB"), return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.device, self.dtype)
        with torch.inference_mode():
            # Ready-to-reuse FPN features (no_memory_embedding + conv_s0/s1
            # already applied) -- exactly what forward(image_embeddings=) expects.
            embeddings = self.model.get_image_embeddings(pixel_values)

        entry = {
            "embeddings": embeddings,
            "original_sizes": inputs["original_sizes"],  # [[H, W]] of the thumb
            "size": true_img.size,  # (W, H)
            "true_img": true_img,
        }
        self._emb_cache[key] = entry
        while len(self._emb_cache) > self._lru_max:
            self._emb_cache.popitem(last=False)  # drop oldest; GC frees its VRAM
        return entry

    def _decode(self, entry: dict[str, Any], points_px: list[list[float]], labels: list[int]) -> tuple[np.ndarray, float]:
        """One prompt decode against cached embeddings -> (bool mask, best IoU).

        All points of the click session go into ONE object slot (4-deep
        input_points, 3-deep input_labels): background points refine the same
        mask instead of spawning a second object.
        """
        torch = self._torch
        prompt = self.processor(
            input_points=[[points_px]],  # [image][object][point][x, y] in thumb pixels
            input_labels=[[labels]],  # [image][object][point], 1=fg / 0=bg
            original_sizes=entry["original_sizes"],  # required when images= is omitted
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            outputs = self.model(
                input_points=prompt["input_points"].to(self.dtype),
                input_labels=prompt["input_labels"],
                image_embeddings=entry["embeddings"],
                multimask_output=True,
            )
        iou = outputs.iou_scores[0, 0]  # (3,) predicted IoU per candidate mask
        best = int(iou.argmax())
        # .float() before .cpu(): numpy has no bf16 and half CPU interp is flaky.
        masks = self.processor.post_process_masks(
            outputs.pred_masks.float().cpu(), entry["original_sizes"]
        )[0]  # (1, 3, H, W) bool at exact thumb size
        mask = masks[0, best].numpy().astype(bool)
        return mask, float(iou[best])

    # -------------------------------------------------------------- scoring

    def _score(self, record: SpawnRecord, scene_id: str, sensor: str, entry: dict[str, Any], mask: np.ndarray) -> dict:
        true_img: Image.Image = entry["true_img"]
        try:
            false_img = _scene_image(record, scene_id, sensor, "false")
        except Exception as exc:  # noqa: BLE001 - score is best-effort
            return {
                "kind": "heuristic",
                "prob": 0.5,
                "note": f"False-color thumb unavailable ({exc}); no spectral score.",
            }
        if self._clf is not None:
            try:
                return self._model_score(true_img, false_img, mask)
            except Exception as exc:  # noqa: BLE001
                log.warning("trained classifier failed (%s); falling back to heuristic", exc)
        return self._heuristic_score(true_img, false_img, mask)

    def _model_score(self, true_img: Image.Image, false_img: Image.Image, mask: np.ndarray) -> dict:
        torch = self._torch
        x = _classifier_input(true_img, false_img, mask)
        with torch.inference_mode():
            logit = self._clf(x.to(self.device))
        logit = torch.as_tensor(logit).reshape(-1)[0].float()
        prob = float(torch.sigmoid(logit))
        return {
            "kind": "model",
            "prob": round(prob, 4),
            "note": "Trained classifier (models/spawn_classifier.pt).",
        }

    def _heuristic_score(self, true_img: Image.Image, false_img: Image.Image, mask: np.ndarray) -> dict:
        """Spectral heuristic: mask interior vs a ring just OUTSIDE the mask.

        Milt is bright in the visible, dark in NIR, and cyan-green dominant;
        whitecaps/clouds are bright in BOTH visible and NIR. Three lifts
        (mask mean minus ring mean) are squashed through a weighted sigmoid.
        """
        size = true_img.size
        t = _rgb_array(true_img, size)
        f = _rgb_array(false_img, size)
        ring = _dilate(mask, _RING_PX) & ~mask
        if not ring.any():
            return {
                "kind": "heuristic",
                "prob": 0.5,
                "note": "Mask covers the whole chip -- no outside ring to compare against.",
            }

        v = t.max(axis=2)  # HSV value = visible brightness
        nir = f[..., 0]  # false color is (NIR, R, G) -> its red channel ~= NIR
        cyan = 0.5 * (t[..., 1] + t[..., 2]) - t[..., 0]  # G+B vs R dominance

        b_lift = float(v[mask].mean() - v[ring].mean())
        nir_lift = float(nir[mask].mean() - nir[ring].mean())
        nir_supp = b_lift - nir_lift  # bright-visible but NOT bright-NIR
        c_lift = float(cyan[mask].mean() - cyan[ring].mean())

        z = _W_BRIGHT * b_lift + _W_NIRSUP * nir_supp + _W_CYAN * c_lift + _BIAS
        prob = 1.0 / (1.0 + math.exp(-z))
        note = (
            "Spectral heuristic, not a trained model: vs a "
            f"{_RING_PX}px outside ring -- visible lift {b_lift:+.3f}, "
            f"NIR suppression {nir_supp:+.3f}, cyan-green lift {c_lift:+.3f}."
        )
        return {"kind": "heuristic", "prob": round(prob, 4), "note": note}


# ---------------------------------------------------------------- singleton

_engine: SegmentEngine | None = None
_engine_lock = threading.Lock()


def engine() -> SegmentEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = SegmentEngine()
        return _engine

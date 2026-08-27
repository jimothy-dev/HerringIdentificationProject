"""Persistent label store backed by data/labels.csv.

The CSV is denormalized (record metadata is copied onto every label row) so
labels.csv alone is a usable training manifest. All writes rewrite the whole
file atomically (temp file + os.replace) under a lock.
"""

from __future__ import annotations

import csv
import datetime as dt
import os
import tempfile
import threading
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELS_PATH = PROJECT_ROOT / "data" / "labels.csv"

COLUMNS = [
    "record_id",
    "scene_id",
    "sensor",
    "scene_date",
    "label",
    "notes",
    "region",
    "year",
    "location_name",
    "lon",
    "lat",
    "start_date",
    "chip_path",
    "labeled_at_utc",
]

VALID_LABELS = ("positive", "negative", "unsure", "unusable")


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class LabelStore:
    """In-memory dict keyed (record_id, scene_id), mirrored to labels.csv."""

    def __init__(self, path: Path = LABELS_PATH) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._rows: dict[tuple[str, str], dict[str, str]] = {}
        self._load_or_create()

    # ------------------------------------------------------------- reads

    def get(self, record_id: str, scene_id: str) -> dict[str, str] | None:
        with self._lock:
            row = self._rows.get((record_id, scene_id))
            return dict(row) if row else None

    def all_rows(self, record_id: str | None = None) -> list[dict[str, str]]:
        with self._lock:
            rows = [dict(r) for r in self._rows.values()]
        if record_id is not None:
            rows = [r for r in rows if r["record_id"] == record_id]
        rows.sort(key=lambda r: (r["record_id"], r["scene_date"], r["scene_id"]))
        return rows

    def label_for(self, record_id: str, scene_id: str) -> str | None:
        row = self.get(record_id, scene_id)
        return row["label"] if row else None

    def record_label_counts(self) -> dict[str, int]:
        with self._lock:
            return dict(Counter(rid for rid, _sid in self._rows))

    def counts(self) -> dict[str, int]:
        with self._lock:
            labels = [r["label"] for r in self._rows.values()]
        return {
            "n_labeled": len(labels),
            "n_positive": sum(1 for x in labels if x == "positive"),
            "n_negative": sum(1 for x in labels if x == "negative"),
        }

    # ------------------------------------------------------------ writes

    def upsert(self, row: dict) -> dict[str, str]:
        """Insert or replace the label for (record_id, scene_id).

        Preserves an existing chip_path unless the caller supplies one.
        """
        key = (str(row["record_id"]), str(row["scene_id"]))
        clean = {col: str(row.get(col, "") if row.get(col) is not None else "") for col in COLUMNS}
        with self._lock:
            existing = self._rows.get(key)
            if existing and not clean.get("chip_path"):
                clean["chip_path"] = existing.get("chip_path", "")
            clean["labeled_at_utc"] = _utc_now_iso()
            self._rows[key] = clean
            self._write_locked()
        return dict(clean)

    def delete(self, record_id: str, scene_id: str) -> bool:
        with self._lock:
            removed = self._rows.pop((record_id, scene_id), None) is not None
            if removed:
                self._write_locked()
        return removed

    def set_chip_path(self, record_id: str, scene_id: str, chip_path: str) -> None:
        with self._lock:
            row = self._rows.get((record_id, scene_id))
            if row is None:
                return
            row["chip_path"] = chip_path
            self._write_locked()

    # ----------------------------------------------------------- internal

    def _load_or_create(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            with self._lock:
                self._write_locked()
            return
        with open(self._path, "r", newline="", encoding="utf-8") as f:
            for raw in csv.DictReader(f):
                row = {col: (raw.get(col) or "") for col in COLUMNS}
                if not row["record_id"] or not row["scene_id"]:
                    continue
                self._rows[(row["record_id"], row["scene_id"])] = row

    def _write_locked(self) -> None:
        """Atomically rewrite labels.csv. Caller must hold self._lock."""
        fd, tmp_name = tempfile.mkstemp(
            prefix="labels_", suffix=".csv.tmp", dir=str(self._path.parent)
        )
        try:
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS)
                writer.writeheader()
                for key in sorted(self._rows):
                    writer.writerow(self._rows[key])
            os.replace(tmp_name, self._path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

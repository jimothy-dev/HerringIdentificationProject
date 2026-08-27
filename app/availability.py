"""Scene-availability cache + background sweep.

Whether a record's spawn window has ANY usable scene is only knowable via a
GEE query, so it is precomputed and cached in data/scene_availability.json:

    {record_id: {"n": int, "ceiling": float, "checked_at": iso}}

where "n" is the usable-scene count found in spawn mode at the regional-cloud
ceiling "ceiling". Entries are written organically by /api/records/{id}/scenes
(config ceiling, no override) and in bulk by the background sweep. All writes
rewrite the whole file atomically (temp file + os.replace) under a lock, the
same pattern as labels.py.

Entries produced while Earth Engine is in MOCK mode carry an extra
"mock": true marker: fabricated counts are only trusted while the app is
still in mock mode (status_of's mock_ok flag) and read as "unknown" once EE
is live, so they get re-checked with real queries instead of hiding real
records (or un-hiding empty ones) forever.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("herring.availability")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AVAILABILITY_PATH = PROJECT_ROOT / "data" / "scene_availability.json"


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class AvailabilityStore:
    """In-memory dict keyed by record_id, mirrored to scene_availability.json."""

    def __init__(self, path: Path = AVAILABILITY_PATH) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, Any]] = {}
        self._load()

    # ------------------------------------------------------------- reads

    def get(self, record_id: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._entries.get(record_id)
            return dict(entry) if entry else None

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """One consistent copy of every entry (one lock acquisition per request)."""
        with self._lock:
            return {rid: dict(entry) for rid, entry in self._entries.items()}

    @staticmethod
    def status_of(entry: dict[str, Any] | None, ceiling: float, mock_ok: bool = False) -> str:
        """Map a cache entry to "ok" | "empty" | "unknown" at the current ceiling.

        An entry checked at a DIFFERENT ceiling is stale ("unknown"), with one
        safe exception: n > 0 found at a stricter (lower) ceiling still
        guarantees scenes exist at the current one.

        An entry marked "mock" (fabricated scenes) is only meaningful while the
        app is still in mock mode (mock_ok=True); once EE is live it must read
        as "unknown" so real queries replace it.
        """
        if entry is None:
            return "unknown"
        if entry.get("mock") and not mock_ok:
            return "unknown"
        entry_ceiling = float(entry["ceiling"])
        n = int(entry["n"])
        if entry_ceiling == float(ceiling):
            return "ok" if n > 0 else "empty"
        if n > 0 and entry_ceiling <= float(ceiling):
            return "ok"
        return "unknown"

    def status_for(self, record_id: str, ceiling: float, mock_ok: bool = False) -> str:
        return self.status_of(self.get(record_id), ceiling, mock_ok)

    # ------------------------------------------------------------ writes

    def set(self, record_id: str, n: int, ceiling: float, mock: bool = False) -> None:
        """Record the usable-scene count found for one record at a ceiling.

        mock=True marks a count fabricated by mock mode (see module docstring);
        a later real-mode write for the same record drops the marker.

        Unlike labels.py this does NOT roll back memory on a failed disk write:
        the count came from a real query and stays authoritative for this
        session, and the next successful write persists everything anyway.
        """
        entry: dict[str, Any] = {
            "n": int(n),
            "ceiling": float(ceiling),
            "checked_at": _utc_now_iso(),
        }
        if mock:
            entry["mock"] = True
        with self._lock:
            self._entries[str(record_id)] = entry
            try:
                self._write_locked()
            except OSError as exc:
                log.warning(
                    "could not write %s: %s (entry kept in memory)", self._path.name, exc
                )

    # ----------------------------------------------------------- internal

    def _load(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
        # ValueError covers JSONDecodeError/UnicodeDecodeError -- a corrupt
        # cache must never stop the app; it just gets re-swept.
        except (OSError, ValueError) as exc:
            log.warning("%s unreadable (%s); starting empty", self._path.name, exc)
            return
        if not isinstance(loaded, dict):
            log.warning("%s is not a JSON object; starting empty", self._path.name)
            return
        for rid, entry in loaded.items():
            try:
                parsed: dict[str, Any] = {
                    "n": int(entry["n"]),
                    "ceiling": float(entry["ceiling"]),
                    "checked_at": str(entry.get("checked_at", "")),
                }
                if entry.get("mock"):
                    parsed["mock"] = True
                self._entries[str(rid)] = parsed
            except (TypeError, KeyError, ValueError):
                continue  # skip malformed entries, keep the rest

    def _write_locked(self) -> None:
        """Atomically rewrite scene_availability.json. Caller must hold self._lock."""
        fd, tmp_name = tempfile.mkstemp(
            prefix="scene_availability_", suffix=".json.tmp", dir=str(self._path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._entries, f, indent=2, sort_keys=True)
            os.replace(tmp_name, self._path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise


class SweepRunner:
    """The single background availability sweep (at most one at a time).

    start() launches a daemon thread over the given records; a second start()
    while one runs reports "already_running" with the remaining count (the new
    filters are ignored -- deliberately simple). A per-record failure leaves
    that record unknown and the sweep continues.
    """

    def __init__(self, store: AvailabilityStore) -> None:
        self._store = store
        self._lock = threading.Lock()
        self._running = False
        self._done = 0
        self._total = 0
        self._empty_found = 0

    def start(
        self,
        targets: list[Any],
        check_fn: Callable[[Any], int],
        ceiling: float,
        sleep_s: float = 0.3,
        mock: bool = False,
    ) -> tuple[str, int]:
        """Start sweeping `targets` (records with an .id attribute).

        check_fn(record) -> usable-scene count (metadata only -- no thumbnails).
        mock=True marks every written entry as fabricated (mock mode).
        Returns ("started", len(targets)) or ("already_running", n_remaining).
        """
        with self._lock:
            if self._running:
                return "already_running", self._total - self._done
            self._running = True
            self._done = 0
            self._total = len(targets)
            self._empty_found = 0
        thread = threading.Thread(
            target=self._run,
            args=(list(targets), check_fn, float(ceiling), sleep_s, bool(mock)),
            name="availability-sweep",
            daemon=True,
        )
        thread.start()
        return "started", len(targets)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "running": self._running,
                "done": self._done,
                "total": self._total,
                "empty_found": self._empty_found,
            }

    def _run(
        self,
        targets: list[Any],
        check_fn: Callable[[Any], int],
        ceiling: float,
        sleep_s: float,
        mock: bool = False,
    ) -> None:
        log.info("availability sweep started: %d record(s)", len(targets))
        try:
            for record in targets:
                try:
                    n = check_fn(record)
                except Exception as exc:  # noqa: BLE001 - one bad record must not stop the sweep
                    log.warning("sweep: check failed for %s: %s (left unknown)", record.id, exc)
                else:
                    # Written per record so /api/records reflects progress live.
                    self._store.set(record.id, n, ceiling, mock=mock)
                    if n == 0:
                        with self._lock:
                            self._empty_found += 1
                with self._lock:
                    self._done += 1
                if sleep_s > 0:
                    time.sleep(sleep_s)
        finally:
            with self._lock:
                self._running = False
                done, empty = self._done, self._empty_found
        log.info("availability sweep finished: %d checked, %d empty", done, empty)

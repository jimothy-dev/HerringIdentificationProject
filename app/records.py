"""Load and query the herring spawn index CSV.

Only rows with Year >= 2016 (Sentinel-2 era), a parseable StartDate and valid
coordinates are kept. Record ids follow the contract:
"{Year}_{LocationCode}_{SpawnNumber}" (with a "_{rowindex}" suffix only if a
collision were ever found -- the 2025 file has none).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "Pacific_herring_spawn_index_data_2025_EN.csv"

MIN_YEAR = 2016


@dataclass(frozen=True)
class SpawnRecord:
    id: str
    region: str
    year: int
    location_name: str
    spawn_number: int
    start_date: dt.date
    end_date: dt.date | None
    lon: float
    lat: float
    length_m: float | None
    width_m: float | None
    method: str

    def to_api(self, n_labels: int) -> dict:
        return {
            "id": self.id,
            "region": self.region,
            "year": self.year,
            "location_name": self.location_name,
            "spawn_number": self.spawn_number,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "lon": self.lon,
            "lat": self.lat,
            "length_m": self.length_m,
            "width_m": self.width_m,
            "method": self.method,
            "n_labels": n_labels,
        }


def _parse_date(raw: str) -> dt.date | None:
    raw = (raw or "").strip()
    if not raw or raw == "NA":
        return None
    try:
        return dt.date.fromisoformat(raw)
    except ValueError:
        return None


def _parse_float(raw: str) -> float | None:
    raw = (raw or "").strip()
    if not raw or raw == "NA":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_int(raw: str, default: int = 0) -> int:
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


class RecordStore:
    """In-memory store of usable spawn records with filter/pagination helpers."""

    def __init__(self, csv_path: Path = CSV_PATH) -> None:
        self.records: list[SpawnRecord] = _load_records(csv_path)
        self.by_id: dict[str, SpawnRecord] = {r.id: r for r in self.records}
        self.years: list[int] = sorted({r.year for r in self.records}, reverse=True)
        self.regions: list[str] = sorted({r.region for r in self.records})

    def get(self, record_id: str) -> SpawnRecord | None:
        return self.by_id.get(record_id)

    def query(
        self,
        year: int | None = None,
        region: str | None = None,
        labeled: str = "all",
        page: int = 1,
        page_size: int = 50,
        n_labels_fn: Callable[[str], int] | None = None,
    ) -> tuple[int, list[tuple[SpawnRecord, int]]]:
        """Return (total, [(record, n_labels), ...]) for the requested page.

        `labeled` is "all" | "unlabeled" | "labeled"; a record counts as
        labeled when it has >= 1 saved label.
        """
        count = n_labels_fn or (lambda _rid: 0)
        matched: list[tuple[SpawnRecord, int]] = []
        for rec in self.records:
            if year is not None and rec.year != year:
                continue
            if region is not None and rec.region != region:
                continue
            n = count(rec.id)
            if labeled == "labeled" and n < 1:
                continue
            if labeled == "unlabeled" and n >= 1:
                continue
            matched.append((rec, n))

        total = len(matched)
        page = max(1, page)
        page_size = max(1, min(page_size, 500))
        start = (page - 1) * page_size
        return total, matched[start : start + page_size]


def _load_records(csv_path: Path) -> list[SpawnRecord]:
    # keep_default_na=False: "NA" is a real Region value (187 rows) and also the
    # literal missing marker in date/number columns -- parse everything by hand.
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_values=[])

    records: list[SpawnRecord] = []
    seen: set[str] = set()
    for row_index, row in enumerate(df.itertuples(index=False)):
        year = _parse_int(row.Year, default=0)
        if year < MIN_YEAR:
            continue
        start_date = _parse_date(row.StartDate)
        lon = _parse_float(row.Longitude)
        lat = _parse_float(row.Latitude)
        if start_date is None or lon is None or lat is None:
            continue

        end_date = _parse_date(row.EndDate)
        if end_date is not None and end_date < start_date:
            # 2 rows in the 2025 file have EndDate < StartDate (day/month typo);
            # treat the end date as invalid there.
            end_date = None

        rec_id = f"{year}_{row.LocationCode}_{row.SpawnNumber}"
        if rec_id in seen:
            rec_id = f"{rec_id}_{row_index}"
        seen.add(rec_id)

        records.append(
            SpawnRecord(
                id=rec_id,
                region=row.Region,
                year=year,
                location_name=row.LocationName,
                spawn_number=_parse_int(row.SpawnNumber, default=0),
                start_date=start_date,
                end_date=end_date,
                lon=lon,
                lat=lat,
                length_m=_parse_float(row.Length),
                width_m=_parse_float(row.Width),
                method=row.Method,
            )
        )

    # Contract: year desc, then start_date (ascending within a year).
    records.sort(key=lambda r: (-r.year, r.start_date))
    return records

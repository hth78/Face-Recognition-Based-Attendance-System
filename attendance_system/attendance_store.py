from __future__ import annotations

import csv
import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class AttendanceRecord:
    person_id: int
    name: str
    date: str
    time: str


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def append_student(student_csv: Path, person_id: int, name: str) -> None:
    ensure_directory(student_csv.parent)
    file_exists = student_csv.exists()
    try:
        with student_csv.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if not file_exists:
                writer.writerow(["Id", "Name"])
            writer.writerow([person_id, name])
    except OSError as exc:
        LOGGER.exception("Failed writing student details CSV")
        raise RuntimeError("Unable to write student details") from exc


def load_students(student_csv: Path) -> pd.DataFrame:
    if not student_csv.exists():
        raise FileNotFoundError(f"Student database not found: {student_csv}")
    df = pd.read_csv(student_csv)
    if df.empty:
        raise ValueError("Student database is empty")
    return df


def build_attendance_filename(attendance_dir: Path, at_time: dt.datetime | None = None) -> Path:
    ensure_directory(attendance_dir)
    at_time = at_time or dt.datetime.now()
    stamp = at_time.strftime("%Y-%m-%d_%H-%M")
    return attendance_dir / f"Attendance_{stamp}.csv"


def write_attendance(path: Path, records: Iterable[AttendanceRecord]) -> Path:
    rows = [
        {"Id": r.person_id, "Name": r.name, "Date": r.date, "Time": r.time}
        for r in records
    ]
    df = pd.DataFrame(rows, columns=["Id", "Name", "Date", "Time"])
    if not df.empty:
        df = df.drop_duplicates(subset=["Id"], keep="first")
    df.to_csv(path, index=False)
    LOGGER.info("Wrote attendance report: %s", path)
    return path


def latest_attendance_report(attendance_dir: Path) -> Path:
    if not attendance_dir.exists():
        raise FileNotFoundError(f"Attendance directory not found: {attendance_dir}")
    files = sorted(attendance_dir.glob("Attendance_*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError("No attendance reports available")
    return files[-1]

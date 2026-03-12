from __future__ import annotations

import argparse
import array
import json
import math
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class RecognitionResult:
    identity: str | None
    distance: float


def _to_float_list(vector: Sequence[float]) -> list[float]:
    return [float(v) for v in vector]


def _serialize_vector(vector: Sequence[float]) -> bytes:
    arr = array.array("f", _to_float_list(vector))
    return arr.tobytes()


def _deserialize_vector(blob: bytes) -> list[float]:
    arr = array.array("f")
    arr.frombytes(blob)
    return list(arr)


def _l2_norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in vector))


def _normalize(vector: Sequence[float]) -> list[float]:
    norm = _l2_norm(vector)
    if norm == 0:
        return [0.0 for _ in vector]
    return [float(v) / norm for v in vector]


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)))


class EmbeddingStore:
    """Versioned embedding persistence backed by SQLite."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS enrollment_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    identity TEXT NOT NULL,
                    sample BLOB NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS embedding_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    status TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    activated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version_id INTEGER NOT NULL,
                    identity TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(version_id) REFERENCES embedding_versions(id)
                );
                """
            )

    @staticmethod
    def _utcnow() -> str:
        return datetime.now(timezone.utc).isoformat()

    def collect_enrollment_sample(self, identity: str, sample: bytes, metadata: dict | None = None) -> int:
        """Persist enrollment-only data; no embedding/index mutation."""
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO enrollment_samples(identity, sample, metadata, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (identity, sample, json.dumps(metadata or {}), self._utcnow()),
            )
            return int(cur.lastrowid)

    def start_version(self, metadata: dict | None = None) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO embedding_versions(status, metadata, created_at)
                VALUES ('building', ?, ?)
                """,
                (json.dumps(metadata or {}), self._utcnow()),
            )
            return int(cur.lastrowid)

    def add_embedding(
        self,
        version_id: int,
        identity: str,
        vector: Sequence[float],
        metadata: dict | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO embeddings(version_id, identity, vector, metadata, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (version_id, identity, _serialize_vector(vector), json.dumps(metadata or {}), self._utcnow()),
            )

    def activate_version(self, version_id: int) -> None:
        """Atomically make one version active and archive previous active versions."""
        with self._connect() as conn:
            conn.execute("BEGIN")
            conn.execute("UPDATE embedding_versions SET status='archived' WHERE status='active'")
            conn.execute(
                """
                UPDATE embedding_versions
                SET status='active', activated_at=?
                WHERE id=? AND status='building'
                """,
                (self._utcnow(), version_id),
            )
            changed = conn.execute("SELECT changes()").fetchone()[0]
            if changed != 1:
                conn.execute("ROLLBACK")
                raise ValueError(f"Version {version_id} cannot be activated")
            conn.commit()

    def fetch_active_embeddings(self) -> list[tuple[str, list[float]]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT e.identity, e.vector
                FROM embeddings e
                JOIN embedding_versions v ON v.id = e.version_id
                WHERE v.status='active'
                ORDER BY e.id
                """
            ).fetchall()
        return [(r["identity"], _deserialize_vector(r["vector"])) for r in rows]


class EmbeddingIndex:
    def __init__(self, identities: Sequence[str], vectors: Sequence[Sequence[float]]):
        self.identities = list(identities)
        self.vectors = [list(v) for v in vectors]

    @classmethod
    def from_store(cls, store: EmbeddingStore) -> "EmbeddingIndex":
        items = store.fetch_active_embeddings()
        if not items:
            return cls([], [])
        identities, vectors = zip(*items)
        return cls(identities, vectors)

    def recognize(self, embedding: Sequence[float], threshold: float) -> RecognitionResult:
        if not self.vectors:
            return RecognitionResult(None, float("inf"))

        query = _normalize(_to_float_list(embedding))
        distances = [_distance(v, query) for v in self.vectors]
        best_idx, best_dist = min(enumerate(distances), key=lambda x: x[1])
        if best_dist <= threshold:
            return RecognitionResult(self.identities[best_idx], best_dist)
        return RecognitionResult(None, best_dist)


def train_embeddings(store: EmbeddingStore, version_metadata: dict | None = None) -> int:
    """Build embeddings from enrolled samples into a new version and activate atomically."""
    version_id = store.start_version(version_metadata)
    with store._connect() as conn:
        rows = conn.execute("SELECT identity, sample FROM enrollment_samples ORDER BY id").fetchall()

    for row in rows:
        sample = _deserialize_vector(row["sample"])
        embedding = _normalize(sample)
        store.add_embedding(version_id, row["identity"], embedding)

    store.activate_version(version_id)
    return version_id


def start_training_in_thread(store: EmbeddingStore, metadata: dict | None = None) -> threading.Thread:
    """Correct callable usage for migrated code: pass target function + args."""
    worker = threading.Thread(target=train_embeddings, args=(store, metadata), daemon=True)
    worker.start()
    return worker


def parse_embedding(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",")]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Model lifecycle utilities")
    parser.add_argument("--db", default="embeddings.sqlite", help="SQLite path")
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect-enrollment", help="Save enrollment sample only")
    collect.add_argument("--identity", required=True)
    collect.add_argument("--sample", required=True, help="Comma-separated floats")

    sub.add_parser("build-index", help="Build embeddings and atomically activate")

    recognize = sub.add_parser("recognize", help="Recognize query embedding")
    recognize.add_argument("--query", required=True, help="Comma-separated floats")
    recognize.add_argument("--threshold", type=float, default=0.75)

    args = parser.parse_args(list(argv) if argv is not None else None)
    store = EmbeddingStore(args.db)

    if args.command == "collect-enrollment":
        sample = parse_embedding(args.sample)
        store.collect_enrollment_sample(args.identity, _serialize_vector(sample))
        print(f"Collected sample for {args.identity}")
        return 0

    if args.command == "build-index":
        version = train_embeddings(store)
        print(f"Activated embedding version {version}")
        return 0

    query = parse_embedding(args.query)
    index = EmbeddingIndex.from_store(store)
    result = index.recognize(query, args.threshold)
    print(json.dumps({"identity": result.identity, "distance": result.distance}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

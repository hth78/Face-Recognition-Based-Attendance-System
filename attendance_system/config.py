from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DetectorConfig:
    provider: str = "insightface"
    model_name: str = "buffalo_l"
    det_size: tuple[int, int] = (640, 640)
    ctx_id: int = -1


@dataclass(frozen=True)
class EmbedderConfig:
    provider: str = "insightface"
    model_name: str = "buffalo_l"
    ctx_id: int = -1


@dataclass(frozen=True)
class MatcherConfig:
    cosine_threshold: float = 0.4


@dataclass(frozen=True)
class AppConfig:
    detector: DetectorConfig = DetectorConfig()
    embedder: EmbedderConfig = EmbedderConfig()
    matcher: MatcherConfig = MatcherConfig()


def _as_tuple(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    return default


def load_config(config_path: str | Path = "config.yaml") -> AppConfig:
    path = Path(config_path)
    if not path.exists():
        return AppConfig()

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    detector_raw = raw.get("detector", {})
    embedder_raw = raw.get("embedder", {})
    matcher_raw = raw.get("matcher", {})

    return AppConfig(
        detector=DetectorConfig(
            provider=detector_raw.get("provider", "insightface"),
            model_name=detector_raw.get("model_name", "buffalo_l"),
            det_size=_as_tuple(detector_raw.get("det_size"), (640, 640)),
            ctx_id=int(detector_raw.get("ctx_id", -1)),
        ),
        embedder=EmbedderConfig(
            provider=embedder_raw.get("provider", "insightface"),
            model_name=embedder_raw.get("model_name", "buffalo_l"),
            ctx_id=int(embedder_raw.get("ctx_id", -1)),
        ),
        matcher=MatcherConfig(
            cosine_threshold=float(matcher_raw.get("cosine_threshold", 0.4)),
        ),
    )

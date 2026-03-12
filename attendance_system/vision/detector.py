from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]
    score: float
    landmarks: np.ndarray | None = None


class FaceDetector:
    """Modern detector wrapper using InsightFace (SCRFD-based models)."""

    def __init__(self, model_name: str = "buffalo_l", det_size: tuple[int, int] = (640, 640), ctx_id: int = -1):
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise ImportError(
                "InsightFace is required for FaceDetector. Install `insightface` to use SCRFD/RetinaFace detectors."
            ) from exc

        self._app = FaceAnalysis(name=model_name, allowed_modules=["detection"])
        self._app.prepare(ctx_id=ctx_id, det_size=det_size)

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        faces: list[Any] = self._app.get(image_bgr)
        detections: list[Detection] = []
        for face in faces:
            bbox = tuple(float(v) for v in face.bbox)
            score = float(getattr(face, "det_score", 0.0))
            landmarks = getattr(face, "kps", None)
            detections.append(Detection(bbox=bbox, score=score, landmarks=landmarks))
        return detections

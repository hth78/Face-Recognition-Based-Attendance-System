from __future__ import annotations

import numpy as np

from .config import AppConfig
from .vision import FaceDetector, FaceEmbedder, FaceMatcher


class AttendanceRecognitionPipeline:
    def __init__(self, config: AppConfig):
        self.detector = FaceDetector(
            model_name=config.detector.model_name,
            det_size=config.detector.det_size,
            ctx_id=config.detector.ctx_id,
        )
        self.embedder = FaceEmbedder(
            model_name=config.embedder.model_name,
            ctx_id=config.embedder.ctx_id,
        )
        self.matcher = FaceMatcher(threshold=config.matcher.cosine_threshold)

    def identify(self, frame_bgr: np.ndarray, enrolled_embeddings: dict[str, np.ndarray]):
        detections = self.detector.detect(frame_bgr)
        if not detections:
            return None

        best_face = max(detections, key=lambda d: d.score)
        x1, y1, x2, y2 = [int(v) for v in best_face.bbox]
        face_crop = frame_bgr[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
        if face_crop.size == 0:
            return None

        embedding = self.embedder.embed_aligned_face(face_crop)
        return self.matcher.match(embedding, enrolled_embeddings)

from __future__ import annotations

import cv2
import numpy as np


class FaceEmbedder:
    """ArcFace/FaceNet-style embedding wrapper using InsightFace models."""

    def __init__(self, model_name: str = "arcface_r100_v1", ctx_id: int = -1):
        try:
            from insightface.model_zoo import get_model
        except ImportError as exc:
            raise ImportError(
                "InsightFace is required for FaceEmbedder. Install `insightface` to use ArcFace embeddings."
            ) from exc

        self._model = get_model(model_name)
        self._model.prepare(ctx_id=ctx_id)

    def embed_aligned_face(self, face_bgr: np.ndarray) -> np.ndarray:
        """Generate a normalized embedding from a pre-aligned face crop."""
        resized = cv2.resize(face_bgr, (112, 112), interpolation=cv2.INTER_AREA)
        embedding = self._model.get_feat(resized).astype(np.float32).reshape(-1)
        return self._normalize(embedding)

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            return vector
        return vector / norm

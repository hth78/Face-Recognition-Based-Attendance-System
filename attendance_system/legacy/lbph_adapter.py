from __future__ import annotations


class LegacyLBPHAdapter:
    """Compatibility-only adapter for legacy Haar/LBPH rollback paths.

    This module is intentionally isolated so LBPH/Haar dependencies are not part
    of the default runtime path.
    """

    def __init__(self):
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("OpenCV is required only when using legacy LBPH rollback adapter.") from exc

        self._cv2 = cv2
        if not hasattr(self._cv2, "face"):
            raise RuntimeError("opencv-contrib-python is required for legacy LBPH support.")
        self._recognizer = self._cv2.face.LBPHFaceRecognizer_create()

    def load_model(self, model_path: str) -> None:
        self._recognizer.read(model_path)

    def predict(self, gray_face):
        return self._recognizer.predict(gray_face)

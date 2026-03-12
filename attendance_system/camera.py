from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


class CameraError(RuntimeError):
    """Raised when camera resources cannot be used safely."""


@dataclass(slots=True)
class CameraConfig:
    device_index: int = 0
    width: int = 640
    height: int = 480


class CameraManager:
    """Thin camera wrapper with consistent error handling and logging."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._capture: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        LOGGER.info("Opening camera device %s", self.config.device_index)
        self._capture = cv2.VideoCapture(self.config.device_index)
        if not self._capture.isOpened():
            raise CameraError(f"Unable to open camera device {self.config.device_index}")
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)

    def read_frame(self) -> np.ndarray:
        if self._capture is None:
            raise CameraError("Camera is not opened")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CameraError("Camera disconnected or failed to read a frame")
        return frame

    def release(self) -> None:
        if self._capture is not None:
            LOGGER.info("Releasing camera device %s", self.config.device_index)
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "CameraManager":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
        cv2.destroyAllWindows()


def run_webcam_check(cascade_path: str, config: CameraConfig | None = None) -> None:
    """Run a simple preview with face rectangles until user presses q."""
    config = config or CameraConfig()
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise FileNotFoundError(f"Could not load cascade classifier: {cascade_path}")

    with CameraManager(config) as camera:
        while True:
            frame = camera.read_frame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
            )
            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (10, 159, 255), 2)
            cv2.imshow("Webcam Check", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                LOGGER.info("Webcam check terminated by user")
                break

"""Video stream ingestion with retry, sampling, and async queueing."""

from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
import time
from typing import Any, Optional



@dataclass(slots=True)
class StreamConfig:
    """Configuration for stream ingestion."""

    source: str = "0"
    target_fps: float = 6.0
    buffer_size: int = 32
    reconnect_attempts: int = 5
    reconnect_backoff_seconds: float = 2.0


@dataclass(slots=True)
class FramePacket:
    """Container holding a sampled frame and metadata."""

    frame: object
    timestamp: float
    frame_number: int


class VideoStreamSource:
    """Async frame producer supporting webcam, file, and RTSP streams."""

    def __init__(self, config: StreamConfig):
        self.config = config
        self._capture: Optional[Any] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._queue: queue.Queue[FramePacket] = queue.Queue(maxsize=max(1, config.buffer_size))
        self._frame_number = 0
        self._last_emit_time = 0.0

    @property
    def is_network_source(self) -> bool:
        source = self.config.source.lower().strip()
        return source.startswith("rtsp://") or source.startswith("http://") or source.startswith("https://")

    def start(self) -> None:
        """Open stream and start capture thread."""

        if self._running:
            return
        self._open_capture()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, name="video-stream-source", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop capture thread and release resources."""

        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._thread = None
        self._release_capture()

    def read(self, timeout: float = 1.0) -> Optional[FramePacket]:
        """Read the next sampled frame from the async buffer."""

        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _capture_loop(self) -> None:
        sample_interval = 1.0 / self.config.target_fps if self.config.target_fps > 0 else 0.0

        while self._running:
            if self._capture is None or not self._capture.isOpened():
                if not self._handle_drop():
                    break

            ok, frame = self._capture.read() if self._capture is not None else (False, None)
            if not ok:
                if not self._handle_drop():
                    break
                continue

            now = time.monotonic()
            if sample_interval > 0 and (now - self._last_emit_time) < sample_interval:
                continue

            self._frame_number += 1
            self._last_emit_time = now
            self._enqueue(FramePacket(frame=frame, timestamp=time.time(), frame_number=self._frame_number))

    def _enqueue(self, packet: FramePacket) -> None:
        """Queue newest frame, dropping oldest frame if buffer is full."""

        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        self._queue.put_nowait(packet)

    def _handle_drop(self) -> bool:
        """Reconnect dropped stream depending on source type."""

        self._release_capture()

        if not self.is_network_source:
            self._running = False
            return False

        for _ in range(max(1, self.config.reconnect_attempts)):
            if not self._running:
                return False
            time.sleep(max(0.0, self.config.reconnect_backoff_seconds))
            try:
                self._open_capture()
                if self._capture is not None and self._capture.isOpened():
                    return True
            except RuntimeError:
                continue

        self._running = False
        return False

    def _release_capture(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None

    def _open_capture(self) -> None:
        source = self._parse_source(self.config.source)
        try:
            import cv2
        except ModuleNotFoundError as exc:
            raise RuntimeError("OpenCV is required for VideoStreamSource (pip install opencv-python)") from exc

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Unable to open source: {self.config.source}")
        self._capture = cap

    @staticmethod
    def _parse_source(source: str) -> int | str:
        source = source.strip()
        return int(source) if source.isdigit() else source

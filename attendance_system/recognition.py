from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2

from .attendance_store import AttendanceRecord, load_students, write_attendance
from .camera import CameraConfig, CameraManager

LOGGER = logging.getLogger(__name__)


class RecognitionError(RuntimeError):
    """Raised when recognition cannot proceed."""


@dataclass(slots=True)
class RecognitionConfig:
    confidence_threshold: float = 67.0
    detect_scale_factor: float = 1.2
    min_neighbors: int = 5


def run_recognition_session(
    model_path: Path,
    cascade_path: Path,
    student_csv: Path,
    output_csv: Path,
    camera_config: CameraConfig | None = None,
    config: RecognitionConfig | None = None,
) -> Path:
    cfg = config or RecognitionConfig()
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError as exc:
        raise RecognitionError("OpenCV LBPH recognizer is unavailable. Install opencv-contrib-python.") from exc

    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RecognitionError(f"Could not load cascade classifier: {cascade_path}")

    students = load_students(student_csv)
    if students.empty:
        raise RecognitionError("Student database is empty")

    try:
        recognizer.read(str(model_path))
    except cv2.error as exc:
        LOGGER.exception("Failed to load trained model")
        raise RecognitionError("Model load failure") from exc

    records: list[AttendanceRecord] = []
    font = cv2.FONT_HERSHEY_SIMPLEX

    with CameraManager(camera_config or CameraConfig()) as camera:
        while True:
            frame = camera.read_frame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            min_w = int(0.1 * frame.shape[1])
            min_h = int(0.1 * frame.shape[0])
            faces = detector.detectMultiScale(
                gray,
                scaleFactor=cfg.detect_scale_factor,
                minNeighbors=cfg.min_neighbors,
                minSize=(min_w, min_h),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )

            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (10, 159, 255), 2)
                predicted_id, conf = recognizer.predict(gray[y : y + h, x : x + w])
                confidence = 100 - conf
                student = students.loc[students["Id"] == predicted_id, "Name"]
                name = student.iloc[0] if not student.empty else "Unknown"

                text = f"{predicted_id}-{name}"
                if confidence >= cfg.confidence_threshold and name != "Unknown":
                    now = dt.datetime.now()
                    records.append(
                        AttendanceRecord(
                            person_id=int(predicted_id),
                            name=str(name),
                            date=now.strftime("%Y-%m-%d"),
                            time=now.strftime("%H:%M:%S"),
                        )
                    )
                    text += " [Pass]"

                cv2.putText(frame, text, (x + 5, y - 5), font, 0.9, (255, 255, 255), 2)
                cv2.putText(frame, f"{confidence:.0f}%", (x + 5, y + h - 5), font, 0.8, (0, 255, 0), 1)

            cv2.imshow("Attendance", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if not records:
        LOGGER.warning("Recognition session completed with zero attendance hits")

    return write_attendance(output_csv, records)

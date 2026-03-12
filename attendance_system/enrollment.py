from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

from .attendance_store import append_student
from .camera import CameraConfig, CameraManager

LOGGER = logging.getLogger(__name__)


class EnrollmentError(RuntimeError):
    """Raised when enrollment or training cannot be completed."""


def validate_person(person_id: int, name: str) -> None:
    if person_id < 0:
        raise ValueError("person_id must be non-negative")
    if not name.isalpha():
        raise ValueError("name must only contain alphabetic characters")


def capture_face_samples(
    person_id: int,
    name: str,
    cascade_path: Path,
    image_dir: Path,
    student_csv: Path,
    max_samples: int = 100,
    camera_config: CameraConfig | None = None,
) -> int:
    validate_person(person_id, name)
    image_dir.mkdir(parents=True, exist_ok=True)

    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise EnrollmentError(f"Could not load cascade classifier: {cascade_path}")

    samples = 0
    try:
        with CameraManager(camera_config or CameraConfig()) as camera:
            while True:
                frame = camera.read_frame()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(
                    gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
                )
                for x, y, w, h in faces:
                    samples += 1
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (10, 159, 255), 2)
                    out_file = image_dir / f"{name}.{person_id}.{samples}.jpg"
                    cv2.imwrite(str(out_file), gray[y : y + h, x : x + w])
                cv2.imshow("Enrollment", frame)
                if cv2.waitKey(100) & 0xFF == ord("q"):
                    break
                if samples >= max_samples:
                    break
    except Exception as exc:
        LOGGER.exception("Face enrollment failed")
        raise EnrollmentError("Could not complete enrollment") from exc

    if samples == 0:
        raise EnrollmentError("No faces detected; cannot enroll person")

    append_student(student_csv, person_id, name)
    LOGGER.info("Captured %s samples for %s (%s)", samples, name, person_id)
    return samples


def get_images_and_labels(path: Path) -> Tuple[list[np.ndarray], list[int]]:
    if not path.exists():
        raise FileNotFoundError(f"Training image directory not found: {path}")

    image_paths = sorted(p for p in path.iterdir() if p.is_file())
    if not image_paths:
        raise EnrollmentError("Training image directory is empty")

    faces: list[np.ndarray] = []
    ids: list[int] = []
    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert("L")
            image_np = np.array(image, "uint8")
            person_id = int(image_path.name.split(".")[1])
        except Exception as exc:
            LOGGER.warning("Skipping malformed training file %s: %s", image_path, exc)
            continue
        faces.append(image_np)
        ids.append(person_id)

    if not faces:
        raise EnrollmentError("No valid training images found")
    return faces, ids


def train_images(training_images_dir: Path, model_path: Path) -> Path:
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError as exc:
        raise EnrollmentError("OpenCV LBPH recognizer is unavailable. Install opencv-contrib-python.") from exc

    faces, ids = get_images_and_labels(training_images_dir)
    try:
        recognizer.train(faces, np.array(ids))
        model_path.parent.mkdir(parents=True, exist_ok=True)
        recognizer.save(str(model_path))
    except Exception as exc:
        LOGGER.exception("Failed to train face recognizer")
        raise EnrollmentError("Model training failed") from exc

    LOGGER.info("Training finished. Model saved to %s", model_path)
    return model_path

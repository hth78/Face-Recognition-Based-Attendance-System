from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .attendance_store import build_attendance_filename, latest_attendance_report
from .camera import CameraConfig, run_webcam_check
from .enrollment import capture_face_samples, train_images
from .recognition import run_recognition_session

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Face Recognition Attendance System")
    parser.add_argument("mode", choices=["enroll", "recognize", "report"], help="Execution mode")
    parser.add_argument("--base-dir", default="data", help="Base data directory")
    parser.add_argument("--cascade", default="haarcascade_frontalface_default.xml", help="Path to Haar cascade XML")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index")
    parser.add_argument("--person-id", type=int, help="Person ID for enrollment")
    parser.add_argument("--name", help="Person name for enrollment")
    parser.add_argument("--max-samples", type=int, default=100, help="Samples to capture during enrollment")
    parser.add_argument("--webcam-check", action="store_true", help="Preview camera with face boxes before operation")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    return parser


def run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    base = Path(args.base_dir)
    training_dir = base / "TrainingImages"
    student_csv = base / "StudentDetails" / "StudentDetails.csv"
    model_path = base / "TrainingImageLabel" / "Trainner.yml"
    attendance_dir = base / "Attendance"
    cascade_path = Path(args.cascade)
    camera_cfg = CameraConfig(device_index=args.camera_index)

    if args.webcam_check:
        run_webcam_check(str(cascade_path), camera_cfg)

    if args.mode == "enroll":
        if args.person_id is None or not args.name:
            raise ValueError("--person-id and --name are required for enroll mode")
        samples = capture_face_samples(
            person_id=args.person_id,
            name=args.name,
            cascade_path=cascade_path,
            image_dir=training_dir,
            student_csv=student_csv,
            max_samples=args.max_samples,
            camera_config=camera_cfg,
        )
        train_images(training_images_dir=training_dir, model_path=model_path)
        LOGGER.info("Enrollment complete with %s samples", samples)
        return 0

    if args.mode == "recognize":
        output_file = build_attendance_filename(attendance_dir)
        written = run_recognition_session(
            model_path=model_path,
            cascade_path=cascade_path,
            student_csv=student_csv,
            output_csv=output_file,
            camera_config=camera_cfg,
        )
        LOGGER.info("Attendance captured in %s", written)
        return 0

    if args.mode == "report":
        report_file = latest_attendance_report(attendance_dir)
        report = pd.read_csv(report_file)
        print(f"Report: {report_file}")
        print(report.to_string(index=False) if not report.empty else "No attendance records.")
        return 0

    raise ValueError(f"Unsupported mode: {args.mode}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except Exception:
        LOGGER.exception("Application failed")
        return 1

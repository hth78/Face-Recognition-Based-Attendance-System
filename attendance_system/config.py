"""Runtime configuration for the attendance system."""

from __future__ import annotations

from dataclasses import dataclass
import argparse


@dataclass(slots=True)
class AppConfig:
    """Top-level runtime configuration."""

    camera_source: str = "0"
    target_fps: float = 6.0
    buffer_size: int = 32
    reconnect_attempts: int = 5
    reconnect_backoff_seconds: float = 2.0
    cooldown_seconds: float = 30.0



def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser with stream-related options."""

    parser = argparse.ArgumentParser(description="Face-attendance stream runtime options")
    parser.add_argument(
        "--camera-source",
        default="0",
        help="Camera source: webcam index (0), local video file path, or RTSP URL",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=6.0,
        help="Frame sampling rate used before recognition",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=32,
        help="Max frame queue size for async processing",
    )
    parser.add_argument(
        "--reconnect-attempts",
        type=int,
        default=5,
        help="Reconnect attempts for dropped remote streams",
    )
    parser.add_argument(
        "--reconnect-backoff-seconds",
        type=float,
        default=2.0,
        help="Seconds between reconnect attempts",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=30.0,
        help="Per-person attendance cooldown to prevent duplicates",
    )
    return parser



def parse_args(argv: list[str] | None = None) -> AppConfig:
    """Parse command-line arguments into AppConfig."""

    args = build_parser().parse_args(argv)
    return AppConfig(
        camera_source=args.camera_source,
        target_fps=args.target_fps,
        buffer_size=args.buffer_size,
        reconnect_attempts=args.reconnect_attempts,
        reconnect_backoff_seconds=args.reconnect_backoff_seconds,
        cooldown_seconds=args.cooldown_seconds,
    )

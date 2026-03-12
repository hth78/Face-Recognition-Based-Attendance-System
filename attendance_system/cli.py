"""CLI entrypoint wiring runtime config to stream source and cooldown tracking."""

from __future__ import annotations

from attendance_system.attendance.cooldown import CooldownTracker
from attendance_system.config import parse_args
from attendance_system.stream.source import StreamConfig, VideoStreamSource


def main() -> int:
    """Minimal runtime that demonstrates stream + cooldown controls."""

    config = parse_args()
    stream = VideoStreamSource(
        StreamConfig(
            source=config.camera_source,
            target_fps=config.target_fps,
            buffer_size=config.buffer_size,
            reconnect_attempts=config.reconnect_attempts,
            reconnect_backoff_seconds=config.reconnect_backoff_seconds,
        )
    )
    cooldown = CooldownTracker(cooldown_seconds=config.cooldown_seconds)

    stream.start()
    print(
        "Stream started "
        f"(source={config.camera_source}, target_fps={config.target_fps}, buffer_size={config.buffer_size})"
    )
    print("Press Ctrl+C to stop")

    try:
        while True:
            packet = stream.read(timeout=1.0)
            if packet is None:
                continue

            # Example integration point: recognition model should provide person_id.
            person_id = "demo-user"
            if cooldown.should_mark(person_id):
                print(f"Attendance event for {person_id} on frame #{packet.frame_number}")
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

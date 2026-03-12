"""Cooldown tracker utilities used during attendance marking."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic


@dataclass(slots=True)
class CooldownTracker:
    """Tracks when a person's attendance was last recorded."""

    cooldown_seconds: float
    _last_seen: dict[str, float] = field(default_factory=dict)

    def should_mark(self, person_id: str, now: float | None = None) -> bool:
        """Return True only if this person is outside cooldown interval."""

        current = monotonic() if now is None else now
        previous = self._last_seen.get(person_id)

        if previous is not None and current - previous < self.cooldown_seconds:
            return False

        self._last_seen[person_id] = current
        return True

    def reset(self, person_id: str | None = None) -> None:
        """Reset one person or all cooldown state."""

        if person_id is None:
            self._last_seen.clear()
            return
        self._last_seen.pop(person_id, None)

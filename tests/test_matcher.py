import numpy as np

from attendance_system.vision.matcher import FaceMatcher


def test_matcher_returns_best_match_when_above_threshold():
    matcher = FaceMatcher(threshold=0.7)
    probe = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    enrolled = {
        "emp_a": np.array([0.2, 0.9, 0.0], dtype=np.float32),
        "emp_b": np.array([0.9, 0.1, 0.0], dtype=np.float32),
    }

    result = matcher.match(probe, enrolled)

    assert result.matched is True
    assert result.employee_id == "emp_b"
    assert result.score > 0.7


def test_matcher_rejects_when_below_threshold():
    matcher = FaceMatcher(threshold=0.99)
    probe = np.array([1.0, 0.0], dtype=np.float32)
    enrolled = {"emp_a": np.array([0.5, 0.5], dtype=np.float32)}

    result = matcher.match(probe, enrolled)

    assert result.matched is False
    assert result.employee_id is None

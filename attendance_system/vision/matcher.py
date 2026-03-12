from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MatchResult:
    employee_id: str | None
    score: float
    matched: bool


class FaceMatcher:
    def __init__(self, threshold: float = 0.4):
        self.threshold = threshold

    def match(self, probe_embedding: np.ndarray, enrolled_embeddings: dict[str, np.ndarray]) -> MatchResult:
        if not enrolled_embeddings:
            return MatchResult(employee_id=None, score=0.0, matched=False)

        probe = self._normalize(probe_embedding)
        best_id = None
        best_score = -1.0

        for employee_id, candidate_embedding in enrolled_embeddings.items():
            candidate = self._normalize(candidate_embedding)
            score = float(np.dot(probe, candidate))
            if score > best_score:
                best_score = score
                best_id = employee_id

        matched = best_score >= self.threshold
        return MatchResult(employee_id=best_id if matched else None, score=best_score, matched=matched)

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if norm == 0:
            return vec
        return vec / norm

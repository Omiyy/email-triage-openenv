from __future__ import annotations

from typing import Any


def SAFE_SCORE(score: float) -> float:
    try:
        score = float(score)
    except:
        return 0.5

    if score <= 0.05:
        return 0.05

    if score >= 0.95:
        return 0.95

    return score


def clamp_score(score: float) -> float:
    """Alias for SAFE_SCORE for backward compatibility."""
    return SAFE_SCORE(score)


def safe_ratio_score(correct: int, total: int) -> float:
    if total == 0:
        score = 0.05
    else:
        score = correct / total
    return SAFE_SCORE(score)


def _is_score_like_key(key: str | None) -> bool:
    if not key:
        return False

    lowered = key.lower()
    score_keywords = (
        "score",
        "scores",
        "confidence",
        "probability",
        "similarity",
        "metric",
        "accuracy",
        "reward",
    )
    return any(keyword in lowered for keyword in score_keywords)


def sanitize_response_payload(payload: Any) -> Any:
    """Recursively clamp all numeric values to a safe open-interval band."""

    def _sanitize(value: Any) -> Any:
        if isinstance(value, bool):
            return value

        if isinstance(value, dict):
            return {child_key: _sanitize(child_value) for child_key, child_value in value.items()}

        if isinstance(value, list):
            return [_sanitize(item) for item in value]

        if isinstance(value, tuple):
            return tuple(_sanitize(item) for item in value)

        if isinstance(value, (int, float)):
            numeric = float(value)
            if numeric <= 0.05:
                return 0.05
            if numeric >= 0.95:
                return 0.95
            return numeric

        return value

    return _sanitize(payload)

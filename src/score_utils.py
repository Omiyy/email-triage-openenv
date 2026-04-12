from __future__ import annotations

from typing import Any


def SAFE_SCORE(score: float) -> float:
    try:
        score = float(score)
    except:
        return 0.01

    if score <= 0:
        return 0.01

    if score >= 1:
        return 0.99

    return score


def clamp_score(score: float) -> float:
    """Alias for SAFE_SCORE for backward compatibility."""
    return SAFE_SCORE(score)


def safe_ratio_score(correct: int, total: int) -> float:
    if total == 0:
        score = 0.01
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
    """Recursively clamp score-like numeric values while leaving counters and ids intact."""

    def _sanitize(value: Any, key: str | None = None, in_score_context: bool = False) -> Any:
        current_score_context = in_score_context or _is_score_like_key(key)

        if isinstance(value, dict):
            return {
                child_key: _sanitize(child_value, str(child_key), current_score_context)
                for child_key, child_value in value.items()
            }

        if isinstance(value, list):
            return [_sanitize(item, key, current_score_context) for item in value]

        if isinstance(value, tuple):
            return tuple(_sanitize(item, key, current_score_context) for item in value)

        if isinstance(value, bool):
            return value

        if isinstance(value, (int, float)) and current_score_context:
            numeric = float(value)
            if numeric <= 0:
                return 0.001
            if numeric >= 1:
                return 0.999
            return numeric

        return value

    return _sanitize(payload)

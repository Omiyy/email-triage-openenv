from __future__ import annotations


def clamp_score(score: float) -> float:
    return max(0.01, min(0.99, score))


def safe_ratio_score(correct: int, total: int) -> float:
    if total <= 0:
        return 0.01
    return clamp_score(correct / total)

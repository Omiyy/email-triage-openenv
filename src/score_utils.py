from __future__ import annotations


def clamp_score(score: float) -> float:
    if score is None:
        return 0.5

    epsilon = 1e-6

    if score <= 0:
        return 0.01 + epsilon
    if score >= 1:
        return 0.99 - epsilon

    return max(0.01 + epsilon, min(0.99 - epsilon, score))


def safe_ratio_score(correct: int, total: int) -> float:
    if total == 0:
        score = 0.01
    else:
        score = correct / total
    return clamp_score(score)

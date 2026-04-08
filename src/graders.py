from __future__ import annotations

from src.models import Action, EmailRecord
from src.score_utils import safe_ratio_score
from src.tasks import TaskConfig


def clamp_score(score: float) -> float:
    if score <= 0:
        score = 0.01
    elif score >= 1:
        score = 0.99

    assert 0 < score < 1, f"Invalid score detected: {score}"
    print(f"[DEBUG SCORE] {score}")
    return score


def safe_score(correct: int, total: int) -> float:
    if total == 0:
        return clamp_score(0.01)

    score = correct / total
    return clamp_score(score)


class DeterministicTriageGrader:
    def __init__(self, task: TaskConfig) -> None:
        self.task = task
        self.correct = 0
        self.total = 0

    def update(self, action: Action, truth: EmailRecord) -> None:
        if self.task.require_category:
            self.total += 1
            if action.category == truth.category:
                self.correct += 1

        if self.task.require_priority:
            self.total += 1
            if action.priority == truth.priority:
                self.correct += 1

        if self.task.require_action:
            self.total += 1
            if action.action == truth.action:
                self.correct += 1

        if self.task.require_reply_template:
            self.total += 1
            if action.reply_template == truth.reply_template:
                self.correct += 1

    def score(self) -> float:
        return safe_ratio_score(correct=self.correct, total=self.total)

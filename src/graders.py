from __future__ import annotations

from src.models import Action, EmailRecord
from src.score_utils import SAFE_SCORE, safe_ratio_score
from src.tasks import TaskConfig


def safe_score(correct: int, total: int) -> float:
    score = safe_ratio_score(correct=correct, total=total)
    return SAFE_SCORE(score)


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
        score = safe_ratio_score(correct=self.correct, total=self.total)
        return SAFE_SCORE(score)

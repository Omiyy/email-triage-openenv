from __future__ import annotations

from src.models import Action, EmailRecord
from src.tasks import TaskConfig


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
        if self.total == 0:
            return 0.0
        return self.correct / self.total

from __future__ import annotations

from typing import Dict

from pydantic import BaseModel


class TaskConfig(BaseModel):
    task_id: str
    name: str
    description: str
    require_category: bool
    require_priority: bool
    require_action: bool
    require_reply_template: bool


TASKS: Dict[str, TaskConfig] = {
    "task_easy": TaskConfig(
        task_id="task_easy",
        name="Email Category Classification",
        description="Predict category only.",
        require_category=True,
        require_priority=False,
        require_action=False,
        require_reply_template=False,
    ),
    "task_medium": TaskConfig(
        task_id="task_medium",
        name="Category and Priority Assignment",
        description="Predict category and priority.",
        require_category=True,
        require_priority=True,
        require_action=False,
        require_reply_template=False,
    ),
    "task_hard": TaskConfig(
        task_id="task_hard",
        name="Full Email Triage",
        description="Predict category, priority, action, and reply template.",
        require_category=True,
        require_priority=True,
        require_action=True,
        require_reply_template=True,
    ),
}


def get_task_config(task_id: str) -> TaskConfig:
    if task_id not in TASKS:
        available = ", ".join(sorted(TASKS.keys()))
        raise ValueError(f"Unknown task_id '{task_id}'. Available: {available}")
    return TASKS[task_id]

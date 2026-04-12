from __future__ import annotations

from src.models import Action, EmailRecord, Reward, TriageAction
from src.score_utils import clamp_score
from src.tasks import TaskConfig


CATEGORY_REWARD = 0.3
PRIORITY_REWARD = 0.2
ACTION_REWARD = 0.2
REPLY_REWARD = 0.3

WRONG_CLASSIFICATION_PENALTY = 0.0
UNNECESSARY_ESCALATION_PENALTY = 0.0


def compute_step_reward(action: Action, truth: EmailRecord, task: TaskConfig) -> Reward:
    category_component = 0.0
    priority_component = 0.0
    action_component = 0.0
    reply_component = 0.0
    penalties: dict[str, float] = {}

    if task.require_category:
        if action.category == truth.category:
            category_component = CATEGORY_REWARD
        else:
            penalties["wrong_classification"] = WRONG_CLASSIFICATION_PENALTY

    if task.require_priority and action.priority == truth.priority:
        priority_component = PRIORITY_REWARD

    if task.require_action and action.action == truth.action:
        action_component = ACTION_REWARD

    if task.require_reply_template and action.reply_template == truth.reply_template:
        reply_component = REPLY_REWARD

    if task.require_action and action.action == TriageAction.ESCALATE and truth.action != TriageAction.ESCALATE:
        penalties["unnecessary_escalation"] = UNNECESSARY_ESCALATION_PENALTY

    raw_total = category_component + priority_component + action_component + reply_component + sum(penalties.values())
    total = clamp_score(raw_total)

    return Reward(
        total=total,
        category_component=category_component,
        priority_component=priority_component,
        action_component=action_component,
        reply_component=reply_component,
        penalties=penalties,
    )

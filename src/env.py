from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from src.dataset import load_synthetic_email_dataset
from src.graders import DeterministicTriageGrader
from src.score_utils import SAFE_SCORE
from src.models import Action, Observation, State
from src.rewards import compute_step_reward
from src.tasks import TaskConfig, get_task_config


class EmailTriageEnv:
    def __init__(self, task_id: str) -> None:
        self.task: TaskConfig = get_task_config(task_id)
        self.dataset = load_synthetic_email_dataset()
        self.grader = DeterministicTriageGrader(self.task)
        self.index = 0
        self.cumulative_reward = 0.0
        self.last_reward = 0.0
        self.done = False
        
        # Enhanced state tracking
        self.emails_processed = 0
        self.replies_sent = 0
        self.escalations = 0
        self.archived = 0
        self.urgent_handled = 0

    def reset(self) -> Observation:
        self.grader = DeterministicTriageGrader(self.task)
        self.index = 0
        self.cumulative_reward = 0.0
        self.last_reward = 0.0
        self.done = False
        
        # Reset enhanced tracking
        self.emails_processed = 0
        self.replies_sent = 0
        self.escalations = 0
        self.archived = 0
        self.urgent_handled = 0
        
        return self._observation_for_index(self.index)

    def state(self) -> State:
        return State(
            task_id=self.task.task_id,
            current_index=self.index,
            total_emails=len(self.dataset),
            cumulative_reward=self.cumulative_reward,
            last_reward=self.last_reward,
            done=self.done,
        )

    def step(self, action: Action | Dict[str, Any]) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("Environment is done. Call reset() before calling step() again.")

        if not isinstance(action, Action):
            action = Action.model_validate(action)

        current_email = self.dataset[self.index]
        reward_obj = compute_step_reward(action=action, truth=current_email, task=self.task)
        reward = SAFE_SCORE(reward_obj.total)
        self.grader.update(action=action, truth=current_email)

        # Update tracking statistics
        self.emails_processed += 1
        if action.action and action.action.value == "reply":
            self.replies_sent += 1
        elif action.action and action.action.value == "escalate":
            self.escalations += 1
        elif action.action and action.action.value == "archive":
            self.archived += 1
        
        if action.priority and action.priority.value == "urgent":
            self.urgent_handled += 1

        self.last_reward = reward
        self.cumulative_reward += reward

        self.index += 1
        self.done = self.index >= len(self.dataset)

        next_observation = self._terminal_observation() if self.done else self._observation_for_index(self.index)

        info = {
            "task_id": self.task.task_id,
            "email_id": current_email.id,
            "truth": {
                "category": current_email.category.value,
                "priority": current_email.priority.value,
                "action": current_email.action.value,
                "reply_template": current_email.reply_template,
            },
            "reward_breakdown": reward_obj.model_dump(),
            "running_score": self.grader.score(),
            "state": self.state().model_dump(),
            "stats": {
                "emails_processed": self.emails_processed,
                "emails_remaining": len(self.dataset) - self.emails_processed,
                "replies_sent": self.replies_sent,
                "escalations": self.escalations,
                "archived": self.archived,
                "urgent_handled": self.urgent_handled,
            },
        }
        return next_observation, reward, self.done, info

    def final_score(self) -> float:
        return SAFE_SCORE(self.grader.score())

    def _observation_for_index(self, index: int) -> Observation:
        email = self.dataset[index]
        return Observation(email_id=email.id, email_text=email.text, task_id=self.task.task_id)

    def _terminal_observation(self) -> Observation:
        return Observation(email_id="TERMINAL", email_text="", task_id=self.task.task_id)


class OpenEnvEmailTriageEnv:
    """Stateful, single-step email triage environment for external RL agents."""

    def __init__(self) -> None:
        self._base_emails = load_synthetic_email_dataset()
        self.emails = list(self._base_emails)
        self.current_index = 0
        self.done = False

    def reset(self) -> Dict[str, Any]:
        self.emails = list(self._base_emails)
        self.current_index = 0
        self.done = len(self.emails) == 0
        return self._build_observation()

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("Environment is done. Call reset() before step().")

        current_email = self.emails[self.current_index]
        predicted_action = str(action.get("action", "")).strip()
        expected_action = self._expected_action_for_email(current_email)
        correct = predicted_action == expected_action
        reward = SAFE_SCORE(1.0 if correct else -1.0)

        self.current_index += 1
        self.done = self.current_index >= len(self.emails)

        observation = self._build_observation()
        info = {
            "correct": correct,
            "expected_action": expected_action,
            "processed_email_id": self._email_id_to_int(current_email.id),
        }
        return observation, reward, self.done, info

    def state(self) -> Dict[str, Any]:
        return {
            "current_index": self.current_index,
            "total_emails": len(self.emails),
            "done": self.done,
        }

    def _build_observation(self) -> Dict[str, Any]:
        if self.done:
            return {
                "current_email": None,
                "remaining_emails": 0,
            }

        email = self.emails[self.current_index]
        body = email.text
        subject = self._infer_subject(body)
        remaining_emails = len(self.emails) - self.current_index - 1
        return {
            "current_email": {
                "id": self._email_id_to_int(email.id),
                "subject": subject,
                "body": body,
            },
            "remaining_emails": max(0, remaining_emails),
        }

    @staticmethod
    def _email_id_to_int(email_id: str) -> int:
        match = re.search(r"(\d+)$", email_id)
        if not match:
            raise ValueError(f"Invalid dataset email id format: {email_id}")
        return int(match.group(1))

    @staticmethod
    def _expected_action_for_email(email: Any) -> str:
        # Single-action supervision target for each email in this RL loop.
        if email.action.value == "archive":
            return "classify_email"
        if email.action.value == "escalate":
            return "extract_entities"
        return "generate_reply"

    @staticmethod
    def _infer_subject(body: str) -> str:
        words = [part for part in body.strip().split() if part]
        if not words:
            return "Support request"
        return " ".join(words[:8]).strip(" .,!?:;") or "Support request"

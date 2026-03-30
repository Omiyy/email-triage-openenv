from __future__ import annotations

from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, Field


class Category(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    SALES = "sales"
    ACCOUNT = "account"
    COMPLAINT = "complaint"
    SHIPPING = "shipping"
    OTHER = "other"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class TriageAction(str, Enum):
    REPLY = "reply"
    ESCALATE = "escalate"
    ARCHIVE = "archive"


class EmailRecord(BaseModel):
    id: str
    text: str
    category: Category
    priority: Priority
    action: TriageAction
    reply_template: str


class Observation(BaseModel):
    email_id: str
    email_text: str
    task_id: str
    valid_categories: list[str] = Field(default_factory=lambda: [c.value for c in Category])
    valid_priorities: list[str] = Field(default_factory=lambda: [p.value for p in Priority])
    valid_actions: list[str] = Field(default_factory=lambda: [a.value for a in TriageAction])


class Action(BaseModel):
    category: Optional[Category] = None
    priority: Optional[Priority] = None
    action: Optional[TriageAction] = None
    reply_template: Optional[str] = None


class Reward(BaseModel):
    total: float
    category_component: float = 0.0
    priority_component: float = 0.0
    action_component: float = 0.0
    reply_component: float = 0.0
    penalties: Dict[str, float] = Field(default_factory=dict)


class State(BaseModel):
    task_id: str
    current_index: int
    total_emails: int
    cumulative_reward: float
    last_reward: float
    done: bool

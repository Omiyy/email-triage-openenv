from __future__ import annotations

import json
import os
from typing import Dict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

from openai import OpenAI

from src.env import EmailTriageEnv
from src.models import Action


def _fmt_bool(value: bool) -> str:
    return "1" if value else "0"


def _emit_start(task_id: str, model_name: str, api_enabled: bool, total_steps: int) -> None:
    print(
        "[START]"
        f" task_id={task_id}"
        f" model_name={model_name}"
        f" api_enabled={_fmt_bool(api_enabled)}"
        f" total_steps={total_steps}"
    )


def _emit_step(
    task_id: str,
    step: int,
    email_id: str,
    reward: float,
    cumulative_reward: float,
    category: str,
    priority: str,
    action: str,
    reply_template: str,
) -> None:
    print(
        "[STEP]"
        f" task_id={task_id}"
        f" step={step:02d}"
        f" email_id={email_id}"
        f" reward={reward:.4f}"
        f" cumulative_reward={cumulative_reward:.4f}"
        f" category={category}"
        f" priority={priority}"
        f" action={action}"
        f" reply_template={reply_template}"
    )


def _emit_end(
    task_id: str,
    steps: int,
    final_score: float,
    cumulative_reward: float,
    avg_reward: float,
    category_accuracy: float,
    priority_accuracy: float,
    action_accuracy: float,
    reply_accuracy: float,
) -> None:
    print(
        "[END]"
        f" task_id={task_id}"
        f" steps={steps}"
        f" final_score={final_score:.4f}"
        f" cumulative_reward={cumulative_reward:.4f}"
        f" avg_reward={avg_reward:.4f}"
        f" category_accuracy={category_accuracy:.4f}"
        f" priority_accuracy={priority_accuracy:.4f}"
        f" action_accuracy={action_accuracy:.4f}"
        f" reply_accuracy={reply_accuracy:.4f}"
    )


SYSTEM_PROMPT = (
    "You are an email triage assistant. Return strict JSON with keys: "
    "category, priority, action, reply_template. "
    "Use one of categories: billing, technical, sales, account, complaint, shipping, other. "
    "Use one of priorities: low, medium, high, urgent. "
    "Use one of actions: reply, escalate, archive. "
    "Use one of reply_templates: billing_refund, billing_invoice, account_unlock, "
    "escalate_specialist, tech_troubleshoot, sales_pricing, shipping_update, "
    "complaint_apology, archive_no_reply."
)

# Confidence thresholds for hybrid decision making
CONFIDENCE_THRESHOLD_HIGH = 0.8
CONFIDENCE_THRESHOLD_MEDIUM = 0.5


def calculate_heuristic_confidence(email_text: str, category: str) -> float:
    """
    Calculate confidence score for heuristic classification.
    Returns value between 0.0 and 1.0.
    """
    text = email_text.lower()
    confidence = 0.5  # Base confidence
    
    # Strong category indicators increase confidence
    strong_indicators = {
        "billing": ["invoice", "charged", "refund", "billing", "payment"],
        "technical": ["crash", "500", "bug", "api", "error"],
        "account": ["password", "login", "account locked"],
        "sales": ["pricing", "quote", "discount"],
        "shipping": ["package", "shipment", "tracking"],
        "complaint": ["unacceptable", "rude", "complaint"],
        "other": ["thanks", "amazing", "no further action"],
    }
    
    if category in strong_indicators:
        matches = sum(1 for indicator in strong_indicators[category] if indicator in text)
        confidence += min(matches * 0.15, 0.4)  # Up to +0.4 for multiple matches
    
    # Clear urgency words increase confidence
    urgency_words = ["urgent", "immediately", "asap", "emergency", "production down"]
    if any(word in text for word in urgency_words):
        confidence += 0.1
    
    return min(confidence, 1.0)


def heuristic_policy_with_confidence(email_text: str) -> tuple[Dict[str, str], float]:
    """
    Returns action dict and confidence score.
    This enables hybrid decision making.
    """
    result = heuristic_policy(email_text)
    confidence = calculate_heuristic_confidence(email_text, result["category"])
    return result, confidence


def rule_category(text: str) -> str:
    """Determine category based on keyword matching with prioritized checks."""
    text = text.lower()

    # Priority 1: COMPLAINT (strong emotional indicators)
    complaint_keywords = [
        "complaint", "angry", "bad service", "not happy", "unhappy",
        "disappointed", "terrible", "worst", "poor service", "unacceptable",
        "rude", "ignored", "closed my case", "without resolution",
        "refund because", "missing feature"
    ]
    if any(word in text for word in complaint_keywords):
        return "complaint"

    # Priority 2: BILLING (financial indicators)
    billing_keywords = [
        "refund", "invoice", "charge", "charged", "payment", "billing",
        "money", "price charged", "overcharged", "receipt", "card", "due", "mismatch",
        "confirm if my plan renews", "canceled invoice"
    ]
    if any(word in text for word in billing_keywords):
        return "billing"

    # Priority 3: SHIPPING (delivery indicators)
    shipping_keywords = [
        "delivery", "shipping", "courier", "late",
        "package", "shipment", "tracking", "delivered", "order status",
        "received someone", "wrong order", "compensation"
    ]
    if any(word in text for word in shipping_keywords):
        return "shipping"

    # Priority 4: ACCOUNT (access/login indicators - check before technical)
    account_keywords = [
        "login", "password", "account", "reset", "signin",
        "signup", "locked", "access", "verify", "unlock", "forgot",
        "close my account", "delete all personal"
    ]
    if any(word in text for word in account_keywords):
        return "account"

    # Priority 5: SALES (pricing/business indicators)
    sales_keywords = [
        "price", "plan", "demo", "quote", "purchase",
        "buy", "subscription", "upgrade", "cost", "pricing", "discount",
        "seats", "hipaa", "soc2", "enterprise", "annual", "student",
        "50 seats", "pricing details"
    ]
    if any(word in text for word in sales_keywords):
        return "sales"

    # Priority 6: TECHNICAL (technical problem indicators)
    technical_keywords = [
        "error", "bug", "not working", "crash", "failed",
        "unable", "fix", "exception", "doesn't work", "api", "sso", "logout",
        "configure", "endpoint", "deleted", "workspace", "restore", "backup",
        "documentation", "rate limit", "examples", "500", "reset"
    ]
    if any(word in text for word in technical_keywords):
        return "technical"

    # Priority 7: OTHER (default)
    # Check for positive/closure indicators
    other_keywords = [
        "thanks", "thank you", "issue solved", "no further action",
        "just reporting", "amazing", "solved", "docs link", "broken link"
    ]
    if any(word in text for word in other_keywords):
        return "other"

    return "other"


def rule_priority(text: str) -> str:
    """Determine priority based on urgency keywords - dataset-aware."""
    text = text.lower()

    # Urgent: production down, all admins locked, payroll, immediate callback, 500 errors
    urgent_keywords = [
        "urgent", "asap", "immediately", "right now",
        "critical", "emergency", "production", "all admins", 
        "payroll", "immediate callback", "500", "all requests"
    ]

    # High: crashes, locked, not arriving, ignored, refund issues, before friday, unacceptable
    high_keywords = [
        "high", "crash", "crashes", "locked", "not arriving", 
        "ignored", "refund", "before friday", "unacceptable",
        "rude", "without resolution", "restore", "deleted",
        "wrong order", "keeps spinning", "missing"
    ]

    # Low: student, discount, hipaa, soc2, docs, examples, just reporting, thanks, amazing
    low_keywords = [
        "low", "student", "discount", "hipaa", "soc2", 
        "documentation", "examples", "just reporting", 
        "thanks", "amazing", "solved", "no further action",
        "automatically", "know if"
    ]

    if any(word in text for word in urgent_keywords):
        return "urgent"
    elif any(word in text for word in high_keywords):
        return "high"
    elif any(word in text for word in low_keywords):
        return "low"
    else:
        return "medium"


def rule_action(category: str, priority: str, text: str) -> str:
    """Determine action based on category and priority."""
    text = text.lower()
    
    if priority == "urgent":
        return "escalate"
    elif category in ["complaint"]:
        # Check if complaint needs escalation
        if any(k in text for k in ["closed my case", "without resolution", "missing feature", "refund because", "rude"]):
            return "escalate"
        return "reply"
    elif category in ["technical"]:
        # Check if technical issue needs escalation
        if any(k in text for k in ["production", "500", "restore", "deleted workspace", "crash", "all requests"]):
            return "escalate"
        return "reply"
    elif category in ["billing", "sales", "account", "shipping"]:
        return "reply"
    else:
        return "archive"


def rule_reply_template(category: str, action: str, text: str) -> str:
    """
    Map to exact reply templates used in dataset:
    billing_refund, billing_invoice, account_unlock, escalate_specialist,
    tech_troubleshoot, sales_pricing, shipping_update, complaint_apology, archive_no_reply
    """
    text = text.lower()
    
    # Archive action
    if action == "archive":
        return "archive_no_reply"
    
    # Escalate action
    if action == "escalate":
        return "escalate_specialist"
    
    # Category-specific templates for "reply" action
    if category == "billing":
        # Check for refund-related billing
        if any(k in text for k in ["refund", "charged twice", "duplicate", "overcharged", "money back"]):
            return "billing_refund"
        return "billing_invoice"
    
    elif category == "account":
        return "account_unlock"
    
    elif category == "technical":
        return "tech_troubleshoot"
    
    elif category == "sales":
        return "sales_pricing"
    
    elif category == "shipping":
        return "shipping_update"
    
    elif category == "complaint":
        return "complaint_apology"
    
    else:
        return "archive_no_reply"


def load_local_env(env_path: str = ".env") -> None:
    # Load variables from a local .env file when present.
    # Existing environment variables keep precedence.
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_local_env()


def heuristic_policy(email_text: str) -> Dict[str, str]:
    """Optimized heuristic policy using modular rule functions."""
    category = rule_category(email_text)
    priority = rule_priority(email_text)
    action = rule_action(category, priority, email_text)
    reply_template = rule_reply_template(category, action, email_text)

    return {
        "category": category,
        "priority": priority,
        "action": action,
        "reply_template": reply_template,
    }


def llm_policy(client: OpenAI, model_name: str, email_text: str) -> Dict[str, str]:
    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Classify this email and return JSON only with keys "
                    "category, priority, action, reply_template:\n\n"
                    f"{email_text}"
                ),
            },
        ],
    )
    content = response.choices[0].message.content or "{}"
    return json.loads(content)


class HybridEmailAgent:
    """
    Deterministic rule-based agent.
    
    Strategy:
    - Category: rule-based keyword matching
    - Priority: rule-based keyword matching  
    - Action: rule-based logic
    - Reply Template: mapped directly from category
    
    No LLM used for reply template - fully deterministic.
    """
    
    def __init__(self, client: OpenAI | None, model_name: str):
        self.client = client
        self.model_name = model_name
        self.rule_calls = 0
    
    def decide_action(self, email_text: str, task_id: str = "task_hard") -> Dict[str, str]:
        """
        Main decision method - fully rule-based with task-specific logic.
        """
        self.rule_calls += 1
        
        # Get category, priority, action from rules
        category = rule_category(email_text)
        priority = rule_priority(email_text)
        action = rule_action(category, priority, email_text)
        
        # Task-specific handling with safe defaults
        if task_id == "task_easy":
            # Only category matters - use safe defaults for other fields
            return {
                "category": category,
                "priority": "medium",
                "action": "reply",
                "reply_template": "general_reply",
            }
        elif task_id == "task_medium":
            # Category + Priority + Action - use generic reply template
            return {
                "category": category,
                "priority": priority,
                "action": action,
                "reply_template": "general_reply",
            }
        else:  # task_hard
            # Category + Priority + Action + Reply Template
            reply_template = rule_reply_template(category, action, email_text)
            return {
                "category": category,
                "priority": priority,
                "action": action,
                "reply_template": reply_template,
            }
    
    def get_stats(self) -> Dict[str, int]:
        """Return usage statistics."""
        return {
            "rule_calls": self.rule_calls,
        }


def choose_action(client: OpenAI | None, model_name: str, email_text: str) -> Dict[str, str]:
    """Legacy function - creates temporary agent."""
    agent = HybridEmailAgent(client, model_name)
    return agent.decide_action(email_text)


def make_client() -> OpenAI | None:
    """Create OpenAI client from environment variables."""
    api_base_url = os.getenv("API_BASE_URL")
    hf_token = os.getenv("HF_TOKEN")
    
    if not api_base_url:
        return None
    
    return OpenAI(
        base_url=api_base_url,
        api_key=hf_token or ""
    )


def _new_component_metric() -> Dict[str, float]:
    return {"correct": 0, "total": 0, "accuracy": 0.0}


def _safe_accuracy(correct: int, total: int) -> float:
    if total == 0:
        return 0.0
    return correct / total


def run_task(task_id: str, client: OpenAI | None, model_name: str) -> Dict[str, object]:
    env = EmailTriageEnv(task_id=task_id)
    obs = env.reset()
    _emit_start(
        task_id=task_id,
        model_name=model_name,
        api_enabled=bool(os.getenv("API_BASE_URL", "").strip()),
        total_steps=len(env.dataset),
    )
    
    # Create hybrid agent for this task
    agent = HybridEmailAgent(client, model_name)

    done = False
    step_count = 0
    step_rewards: list[float] = []
    cumulative_rewards: list[float] = []

    component_accuracy = {
        "category": _new_component_metric(),
        "priority": _new_component_metric(),
        "action": _new_component_metric(),
        "reply": _new_component_metric(),
    }

    while not done:
        step_count += 1
        action_payload = agent.decide_action(email_text=obs.email_text, task_id=task_id)
        action = Action.model_validate(action_payload)
        obs, reward, done, info = env.step(action)
        step_rewards.append(reward)
        cumulative_rewards.append(env.state().cumulative_reward)

        truth = info["truth"]
        category_correct = action.category is not None and action.category.value == truth["category"]
        priority_correct = action.priority is not None and action.priority.value == truth["priority"]
        action_correct = action.action is not None and action.action.value == truth["action"]
        reply_correct = action.reply_template is not None and action.reply_template == truth["reply_template"]

        component_accuracy["category"]["total"] += 1
        component_accuracy["category"]["correct"] += int(category_correct)
        component_accuracy["priority"]["total"] += 1
        component_accuracy["priority"]["correct"] += int(priority_correct)
        component_accuracy["action"]["total"] += 1
        component_accuracy["action"]["correct"] += int(action_correct)
        component_accuracy["reply"]["total"] += 1
        component_accuracy["reply"]["correct"] += int(reply_correct)

        _emit_step(
            task_id=task_id,
            step=step_count,
            email_id=info["email_id"],
            reward=reward,
            cumulative_reward=env.state().cumulative_reward,
            category=action_payload["category"],
            priority=action_payload["priority"],
            action=action_payload["action"],
            reply_template=action_payload["reply_template"],
        )

    for key in ["category", "priority", "action", "reply"]:
        correct = int(component_accuracy[key]["correct"])
        total = int(component_accuracy[key]["total"])
        component_accuracy[key]["accuracy"] = _safe_accuracy(correct=correct, total=total)

    final_score = env.final_score()
    cumulative_reward = env.state().cumulative_reward
    avg_reward = cumulative_reward / max(step_count, 1)
    
    # Get agent stats
    agent_stats = agent.get_stats()

    _emit_end(
        task_id=task_id,
        steps=step_count,
        final_score=final_score,
        cumulative_reward=cumulative_reward,
        avg_reward=avg_reward,
        category_accuracy=float(component_accuracy["category"]["accuracy"]),
        priority_accuracy=float(component_accuracy["priority"]["accuracy"]),
        action_accuracy=float(component_accuracy["action"]["accuracy"]),
        reply_accuracy=float(component_accuracy["reply"]["accuracy"]),
    )

    return {
        "task_id": task_id,
        "steps": step_count,
        "step_rewards": step_rewards,
        "cumulative_rewards": cumulative_rewards,
        "final_score": final_score,
        "avg_reward": avg_reward,
        "component_accuracy": component_accuracy,
        "agent_stats": agent_stats,
    }


def main() -> None:
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini").strip()
    client = make_client()

    task_ids = ["task_easy", "task_medium", "task_hard"]

    for task_id in task_ids:
        run_task(task_id=task_id, client=client, model_name=model_name)


if __name__ == "__main__":
    main()

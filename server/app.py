from __future__ import annotations

import json
import os
import random
import re
from threading import Lock
from collections import Counter
from typing import Any, Dict, Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.env import EmailTriageEnv, OpenEnvEmailTriageEnv
from src.models import Action


ALLOWED_CATEGORIES = {"billing", "technical", "sales", "account", "complaint", "shipping", "other"}
ALLOWED_PRIORITIES = {"low", "medium", "high", "urgent"}
ALLOWED_ACTIONS = {"reply", "escalate", "archive"}
ALLOWED_REPLY_TEMPLATES = {
    "billing_refund",
    "billing_invoice",
    "account_unlock",
    "escalate_specialist",
    "tech_troubleshoot",
    "sales_pricing",
    "shipping_update",
    "complaint_apology",
    "archive_no_reply",
    "general_reply",
}

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "mistralai/Mistral-7B-Instruct-v0.2"
API_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
LLM_CLASSIFY_SYSTEM_PROMPT = (
    "You are an email triage assistant. "
    "Return strict JSON with keys: category, priority, action, reply_template. "
    "Use categories billing|technical|sales|account|complaint|shipping|other, "
    "priorities low|medium|high|urgent, actions reply|escalate|archive."
)
LLM_REPLY_SYSTEM_PROMPT = (
    "You are a customer support assistant. "
    "Write exactly one concise support reply sentence with a clear next step."
)

URGENT_KEYWORDS = (
    "urgent",
    "asap",
    "immediately",
    "right now",
    "critical",
    "emergency",
    "production",
    "outage",
    "500",
)

app = FastAPI(
    title="Email Triage Hackathon API",
    version="2.0.0",
    description="Strict validation API for email classification, extraction, and reply generation",
)
_TASK_IDS = ("task_easy", "task_medium", "task_hard")


def _new_scoreboard() -> Dict[str, Dict[str, Any]]:
    return {
        task_id: {
            "score": 0.0,
            "steps": 0,
            "cumulative_reward": 0.0,
            "done": False,
        }
        for task_id in _TASK_IDS
    }


_scoreboard = _new_scoreboard()
_scoreboard_status = "idle"
_llm_client: OpenAI | None = None
_rl_env = OpenEnvEmailTriageEnv()
_rl_env_lock = Lock()


class StepRequest(BaseModel):
    action_type: Literal["mark_spam", "mark_important", "skip"]
    email_id: int


def _combine_text(email: str, subject: str | None) -> str:
    if subject:
        return f"{subject}\n{email}"
    return email


def _rule_category(text: str) -> str:
    lowered = text.lower()

    complaint_keywords = [
        "complaint",
        "angry",
        "bad service",
        "not happy",
        "unhappy",
        "disappointed",
        "terrible",
        "worst",
        "poor service",
        "unacceptable",
        "rude",
        "ignored",
        "closed my case",
        "without resolution",
        "refund because",
        "missing feature",
    ]
    if any(word in lowered for word in complaint_keywords):
        return "complaint"

    billing_keywords = [
        "refund",
        "invoice",
        "charge",
        "charged",
        "payment",
        "billing",
        "money",
        "overcharged",
        "receipt",
        "card",
        "due",
        "mismatch",
        "renew",
    ]
    if any(word in lowered for word in billing_keywords):
        return "billing"

    shipping_keywords = [
        "delivery",
        "shipping",
        "courier",
        "late",
        "package",
        "shipment",
        "tracking",
        "delivered",
        "order status",
        "wrong order",
        "compensation",
    ]
    if any(word in lowered for word in shipping_keywords):
        return "shipping"

    account_keywords = [
        "login",
        "password",
        "account",
        "reset",
        "signin",
        "signup",
        "locked",
        "access",
        "verify",
        "unlock",
        "forgot",
        "delete personal",
    ]
    if any(word in lowered for word in account_keywords):
        return "account"

    sales_keywords = [
        "price",
        "plan",
        "demo",
        "quote",
        "purchase",
        "buy",
        "subscription",
        "upgrade",
        "cost",
        "pricing",
        "discount",
        "seats",
        "enterprise",
        "annual",
        "student",
        "soc2",
        "hipaa",
    ]
    if any(word in lowered for word in sales_keywords):
        return "sales"

    technical_keywords = [
        "error",
        "bug",
        "not working",
        "crash",
        "failed",
        "unable",
        "fix",
        "exception",
        "api",
        "sso",
        "logout",
        "endpoint",
        "restore",
        "backup",
        "documentation",
        "rate limit",
        "500",
    ]
    if any(word in lowered for word in technical_keywords):
        return "technical"

    other_keywords = [
        "thanks",
        "thank you",
        "issue solved",
        "no further action",
        "just reporting",
        "amazing",
        "solved",
    ]
    if any(word in lowered for word in other_keywords):
        return "other"

    return "other"


def _rule_priority(text: str) -> str:
    lowered = text.lower()

    high_keywords = [
        "high",
        "crash",
        "locked",
        "not arriving",
        "ignored",
        "refund",
        "before friday",
        "unacceptable",
        "restore",
        "deleted",
        "wrong order",
        "missing",
    ]

    low_keywords = [
        "low",
        "student",
        "discount",
        "hipaa",
        "soc2",
        "documentation",
        "examples",
        "just reporting",
        "thanks",
        "amazing",
        "solved",
        "no further action",
    ]

    if any(word in lowered for word in URGENT_KEYWORDS):
        return "urgent"
    if any(word in lowered for word in high_keywords):
        return "high"
    if any(word in lowered for word in low_keywords):
        return "low"
    return "medium"


def _rule_action(category: str, priority: str, text: str) -> str:
    lowered = text.lower()

    if priority == "urgent":
        return "escalate"
    if category == "complaint":
        if any(k in lowered for k in ["closed my case", "without resolution", "missing feature", "rude"]):
            return "escalate"
        return "reply"
    if category == "technical":
        if any(k in lowered for k in ["production", "500", "restore", "deleted workspace", "crash"]):
            return "escalate"
        return "reply"
    if category in {"billing", "sales", "account", "shipping"}:
        return "reply"
    return "archive"


def _rule_reply_template(category: str, action: str, text: str) -> str:
    lowered = text.lower()
    if action == "archive":
        return "archive_no_reply"
    if action == "escalate":
        return "escalate_specialist"
    if category == "billing":
        if any(k in lowered for k in ["refund", "charged twice", "duplicate", "overcharged", "money back"]):
            return "billing_refund"
        return "billing_invoice"
    if category == "account":
        return "account_unlock"
    if category == "technical":
        return "tech_troubleshoot"
    if category == "sales":
        return "sales_pricing"
    if category == "shipping":
        return "shipping_update"
    if category == "complaint":
        return "complaint_apology"
    return "archive_no_reply"


def _classify_email(email: str, subject: str | None = None) -> Dict[str, str]:
    combined = _combine_text(email=email, subject=subject)
    category = _rule_category(combined)
    priority = _rule_priority(combined)
    action = _rule_action(category=category, priority=priority, text=combined)
    reply_template = _rule_reply_template(category=category, action=action, text=combined)
    return {
        "category": category,
        "priority": priority,
        "action": action,
        "reply_template": reply_template,
    }


def _get_llm_client() -> OpenAI | None:
    global _llm_client
    if not API_TOKEN:
        return None
    if _llm_client is None:
        _llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_TOKEN)
    return _llm_client


def _extract_json_object(content: str) -> Dict[str, Any] | None:
    text = content.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None


def _normalize_llm_action(raw: Dict[str, Any], email: str, subject: str | None = None) -> Dict[str, str]:
    fallback = _classify_email(email=email, subject=subject)
    combined = _combine_text(email=email, subject=subject)

    category = str(raw.get("category", "")).strip().lower()
    priority = str(raw.get("priority", "")).strip().lower()
    action = str(raw.get("action", "")).strip().lower()
    reply_template = str(raw.get("reply_template", "")).strip().lower()

    if category not in ALLOWED_CATEGORIES:
        category = fallback["category"]
    if priority not in ALLOWED_PRIORITIES:
        priority = fallback["priority"]
    if action not in ALLOWED_ACTIONS:
        action = fallback["action"]
    if reply_template not in ALLOWED_REPLY_TEMPLATES:
        reply_template = _rule_reply_template(category=category, action=action, text=combined)

    return {
        "category": category,
        "priority": priority,
        "action": action,
        "reply_template": reply_template,
    }


def _infer_subject(email: str) -> str:
    words = [w for w in re.split(r"\s+", email.strip()) if w]
    if not words:
        return "Support request"
    subject = " ".join(words[:8]).strip(" .,!?:;")
    return subject if subject else "Support request"


def _classify_email_with_llm(email: str, subject: str | None = None) -> tuple[Dict[str, str], bool]:
    client = _get_llm_client()
    if client is None:
        return _classify_email(email=email, subject=subject), False

    user_prompt = (
        "Classify this email and return only JSON with keys category, priority, action, reply_template.\n\n"
        f"subject: {subject or ''}\n"
        f"email: {email}"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=180,
            stream=False,
            messages=[
                {"role": "system", "content": LLM_CLASSIFY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = (response.choices[0].message.content or "").strip()
        parsed = _extract_json_object(content)
        if parsed is None:
            return _classify_email(email=email, subject=subject), False
        return _normalize_llm_action(raw=parsed, email=email, subject=subject), True
    except Exception:
        return _classify_email(email=email, subject=subject), False


def _generate_llm_one_line_reply(email: str, subject: str) -> tuple[str, bool]:
    client = _get_llm_client()
    if client is None:
        return "", False

    prompt = (
        f"subject: {subject}\n"
        f"email: {email}\n"
        "Return one sentence only."
    )
    for _ in range(2):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0.8,
                max_tokens=120,
                stream=False,
                messages=[
                    {"role": "system", "content": LLM_REPLY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            text = (response.choices[0].message.content or "").strip().replace("\n", " ")
            if not text:
                continue
            # Enforce one-line/one-sentence shape.
            first_sentence = re.split(r"(?<=[.!?])\s+", text)[0].strip()
            return (first_sentence if first_sentence else text), True
        except Exception:
            continue
    # If provider calls fail (e.g., quota/billing), fall back to a stochastic one-liner.
    return _fallback_one_line_reply(email=email, subject=subject), False


def _fallback_one_line_reply(email: str, subject: str) -> str:
    lowered = f"{subject} {email}".lower()

    billing_replies = [
        "Sorry for the billing issue; we are reviewing the charge now and will send an update within 24 hours.",
        "Thanks for reporting this billing problem; our team is checking your account and will follow up shortly.",
        "We understand the billing concern and will verify the transaction details before the next update.",
    ]
    tech_replies = [
        "Thanks for flagging this technical issue; our engineers are investigating and we will share progress updates soon.",
        "We are actively troubleshooting this error and will provide an ETA as soon as diagnostics are complete.",
        "Our team is reviewing the service issue now and will follow up with clear next steps shortly.",
    ]
    shipping_replies = [
        "We are checking shipment status with the carrier and will send you a tracking update soon.",
        "Thanks for reporting the delivery issue; we are coordinating with logistics and will follow up shortly.",
        "We will verify your shipment details right away and provide the next update as soon as possible.",
    ]
    account_replies = [
        "We can help with this account issue and will share secure recovery steps after verification.",
        "Thanks for reporting the account problem; we are validating details and will provide next actions soon.",
        "We are reviewing your account request now and will follow up with confirmation shortly.",
    ]
    default_replies = [
        "Thank you for contacting support; we are reviewing your request and will share next steps shortly.",
        "We appreciate your message and will follow up soon with a clear action plan.",
        "Our support team is reviewing this now and will send an update as soon as possible.",
    ]

    if any(k in lowered for k in ("refund", "invoice", "charged", "billing", "payment")):
        pool = billing_replies
    elif any(k in lowered for k in ("error", "500", "crash", "api", "outage", "bug")):
        pool = tech_replies
    elif any(k in lowered for k in ("shipment", "delivery", "tracking", "package")):
        pool = shipping_replies
    elif any(k in lowered for k in ("account", "password", "login", "unlock", "reset")):
        pool = account_replies
    else:
        pool = default_replies

    return random.choice(pool)


def _reply_quality_component(reply: str, email: str, subject: str) -> float:
    if not reply:
        return 0.0

    combined = f"{subject} {email}".lower()
    reply_lower = reply.lower()

    email_tokens = re.findall(r"[a-z0-9']+", combined)
    reply_tokens = re.findall(r"[a-z0-9']+", reply_lower)
    if not reply_tokens:
        return 0.0

    overlap = sum((Counter(reply_tokens) & Counter(email_tokens)).values())
    overlap_ratio = overlap / max(1, len(set(email_tokens)))

    politeness = 0.0
    if any(k in reply_lower for k in ("sorry", "thanks", "thank you", "please", "appreciate")):
        politeness += 0.4
    if any(k in reply_lower for k in ("we will", "i will", "within", "next", "follow up", "update")):
        politeness += 0.4

    length_score = min(1.0, len(reply_tokens) / 16.0)
    raw = 0.45 * min(1.0, overlap_ratio * 5.0) + 0.35 * politeness + 0.20 * length_score
    return round(max(0.0, min(0.3, raw * 0.3)), 4)


def _run_full_task(task_id: str) -> Dict[str, Any]:
    env = EmailTriageEnv(task_id=task_id)
    observation = env.reset()
    done = False
    steps = 0
    llm_calls = 0
    hard_custom_total = 0.0

    while not done:
        # Keep classification deterministic and use LLM for one-line hard-task replies.
        action_payload = _classify_email(email=observation.email_text)
        action = Action.model_validate(action_payload)
        current_email_text = observation.email_text
        observation, _reward, done, info = env.step(action)

        if task_id == "task_hard":
            subject = _infer_subject(current_email_text)
            llm_reply, reply_used = _generate_llm_one_line_reply(email=current_email_text, subject=subject)
            llm_calls += int(reply_used)
            breakdown = info.get("reward_breakdown", {})
            base_without_template = (
                float(breakdown.get("category_component", 0.0))
                + float(breakdown.get("priority_component", 0.0))
                + float(breakdown.get("action_component", 0.0))
                + sum(float(v) for v in (breakdown.get("penalties") or {}).values())
            )
            hard_custom_total += max(
                0.0,
                min(
                    1.0,
                    base_without_template + _reply_quality_component(reply=llm_reply, email=current_email_text, subject=subject),
                ),
            )
        steps += 1

    state = env.state()
    final_score = round(float(env.final_score()), 2)
    if task_id == "task_hard" and steps > 0:
        final_score = round(hard_custom_total / steps, 2)

    return {
        "score": final_score,
        "steps": steps,
        "cumulative_reward": round(float(state.cumulative_reward), 2),
        "llm_calls": llm_calls,
        "done": True,
    }


def _scoreboard_overall() -> float:
    if not _scoreboard:
        return 0.0
    total = sum(float(task_data["score"]) for task_data in _scoreboard.values())
    return round(total / len(_scoreboard), 2)


def _total_email_count() -> int:
    return sum(len(EmailTriageEnv(task_id=task_id).dataset) for task_id in _TASK_IDS)


@app.get("/")
def root() -> Dict[str, str]:
    return {
        "name": app.title,
        "version": app.version,
        "openapi": "/openapi.json",
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
def reset() -> Dict[str, Any]:
    with _rl_env_lock:
        return _rl_env.reset()


@app.post("/step")
def step(payload: StepRequest) -> Dict[str, Any]:
    with _rl_env_lock:
        try:
            observation, reward, done, info = _rl_env.step(payload.model_dump())
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": {
            "correct": bool(info.get("correct", False)),
        },
    }


@app.get("/state")
def state() -> Dict[str, Any]:
    with _rl_env_lock:
        return _rl_env.state()


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()

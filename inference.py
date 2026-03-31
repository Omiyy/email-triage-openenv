from __future__ import annotations

import json
import os
from typing import Dict
from openai import OpenAI

from src.env import EmailTriageEnv
from src.models import Action
from src.visualization import plot_cumulative_rewards, plot_rewards, plot_task_scores, show_plots


SYSTEM_PROMPT = (
    "You are an email triage assistant. Return strict JSON with keys: "
    "category, priority, action, reply_template. "
    "Use one of categories: billing, technical, sales, account, complaint, shipping, other. "
    "Use one of priorities: low, medium, high, urgent. "
    "Use one of actions: reply, escalate, archive."
)


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
    text = email_text.lower()

    category = "other"
    priority = "low"
    triage_action = "archive"
    reply_template = "archive_no_reply"

    if any(k in text for k in ["invoice", "charged", "refund", "card", "billing"]):
        category = "billing"
        priority = "medium"
        triage_action = "reply"
        reply_template = "billing_invoice"
        if "refund" in text or "charged twice" in text:
            priority = "high"
            reply_template = "billing_refund"

    elif any(k in text for k in ["password", "unlock", "account", "login"]):
        category = "account"
        priority = "high"
        triage_action = "reply"
        reply_template = "account_unlock"
        if any(k in text for k in ["urgent", "all admins", "payroll"]):
            priority = "urgent"
            triage_action = "escalate"
            reply_template = "escalate_specialist"

    elif any(k in text for k in ["crash", "500", "bug", "api", "sso", "logout", "reset"]):
        category = "technical"
        priority = "medium"
        triage_action = "reply"
        reply_template = "tech_troubleshoot"
        if any(k in text for k in ["production", "urgent", "500", "restore"]):
            priority = "urgent" if "urgent" in text or "500" in text else "high"
            triage_action = "escalate"
            reply_template = "escalate_specialist"

    elif any(k in text for k in ["pricing", "quote", "discount", "seats", "hipaa", "soc2"]):
        category = "sales"
        priority = "low" if "discount" in text else "medium"
        triage_action = "reply"
        reply_template = "sales_pricing"

    elif any(k in text for k in ["package", "shipment", "tracking", "delayed"]):
        category = "shipping"
        priority = "medium"
        triage_action = "reply"
        reply_template = "shipping_update"
        if any(k in text for k in ["someone else's", "wrong order"]):
            priority = "high"
            triage_action = "escalate"
            reply_template = "escalate_specialist"

    elif any(k in text for k in ["unacceptable", "rude", "ignored", "complaint"]):
        category = "complaint"
        priority = "high"
        triage_action = "reply"
        reply_template = "complaint_apology"
        if any(k in text for k in ["closed my case", "refund", "missing feature"]):
            triage_action = "escalate"
            reply_template = "escalate_specialist"

    elif any(k in text for k in ["thanks", "no further action", "just reporting", "amazing"]):
        category = "other"
        priority = "low"
        triage_action = "archive"
        reply_template = "archive_no_reply"

    return {
        "category": category,
        "priority": priority,
        "action": triage_action,
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


def choose_action(client: OpenAI | None, model_name: str, email_text: str) -> Dict[str, str]:
    if client is None:
        return heuristic_policy(email_text)

    try:
        return llm_policy(client=client, model_name=model_name, email_text=email_text)
    except Exception:
        return heuristic_policy(email_text)


def make_client() -> OpenAI | None:
    api_base_url = os.getenv("API_BASE_URL", "").strip()
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini").strip()
    hf_token = os.getenv("HF_TOKEN", "").strip()

    if not api_base_url:
        return None

    # HF Inference endpoints require a token in api_key field.
    _ = model_name
    return OpenAI(base_url=api_base_url, api_key=hf_token or "dummy")


def _new_component_metric() -> Dict[str, float]:
    return {"correct": 0, "total": 0, "accuracy": 0.0}


def _safe_accuracy(correct: int, total: int) -> float:
    if total == 0:
        return 0.0
    return correct / total


def run_task(task_id: str, client: OpenAI | None, model_name: str) -> Dict[str, object]:
    env = EmailTriageEnv(task_id=task_id)
    obs = env.reset()

    print(f"\n=== Running {task_id} ===")
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
        action_payload = choose_action(client=client, model_name=model_name, email_text=obs.email_text)
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

        breakdown = info["reward_breakdown"]
        print(
            f"Step {step_count:02d} | email={info['email_id']} | reward={reward:+.2f} | "
            f"cat={breakdown['category_component']:+.2f} pri={breakdown['priority_component']:+.2f} "
            f"act={breakdown['action_component']:+.2f} rep={breakdown['reply_component']:+.2f} "
            f"penalties={breakdown['penalties']} | "
            f"ok(cat={int(category_correct)} pri={int(priority_correct)} act={int(action_correct)} rep={int(reply_correct)})"
        )

    for key in ["category", "priority", "action", "reply"]:
        correct = int(component_accuracy[key]["correct"])
        total = int(component_accuracy[key]["total"])
        component_accuracy[key]["accuracy"] = _safe_accuracy(correct=correct, total=total)

    final_score = env.final_score()
    cumulative_reward = env.state().cumulative_reward
    avg_reward = cumulative_reward / max(step_count, 1)

    print(f"Final score ({task_id}): {final_score:.4f}")
    print(f"Cumulative reward ({task_id}): {cumulative_reward:.4f}")
    print(f"Average reward ({task_id}): {avg_reward:.4f}")

    return {
        "task_id": task_id,
        "steps": step_count,
        "step_rewards": step_rewards,
        "cumulative_rewards": cumulative_rewards,
        "final_score": final_score,
        "avg_reward": avg_reward,
        "component_accuracy": component_accuracy,
    }


def print_summary(task_label: str, metrics: Dict[str, object]) -> None:
    component_accuracy = metrics["component_accuracy"]
    print(f"\n{task_label}:")
    print(f"- Final Score: {float(metrics['final_score']):.4f}")
    print(f"- Avg Reward: {float(metrics['avg_reward']):.4f}")
    print(f"- Category Accuracy: {float(component_accuracy['category']['accuracy']) * 100:.2f}%")
    print(f"- Priority Accuracy: {float(component_accuracy['priority']['accuracy']) * 100:.2f}%")
    print(f"- Action Accuracy: {float(component_accuracy['action']['accuracy']) * 100:.2f}%")
    print(f"- Reply Accuracy: {float(component_accuracy['reply']['accuracy']) * 100:.2f}%")


def main() -> None:
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini").strip()
    client = make_client()

    print("Email Triage OpenEnv Inference")
    print(f"API_BASE_URL set: {bool(os.getenv('API_BASE_URL', '').strip())}")
    print(f"MODEL_NAME: {model_name}")
    print(f"HF_TOKEN set: {bool(os.getenv('HF_TOKEN', '').strip())}")

    task_ids = ["task_easy", "task_medium", "task_hard"]
    task_labels = {
        "task_easy": "Task 1 (Easy)",
        "task_medium": "Task 2 (Medium)",
        "task_hard": "Task 3 (Hard)",
    }
    all_metrics: dict[str, Dict[str, object]] = {}

    for task_id in task_ids:
        all_metrics[task_id] = run_task(task_id=task_id, client=client, model_name=model_name)

    print("\n=== Final Summary ===")
    for task_id in task_ids:
        print_summary(task_labels[task_id], all_metrics[task_id])

    for task_id in task_ids:
        label = task_labels[task_id]
        plot_rewards(all_metrics[task_id]["step_rewards"], title=f"{label}: Reward vs Steps")
        plot_cumulative_rewards(
            all_metrics[task_id]["cumulative_rewards"],
            title=f"{label}: Cumulative Reward vs Steps",
        )

    plot_task_scores({task_labels[task_id]: float(all_metrics[task_id]["final_score"]) for task_id in task_ids})
    show_plots()


if __name__ == "__main__":
    main()

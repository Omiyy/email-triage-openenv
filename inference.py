"""
Email Triage Hackathon - Inference Script

This script calls the API endpoints and prints structured logs for evaluation.
"""

from __future__ import annotations

import os
import sys
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests


# Configuration from environment
API_BASE_URL = os.getenv("API_BASE_URL", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# For local testing, use localhost
# For Hugging Face deployment, use the Space URL from environment or default
import sys
if len(sys.argv) > 1 and sys.argv[1].startswith("http"):
    # URL provided as argument
    BASE_URL = sys.argv[1].rstrip("/")
else:
    # Default to localhost for local testing
    BASE_URL = "http://localhost:7860"


class Logger:
    """Structured logger for evaluation."""
    
    def __init__(self, task: str, env: str, model: str):
        self.task = task
        self.env = env
        self.model = model
        self.step_count = 0
        self.rewards: list[float] = []
        self.errors: list[str | None] = []
    
    def start(self) -> None:
        """Print [START] log."""
        print(f"[START] task={self.task} env={self.env} model={self.model}", flush=True)
    
    def step(self, action: str, reward: float, done: bool, error: str | None = None) -> None:
        """Print [STEP] log."""
        self.step_count += 1
        self.rewards.append(reward)
        self.errors.append(error)
        
        error_str = "null" if error is None else f"{error}"
        print(
            f"[STEP] step={self.step_count} action={action} reward={reward:.2f} done={str(done).lower()} error={error_str}",
            flush=True
        )
    
    def end(self, success: bool) -> None:
        """Print [END] log."""
        score = sum(self.rewards)
        rewards_str = ",".join(f"{r:.2f}" for r in self.rewards)
        print(
            f"[END] success={str(success).lower()} steps={self.step_count} score={score:.2f} rewards={rewards_str}",
            flush=True
        )


def make_api_call(endpoint: str, payload: dict) -> tuple[bool, Any]:
    """
    Make API call and return (success, result_or_error).
    """
    url = f"{BASE_URL}{endpoint}"
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        return True, response.json()
    except requests.exceptions.RequestException as e:
        return False, str(e)


def run_inference(email: str) -> None:
    """
    Run the complete inference pipeline with structured logging.
    
    Steps:
    1. Classify email
    2. Extract information
    3. Generate reply
    """
    logger = Logger(
        task="email-triage",
        env="openenv",
        model=MODEL_NAME
    )
    
    # Track if we should mark as successful
    all_success = True
    
    try:
        # Print START
        logger.start()
        
        # Step 1: Classify email
        success, result = make_api_call("/classify", {"email": email})
        if success and "category" in result:
            category = result["category"]
            logger.step("classify_email", 0.33, False, None)
        else:
            category = "other"
            error_msg = result if not success else "missing category"
            logger.step("classify_email", 0.0, False, error_msg)
            all_success = False
        
        # Step 2: Extract information
        success, result = make_api_call("/extract", {"email": email})
        if success and all(k in result for k in ["order_id", "issue", "intent"]):
            logger.step("extract_entities", 0.33, False, None)
        else:
            error_msg = result if not success else "missing fields"
            logger.step("extract_entities", 0.0, False, error_msg)
            all_success = False
        
        # Step 3: Generate reply
        success, result = make_api_call("/suggest", {
            "email": email,
            "category": category
        })
        if success and "response" in result:
            logger.step("generate_reply", 0.34, True, None)
        else:
            error_msg = result if not success else "missing response"
            logger.step("generate_reply", 0.0, True, error_msg)
            all_success = False
        
        # Print END
        logger.end(all_success)
        
    except Exception as e:
        # Ensure END is always printed even on exception
        logger.end(False)
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    # Sample email for testing
    sample_email = """Subject: Request for refund on order #12345

Hi Support Team,

I ordered a laptop last week (Order #12345) but it arrived damaged. 
The screen is cracked and it won't turn on. I would like a full refund.

Please process this as soon as possible.

Thanks,
John Smith"""
    
    # Check if email provided as argument
    if len(sys.argv) > 1:
        # Read email from file if argument is a file path
        email_path = sys.argv[1]
        if os.path.isfile(email_path):
            with open(email_path, 'r') as f:
                email = f.read()
        else:
            email = sys.argv[1]
    else:
        email = sample_email
    
    run_inference(email)


if __name__ == "__main__":
    main()

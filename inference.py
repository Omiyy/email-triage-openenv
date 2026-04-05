"""
Email Triage Hackathon - Inference Script with Strict Logging

This script calls the API endpoints and prints structured logs for evaluation.
Log format is STRICT and must not be changed.
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


# ============================================================================
# CONFIGURATION
# ============================================================================
API_BASE_URL = os.getenv("API_BASE_URL", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Determine base URL
if len(sys.argv) > 1 and sys.argv[1].startswith("http"):
    BASE_URL = sys.argv[1].rstrip("/")
else:
    BASE_URL = "http://localhost:7860"


# ============================================================================
# STRICT LOGGER (DO NOT MODIFY)
# ============================================================================
class StrictLogger:
    """
    Strict logger for hackathon evaluation.
    Output format is FIXED and cannot be changed.
    """
    
    def __init__(self, task: str, env: str, model: str):
        self.task = task
        self.env = env
        self.model = model
        self.step_count = 0
        self.rewards: list[float] = []
        self.has_error = False
    
    def start(self) -> None:
        """Print [START] log - SINGLE LINE ONLY."""
        print(f"[START] task={self.task} env={self.env} model={self.model}", flush=True)
    
    def step(self, action: str, reward: float, done: bool, error: str | None = None) -> None:
        """
        Print [STEP] log - SINGLE LINE ONLY.
        
        Format: [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
        """
        self.step_count += 1
        self.rewards.append(reward)
        
        if error:
            self.has_error = True
            # Escape any special characters in error message
            error_str = error.replace("\n", " ").replace("\r", " ").strip()
            if len(error_str) > 100:
                error_str = error_str[:100] + "..."
        else:
            error_str = "null"
        
        print(
            f"[STEP] step={self.step_count} action={action} reward={reward:.2f} done={str(done).lower()} error={error_str}",
            flush=True
        )
    
    def end(self, success: bool) -> None:
        """
        Print [END] log - SINGLE LINE ONLY - ALWAYS CALLED.
        
        Format: [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
        """
        score = sum(self.rewards)
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, score))
        
        rewards_str = ",".join(f"{r:.2f}" for r in self.rewards)
        
        print(
            f"[END] success={str(success).lower()} steps={self.step_count} score={score:.2f} rewards={rewards_str}",
            flush=True
        )


# ============================================================================
# API CALL FUNCTION
# ============================================================================
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


# ============================================================================
# MAIN INFERENCE PIPELINE
# ============================================================================
def run_inference(email: str) -> None:
    """
    Run the complete inference pipeline with strict logging.
    
    Steps:
    1. Classify email
    2. Extract information
    3. Generate reply
    """
    logger = StrictLogger(
        task="email-triage",
        env="openenv",
        model=MODEL_NAME
    )
    
    # Track success
    all_success = True
    category = "general_inquiry"  # Default
    extracted_data = {}
    
    try:
        # Print START
        logger.start()
        
        # ========================================================================
        # STEP 1: Classify email
        # ========================================================================
        success, result = make_api_call("/classify", {"email": email})
        
        if success and isinstance(result, dict) and "category" in result:
            category = result["category"]
            # Validate category is in valid list
            valid_categories = [
                "refund", "complaint", "order_status", "technical_support",
                "billing", "general_inquiry", "urgent", "spam"
            ]
            if category not in valid_categories:
                category = "general_inquiry"
            logger.step("classify_email", 0.33, False, None)
        else:
            error_msg = str(result) if not success else "missing category"
            logger.step("classify_email", 0.00, False, error_msg)
            all_success = False
        
        # ========================================================================
        # STEP 2: Extract information
        # ========================================================================
        success, result = make_api_call("/extract", {"email": email})
        
        if success and isinstance(result, dict):
            # Validate all required fields exist
            required_fields = ["customer_name", "order_id", "product", "issue", "intent", "urgency"]
            has_all_fields = all(field in result for field in required_fields)
            
            if has_all_fields:
                extracted_data = result
                logger.step("extract_entities", 0.33, False, None)
            else:
                missing = [f for f in required_fields if f not in result]
                logger.step("extract_entities", 0.00, False, f"missing fields: {missing}")
                all_success = False
                # Fill missing fields with null
                for field in required_fields:
                    if field not in extracted_data:
                        extracted_data[field] = None
        else:
            error_msg = str(result) if not success else "invalid response"
            logger.step("extract_entities", 0.00, False, error_msg)
            all_success = False
            # Initialize with nulls
            extracted_data = {
                "customer_name": None,
                "order_id": None,
                "product": None,
                "issue": None,
                "intent": None,
                "urgency": None
            }
        
        # ========================================================================
        # STEP 3: Generate reply
        # ========================================================================
        suggest_payload = {
            "email": email,
            "category": category,
            "extracted": extracted_data
        }
        
        success, result = make_api_call("/suggest", suggest_payload)
        
        if success and isinstance(result, dict) and "response" in result:
            response_text = result["response"]
            # Validate response is not empty and is a string
            if isinstance(response_text, str) and len(response_text.strip()) > 0:
                logger.step("generate_reply", 0.34, True, None)
            else:
                logger.step("generate_reply", 0.00, True, "empty response")
                all_success = False
        else:
            error_msg = str(result) if not success else "missing response"
            logger.step("generate_reply", 0.00, True, error_msg)
            all_success = False
        
        # ========================================================================
        # END: Print final log
        # ========================================================================
        logger.end(all_success)
        
    except Exception as e:
        # Ensure END is always printed even on exception
        logger.end(False)
        sys.exit(1)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
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
    if len(sys.argv) > 1 and not sys.argv[1].startswith("http"):
        # Read email from file if argument is a file path
        email_path = sys.argv[1]
        if os.path.isfile(email_path):
            with open(email_path, 'r', encoding='utf-8') as f:
                email = f.read()
        else:
            email = sys.argv[1]
    else:
        email = sample_email
    
    run_inference(email)


if __name__ == "__main__":
    main()

"""
Email Triage Hackathon API - Strict Validation Version
FastAPI server with validated outputs for hackathon evaluation.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI


# ============================================================================
# ENVIRONMENT VARIABLES
# ============================================================================
API_BASE_URL = os.getenv("API_BASE_URL", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN", "")


# ============================================================================
# VALID CATEGORY LIST (STRICT)
# ============================================================================
VALID_CATEGORIES = [
    "refund",
    "complaint",
    "promotion",
    "order_status",
    "technical_support",
    "billing",
    "general_inquiry",
    "urgent",
    "spam"
]


# ============================================================================
# PYDANTIC MODELS
# ============================================================================
class ClassifyRequest(BaseModel):
    email: str


class ClassifyResponse(BaseModel):
    category: str = Field(..., description="Must be one of the 8 valid categories")


class ExtractRequest(BaseModel):
    email: str


class ExtractResponse(BaseModel):
    customer_name: str | None = None
    order_id: str | None = None
    product: str | None = None
    issue: str | None = None
    intent: str = "unknown"
    urgency: str = "low"


class SuggestRequest(BaseModel):
    email: str
    category: str | None = None
    extracted: ExtractResponse | None = None


class SuggestResponse(BaseModel):
    response: str


# ============================================================================
# OPENENV MODELS
# ============================================================================
class StepRequest(BaseModel):
    action: str  # "classify_email", "extract_entities", "generate_reply"


class StepResponse(BaseModel):
    state: str
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


class StateResponse(BaseModel):
    step: int
    done: bool


class ResetResponse(BaseModel):
    state: str
    step: int
    done: bool
    current_email: dict[str, str]


# ============================================================================
# OPENENV STATE MANAGEMENT
# ============================================================================
class OpenEnvState:
    """Manages the OpenEnv environment state."""
    
    def __init__(self):
        self.step_count = 0
        self.done = False
        self.current_email = ""
        self.category = ""
        self.extracted_data = {}
        self.response = ""
    
    def reset(self) -> dict:
        """Reset the environment state."""
        self.step_count = 0
        self.done = False
        self.current_email = self._load_sample_email()
        self.category = ""
        self.extracted_data = {}
        self.response = ""
        
        return {
            "state": "environment reset",
            "step": self.step_count,
            "done": self.done,
            "current_email": {
                "email": self.current_email
            },
        }
    
    def get_state(self) -> dict:
        """Get current environment state."""
        return {
            "step": self.step_count,
            "done": self.done
        }
    
    def step(self, action: str) -> dict:
        """Execute one step in the environment."""
        if self.done:
            return {
                "state": "environment already done",
                "reward": 0.0,
                "done": True
            }
        
        if action == "classify_email":
            return self._classify_email()
        elif action == "extract_entities":
            return self._extract_entities()
        elif action == "generate_reply":
            return self._generate_reply()
        else:
            return {
                "state": f"invalid action: {action}",
                "reward": 0.0,
                "done": self.done
            }
    
    def _classify_email(self) -> dict:
        """Classify the current email."""
        from app import classify_email_rule_based
        
        self.category = classify_email_rule_based(self.current_email)
        self.step_count += 1
        
        return {
            "state": f"classified as: {self.category}",
            "reward": 0.33,
            "done": False,
            "info": {
                "action": "classify_email",
                "result": self.category,
            },
        }
    
    def _extract_entities(self) -> dict:
        """Extract entities from the current email."""
        from app import rule_based_extract
        
        try:
            self.extracted_data = validate_extraction(rule_based_extract(self.current_email)).model_dump()
        except Exception:
            self.extracted_data = {
                "customer_name": None,
                "order_id": None,
                "product": None,
                "issue": None,
                "intent": "unknown",
                "urgency": "low",
            }
        self.step_count += 1
        
        return {
            "state": f"extracted: {json.dumps(self.extracted_data)}",
            "reward": 0.33,
            "done": False,
            "info": {
                "action": "extract_entities",
                "result": self.extracted_data,
            },
        }
    
    def _generate_reply(self) -> dict:
        """Generate reply for the current email."""
        from app import template_based_suggest, ExtractResponse
        
        if not self.category:
            self.category = "general_inquiry"
        
        if not self.extracted_data:
            self.extracted_data = {
                "customer_name": None,
                "order_id": None,
                "product": None,
                "issue": None,
                "intent": None,
                "urgency": None
            }
        
        extracted = ExtractResponse(**self.extracted_data)
        self.response = template_based_suggest(self.current_email, self.category, extracted)
        self.step_count += 1
        self.done = True
        
        return {
            "state": f"reply generated: {self.response[:100]}...",
            "reward": 0.34,
            "done": True,
            "info": {
                "action": "generate_reply",
                "result": self.response,
            },
        }
    
    def _load_sample_email(self) -> str:
        """Load a sample email for the task."""
        return """Subject: Request for refund on order #12345

Hi Support Team,

I ordered a laptop last week (Order #12345) but it arrived damaged. 
The screen is cracked and it won't turn on. I would like a full refund.

Please process this as soon as possible.

Thanks,
John Smith"""


# Global state instance
env_state = OpenEnvState()


# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(
    title="Email Triage Hackathon API",
    description="Strict validation API for email classification, extraction, and reply generation",
    version="2.0.0"
)


def get_openai_client() -> OpenAI | None:
    """Initialize OpenAI client from environment variables."""
    if not API_BASE_URL:
        return None
    return OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN or ""
    )


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================
def validate_category(output: str) -> str:
    """
    Validate and map category to valid list.
    Returns closest valid category or 'general_inquiry' as default.
    """
    output_clean = output.lower().strip()
    
    # Direct match
    if output_clean in VALID_CATEGORIES:
        return output_clean
    
    # Keyword mapping
    keyword_map = {
        "refund": ["refund", "money back", "return", "reimburse"],
        "complaint": ["complaint", "unhappy", "disappointed", "terrible", "bad", "poor"],
        "order_status": ["order", "status", "tracking", "shipment", "delivery", "package"],
        "technical_support": ["technical", "bug", "error", "crash", "not working", "issue", "problem", "support"],
        "billing": ["billing", "invoice", "charge", "payment", "subscription", "price"],
        "general_inquiry": ["question", "inquiry", "information", "help", "general"],
        "urgent": ["urgent", "asap", "emergency", "critical", "immediately"],
        "spam": ["spam", "promotion", "marketing", "unsubscribe", "advertisement"]
    }
    
    for category, keywords in keyword_map.items():
        if any(keyword in output_clean for keyword in keywords):
            return category
    
    # Default fallback
    return "general_inquiry"


def validate_extraction(data: dict) -> ExtractResponse:
    """
    Validate extraction output and ensure all fields exist.
    Missing fields are set to null.
    Intent is validated against VALID_CATEGORIES.
    Urgency is validated to be "high", "medium", or "low".
    """
    required_fields = ["customer_name", "order_id", "product", "issue", "intent", "urgency"]
    
    validated = {}
    for field in required_fields:
        value = data.get(field)
        # Convert empty strings or missing values to null
        if value is None or value == "" or value == "null":
            validated[field] = None
        else:
            validated[field] = str(value).strip()
    
    # Validate intent against valid categories
    if validated.get("intent"):
        validated["intent"] = validate_category(validated["intent"])
    else:
        validated["intent"] = "unknown"
    
    # Validate urgency
    urgency = str(validated.get("urgency", "")).lower()
    if urgency not in ["high", "medium", "low"]:
        validated["urgency"] = "low"
    else:
        validated["urgency"] = urgency
    
    return ExtractResponse(**validated)


def validate_response(text: str) -> str:
    """
    Validate and trim response to 120 words max.
    Ensures professional tone.
    """
    # Clean up the text
    text = text.strip()
    
    # Split into words and limit to 120
    words = text.split()
    if len(words) > 120:
        text = " ".join(words[:120]) + "..."
    
    return text


# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.get("/")
def root() -> dict:
    """Root endpoint - API info."""
    return {
        "message": "Email Triage Hackathon API is running",
        "version": "2.0.0",
        "endpoints": {
            "classify": "POST /classify",
            "extract": "POST /extract",
            "suggest": "POST /suggest"
        }
    }


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/classify", response_model=ClassifyResponse)
def classify(payload: ClassifyRequest) -> dict:
    """
    Classify email into one of 8 strict categories.
    """
    client = get_openai_client()
    
    if not client:
        # Fallback: rule-based classification
        category = classify_email_rule_based(payload.email)
        return {"category": validate_category(category)}
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=20,
            messages=[
                {
                    "role": "system",
                    "content": f"""Classify the email into exactly one of these categories: {', '.join(VALID_CATEGORIES)}.
Respond with ONLY the category name, nothing else."""
                },
                {"role": "user", "content": f"Email:\n{payload.email}"}
            ]
        )
        
        raw_category = response.choices[0].message.content.strip()
        validated_category = validate_category(raw_category)
        
        return {"category": validated_category}
    
    except Exception as exc:
        # Fallback on error
        category = classify_email_rule_based(payload.email)
        return {"category": validate_category(category)}


@app.post("/extract", response_model=ExtractResponse)
def extract(payload: ExtractRequest) -> dict:
    """
    Extract structured information from email.
    Returns strict JSON with all 6 fields (null if missing).
    """
    try:
        client = get_openai_client()
        if not client:
            data = rule_based_extract(payload.email)
            return validate_extraction(data).model_dump()

        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=150,
            messages=[
                {
                    "role": "system",
                    "content": f"""Extract information from the email and return ONLY a JSON object with these exact keys:
- customer_name: Name of the customer (null if not found)
- order_id: Order ID mentioned (null if not found)
- product: Product mentioned (null if not found)
- issue: Brief description of the issue (null if not clear)
- intent: Must be one of: {', '.join(VALID_CATEGORIES)} (choose the closest match)
- urgency: "high", "medium", or "low"

Return valid JSON only. No markdown, no explanation. All keys must be present."""
                },
                {"role": "user", "content": f"Extract from this email:\n\n{payload.email}"}
            ]
        )
        
        content = response.choices[0].message.content.strip()
        
        # Try to parse JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code block
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(1))
                except:
                    data = {}
            else:
                data = {}
        
        return validate_extraction(data).model_dump()

    except Exception:
        # Never fail: return safe extraction fallback.
        try:
            data = rule_based_extract(payload.email)
            return validate_extraction(data).model_dump()
        except Exception:
            return {
                "customer_name": None,
                "order_id": None,
                "product": None,
                "issue": None,
                "intent": "unknown",
                "urgency": "low",
            }


@app.post("/suggest", response_model=SuggestResponse)
def suggest(payload: SuggestRequest) -> dict:
    """
    Generate professional reply under 120 words.
    """
    client = get_openai_client()

    category = validate_category(payload.category) if payload.category else classify_email_rule_based(payload.email)

    if payload.extracted is None:
        try:
            extracted = validate_extraction(rule_based_extract(payload.email))
        except Exception:
            extracted = ExtractResponse(
                customer_name=None,
                order_id=None,
                product=None,
                issue=None,
                intent="unknown",
                urgency="low",
            )
    else:
        extracted = payload.extracted
    
    # Build context from extracted data
    extracted_info = []
    if extracted.customer_name:
        extracted_info.append(f"Customer: {extracted.customer_name}")
    if extracted.order_id:
        extracted_info.append(f"Order: {extracted.order_id}")
    if extracted.product:
        extracted_info.append(f"Product: {extracted.product}")
    if extracted.issue:
        extracted_info.append(f"Issue: {extracted.issue}")
    if extracted.intent:
        extracted_info.append(f"Intent: {extracted.intent}")
    if extracted.urgency:
        extracted_info.append(f"Urgency: {extracted.urgency}")
    
    context = "\n".join(extracted_info) if extracted_info else "No additional context."
    
    if not client:
        # Fallback: template-based response
        response_text = template_based_suggest(payload.email, category, extracted)
        return {"response": validate_response(response_text)}
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.7,
            max_tokens=200,
            messages=[
                {
                    "role": "system",
                    "content": """You are a professional customer support agent.
Write a polite, helpful email response.
Use the customer's name if available.
Address their specific issue.
Keep the response under 120 words.
Be concise but empathetic."""
                },
                {
                    "role": "user",
                    "content": f"""Category: {category}

Extracted Information:
{context}

Customer Email:
{payload.email}

Write a professional response:"""
                }
            ]
        )
        
        response_text = response.choices[0].message.content.strip()
        return {"response": validate_response(response_text)}
    
    except Exception:
        # Fallback on error
        response_text = template_based_suggest(payload.email, category, extracted)
        return {"response": validate_response(response_text)}


# ============================================================================
# OPENENV ENDPOINTS
# ============================================================================
@app.post("/reset", response_model=ResetResponse)
def reset() -> dict:
    """
    Reset the OpenEnv environment.
    Initializes step counter to 0, sets done = false, loads sample email.
    """
    result = env_state.reset()
    return result


@app.post("/step", response_model=StepResponse)
def step(payload: StepRequest) -> dict:
    """
    Execute one step in the OpenEnv environment.
    
    Actions:
    - classify_email: Classify the current email
    - extract_entities: Extract entities from the current email
    - generate_reply: Generate a reply for the current email
    
    Rewards:
    - classify_email: 0.33
    - extract_entities: 0.33
    - generate_reply: 0.34
    """
    result = env_state.step(payload.action)
    return result


@app.get("/state", response_model=StateResponse)
def state() -> dict:
    """
    Get the current OpenEnv environment state.
    Returns step count and done status.
    """
    result = env_state.get_state()
    return result


# ============================================================================
# FALLBACK FUNCTIONS
# ============================================================================
def classify_email_rule_based(email: str) -> str:
    """Rule-based classification when LLM unavailable."""
    text = email.lower()
    
    # Check for spam first to avoid false general inquiries for scam-like mail.
    if any(word in text for word in ["lottery", "win", "claim", "prize", "free money", "unsubscribe", "spam"]):
        return "spam"

    # Promotions are distinct from spam.
    if any(word in text for word in ["sale", "offer", "discount", "limited time", "deal", "promo"]):
        return "promotion"

    # Check for urgent next.
    if any(word in text for word in ["urgent", "asap", "emergency", "critical", "immediately", "today"]):
        return "urgent"
    
    # Check for refund
    if any(word in text for word in ["refund", "money back", "return", "reimburse"]):
        return "refund"
    
    # Check for complaint
    if any(word in text for word in ["not working", "issue", "problem", "complaint", "unhappy", "disappointed", "terrible", "bad", "poor", "worst"]):
        return "complaint"
    
    # Check for billing
    if any(word in text for word in ["billing", "invoice", "charge", "payment", "subscription", "price", "cost"]):
        return "billing"
    
    # Check for order status
    if any(word in text for word in ["order", "tracking", "shipment", "delivery", "package", "shipping"]):
        return "order_status"
    
    # Check for technical support
    if any(word in text for word in ["technical", "bug", "error", "crash", "not working", "broken", "issue", "problem"]):
        return "technical_support"
    
    # Default
    return "general_inquiry"


def rule_based_extract(email: str) -> dict:
    """Rule-based extraction when LLM unavailable."""
    text = email.lower()
    
    # Extract order ID
    order_patterns = [
        r'order\s*(?:#|number|id)?\s*:?\s*([A-Z0-9-]+)',
        r'#\s*([0-9]+)',
        r'order\s+([0-9]+)'
    ]
    order_id = None
    for pattern in order_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            order_id = match.group(1)
            break
    
    # Extract customer name (simple pattern)
    name = None
    name_patterns = [
        r'(?:from|name|regards|sincerely)[,:]?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*$'
    ]
    for pattern in name_patterns:
        match = re.search(pattern, email, re.MULTILINE | re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            break
    
    # Determine issue
    issue_keywords = {
        "refund request": ["refund", "money back"],
        "damaged product": ["damaged", "broken", "defective"],
        "delivery issue": ["not delivered", "late delivery", "missing"],
        "technical problem": ["not working", "error", "bug", "crash"],
        "billing question": ["charge", "billing", "invoice", "payment"]
    }
    issue = None
    for issue_type, keywords in issue_keywords.items():
        if any(kw in text for kw in keywords):
            issue = issue_type
            break
    
    # Determine intent (must be one of VALID_CATEGORIES)
    intent = classify_email_rule_based(email)
    
    # Determine urgency
    urgency = None
    if any(word in text for word in ["urgent", "asap", "emergency", "immediately"]):
        urgency = "high"
    elif any(word in text for word in ["soon", "quickly", "please"]):
        urgency = "medium"
    else:
        urgency = "low"
    
    # Extract product (simple approach)
    product = None
    product_keywords = ["laptop", "phone", "tablet", "computer", "device", "product", "item"]
    for kw in product_keywords:
        if kw in text:
            product = kw
            break
    
    return {
        "customer_name": name,
        "order_id": order_id,
        "product": product,
        "issue": issue,
        "intent": intent,
        "urgency": urgency
    }


def template_based_suggest(email: str, category: str, extracted: ExtractResponse) -> str:
    """Template-based response when LLM unavailable."""
    
    # Use customer name if available
    greeting = f"Dear {extracted.customer_name}," if extracted.customer_name else "Dear Customer,"
    
    # Reference order if available
    order_ref = f" (Order: {extracted.order_id})" if extracted.order_id else ""
    
    templates = {
        "refund": f"{greeting}\n\nThank you for contacting us regarding your refund request{order_ref}. We have received your inquiry and are reviewing it. Our team will process your refund within 3-5 business days. You will receive a confirmation email once completed.\n\nBest regards,\nSupport Team",
        
        "complaint": f"{greeting}\n\nWe sincerely apologize for the inconvenience you experienced{order_ref}. Your feedback is important to us. We are investigating this matter and will contact you within 24 hours with a resolution.\n\nBest regards,\nCustomer Relations Team",
        
        "order_status": f"{greeting}\n\nThank you for contacting us about your order{order_ref}. We are tracking your package and will provide an update on its status within 24 hours. We appreciate your patience.\n\nBest regards,\nSupport Team",
        
        "technical_support": f"{greeting}\n\nThank you for reporting this technical issue{order_ref}. Our engineering team has been notified and is investigating. We will provide an update within 24 hours. In the meantime, please try restarting your device.\n\nBest regards,\nTechnical Support Team",
        
        "billing": f"{greeting}\n\nThank you for reaching out regarding your billing inquiry{order_ref}. We have reviewed your account and will investigate this matter. Our billing team will contact you within 24 hours with a resolution.\n\nBest regards,\nBilling Team",
        
        "general_inquiry": f"{greeting}\n\nThank you for contacting us{order_ref}. We have received your message and will respond within 24 hours. If this is urgent, please call our support line.\n\nBest regards,\nSupport Team",
        
        "urgent": f"{greeting}\n\nWe have received your urgent request{order_ref} and are prioritizing it. A specialist will contact you within 2 hours. Thank you for your patience.\n\nBest regards,\nUrgent Support Team",
        
        "spam": "Thank you for your message. If you have a legitimate inquiry, please contact our support team directly."
    }
    
    return templates.get(category, templates["general_inquiry"])


# ============================================================================
# MAIN
# ============================================================================
def main() -> None:
    """Main entry point for running the server."""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()

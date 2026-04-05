"""
Email Triage OpenEnv - FastAPI Server for Hugging Face Spaces

This is the main entry point for Hugging Face Spaces deployment.
It exposes OpenEnv-compatible endpoints for environment interaction.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.env import EmailTriageEnv
from src.models import Action


class ResetRequest(BaseModel):
    """Request model for reset endpoint."""
    task_id: str = "task_hard"


class StepRequest(BaseModel):
    """Request model for step endpoint."""
    category: str | None = None
    priority: str | None = None
    action: str | None = None
    reply_template: str | None = None


# Initialize FastAPI app
app = FastAPI(
    title="Email Triage OpenEnv",
    description="OpenEnv-compatible email triage environment",
    version="1.0.0"
)

# Global environment instance
_runtime_env: EmailTriageEnv | None = None


@app.get("/")
def root() -> Dict[str, Any]:
    """Root endpoint - health check."""
    return {
        "message": "Email Triage OpenEnv is running",
        "endpoints": {
            "health": "/health",
            "reset_get": "/reset?task_id=task_easy",
            "reset_post": "POST /reset",
            "step": "POST /step",
            "state": "/state"
        }
    }


@app.get("/health")
def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "environment": "ready"}


@app.get("/reset")
def reset_get(task_id: str = "task_hard") -> Dict[str, Any]:
    """
    Reset the environment with specified task.
    
    Query Parameters:
        task_id: One of "task_easy", "task_medium", "task_hard"
    
    Returns:
        Initial observation and state
    """
    global _runtime_env
    _runtime_env = EmailTriageEnv(task_id=task_id)
    obs = _runtime_env.reset()
    return {
        "observation": obs.model_dump(),
        "state": _runtime_env.state().model_dump()
    }


@app.post("/reset")
def reset_post(payload: ResetRequest) -> Dict[str, Any]:
    """
    Reset the environment with specified task (POST version).
    
    Request Body:
        task_id: One of "task_easy", "task_medium", "task_hard"
    
    Returns:
        Initial observation and state
    """
    global _runtime_env
    _runtime_env = EmailTriageEnv(task_id=payload.task_id)
    obs = _runtime_env.reset()
    return {
        "observation": obs.model_dump(),
        "state": _runtime_env.state().model_dump()
    }


@app.post("/step")
def step(payload: StepRequest) -> Dict[str, Any]:
    """
    Execute one step in the environment.
    
    Request Body:
        category: Email category (billing, technical, sales, account, complaint, shipping, other)
        priority: Priority level (low, medium, high, urgent)
        action: Action to take (reply, escalate, archive)
        reply_template: Reply template to use
    
    Returns:
        observation: Next observation
        reward: Step reward
        done: Whether episode is complete
        info: Additional information
        state: Current environment state
    """
    global _runtime_env
    
    if _runtime_env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    try:
        action = Action.model_validate(payload.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid action payload: {exc}") from exc

    try:
        observation, reward, done, info = _runtime_env.step(action)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "observation": observation.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
        "state": _runtime_env.state().model_dump(),
    }


@app.get("/state")
def state() -> Dict[str, Any]:
    """
    Get current environment state.
    
    Returns:
        Environment state including:
        - task_id: Current task identifier
        - current_index: Position in email queue
        - total_emails: Total number of emails
        - cumulative_reward: Total reward accumulated
        - last_reward: Reward from last step
        - done: Whether episode is complete
    """
    global _runtime_env
    
    if _runtime_env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    return _runtime_env.state().model_dump()


def main() -> None:
    """Main entry point for running the server."""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()

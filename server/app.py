from __future__ import annotations

import os
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.env import EmailTriageEnv
from src.models import Action


class ResetRequest(BaseModel):
    task_id: str = "task_hard"


class StepRequest(BaseModel):
    category: str | None = None
    priority: str | None = None
    action: str | None = None
    reply_template: str | None = None


app = FastAPI(title="email-triage-env")
_runtime_env = EmailTriageEnv(task_id="task_hard")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
def reset(payload: ResetRequest) -> Dict[str, Any]:
    global _runtime_env
    _runtime_env = EmailTriageEnv(task_id=payload.task_id)
    obs = _runtime_env.reset()
    return {"observation": obs.model_dump(), "state": _runtime_env.state().model_dump()}


@app.post("/step")
def step(payload: StepRequest) -> Dict[str, Any]:
    try:
        action = Action.model_validate(payload.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid action payload: {exc}") from exc

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
    return _runtime_env.state().model_dump()


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()

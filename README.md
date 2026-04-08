---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Email Triage Environment

## Problem

Customer-support inboxes mix routine requests with urgent incidents. A triage system has to do more than classify text; it has to make usable decisions under constraints:

- identify the issue type
- assess urgency
- choose an action (reply, escalate, archive)
- keep behavior consistent across a full queue

This project provides a repeatable environment to evaluate that behavior with task-level scoring.

## Approach

The environment is split into three evaluation tasks that increase in difficulty. Each task runs through a full email queue and computes a score from step-level rewards.

Design principles used here:

- one shared environment contract for all tasks
- stricter requirements as difficulty increases
- deterministic baseline available at all times
- optional LLM-backed path for harder behavior checks

The result is practical: you can compare stable baseline behavior vs model-assisted behavior without changing the core environment loop.

## Tasks

- task_easy: basic triage signal quality (mainly category correctness)
- task_medium: category plus priority/action quality
- task_hard: category , priority , action , reply_template

## Results (Current Snapshot)

Recent runs are typically in this range:

- task_easy: about 0.93
- task_medium: about 0.82 to 0.90
- task_hard: can vary more when model-assisted paths are active , currently for the dataset giving in range (0.76 - 0.79)

Interpretation:

- repeated identical scores usually mean deterministic path only
- variability usually indicates model-assisted path or stochastic fallback behavior

## Setup

1. Create and activate your virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment values in .env:

```dotenv
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
HF_TOKEN=your_token_here
```

If provider calls fail (quota/rate limits), fallback logic is used where implemented.

## Running

Run the evaluator:

```bash
python inference.py
```

This executes easy, medium, and hard, printing START/STEP/END logs and one final summary line.

Run the API server:

```bash
python app.py
```

Default bind:

- host: 0.0.0.0
- port: 7860

## How To Check

Use this API sequence:

1. POST /reset
2. POST /step (repeat until done=true)
3. GET /state
4. GET /health

Expected behavior:

- /reset initializes environment state and returns `{ "observation": { "current_email": { "id", "subject", "body" }, "remaining_emails": N } }`
- /step processes exactly one email using `{ "action": "classify_email" | "extract_entities" | "generate_reply" }` and returns per-step reward
- /state returns current index and done flag for the active episode

## Notes

- Keep .env and tokens private.
- For stable benchmarking, use deterministic mode.
- For behavior stress tests, enable model-assisted paths.

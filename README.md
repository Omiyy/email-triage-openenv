---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: inference.py
pinned: false
---

# Email Triage OpenEnv Environment

A realistic Email Triage OpenEnv environment where an AI agent must manage an inbox by classifying emails, prioritizing urgent emails, taking appropriate actions, and generating professional replies.

## Overview

This environment simulates a real-world customer support email workflow where an AI agent:
- **Classifies** incoming emails into categories (billing, technical, sales, account, complaint, shipping, other)
- **Prioritizes** emails based on urgency (low, medium, high, urgent)
- **Takes actions** (reply, escalate, archive)
- **Selects appropriate reply templates** for professional responses

The environment provides step-wise rewards based on:
- Category classification accuracy (+0.3)
- Priority handling (+0.2)
- Action correctness (+0.2)
- Reply template selection (+0.3)

The agent is evaluated using a deterministic grading system that produces a normalized final score between 0.0 and 1.0.

## Key Features

- **Typed Pydantic models** for observation, action, state, and reward
- **Gym-compatible loop**: `reset()`, `step(action)`, `state()`
- **Deterministic reward function** with partial credit
- **Normalized grader** with scores in `[0.0, 1.0]`
- **Three progressively harder tasks** evaluating different agent capabilities
- **Synthetic dataset** of 30 fully labeled support emails
- **Hybrid agent architecture** supporting both rule-based and LLM-based approaches
- **Docker deployment** for reproducible execution
- **Hugging Face Spaces deployment** ready

## Project Structure

```text
email-triage-env/
├── inference.py          # Main entry point for agent evaluation
├── openenv.yaml          # OpenEnv specification file
├── Dockerfile            # Docker container configuration
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
├── src/
│   ├── env.py           # EmailTriageEnv environment implementation
│   ├── models.py        # Pydantic models (EmailRecord, Action, Observation, State)
│   ├── dataset.py       # Synthetic email dataset (30 labeled emails)
│   ├── tasks.py         # Task configurations (easy, medium, hard)
│   ├── rewards.py       # Reward computation logic
│   ├── graders.py       # Deterministic grading system
│   └── visualization.py # Reward plotting utilities
```

## Environment Design

### Observation Space

Each step provides an `Observation` with:
- `email_id`: Unique identifier for the email
- `email_text`: Raw email body text
- `task_id`: Active task identifier
- `valid_categories`: List of valid category values
- `valid_priorities`: List of valid priority values
- `valid_actions`: List of valid action values

### Action Space

Agent action (`Action` model):
- `category`: One of `billing|technical|sales|account|complaint|shipping|other`
- `priority`: One of `low|medium|high|urgent`
- `action`: One of `reply|escalate|archive`
- `reply_template`: Template key string (e.g., `billing_refund`, `tech_troubleshoot`)

### State Space

`state()` returns:
- `task_id`: Active task identifier
- `current_index`: Position in email queue
- `total_emails`: Total emails in dataset
- `cumulative_reward`: Total reward accumulated
- `last_reward`: Reward from last step
- `done`: Episode completion flag

Additionally, the environment tracks enhanced statistics in `info["stats"]`:
- `emails_processed`: Count of processed emails
- `emails_remaining`: Emails left to process
- `replies_sent`: Number of reply actions taken
- `escalations`: Number of escalate actions
- `archived`: Number of archive actions
- `urgent_handled`: Number of urgent priority emails processed

## Tasks

| Task | Difficulty | Required Outputs | Description |
|------|------------|------------------|-------------|
| `task_easy` | Easy | category | Email category classification only |
| `task_medium` | Medium | category, priority, action | Category + priority assignment + action selection |
| `task_hard` | Hard | category, priority, action, reply_template | Full email triage with reply template selection |

### Task Logic

- **task_easy**: Only category matters. Other fields use safe defaults (priority=medium, action=reply).
- **task_medium**: Category, priority, and action must be correct. Reply template uses a generic value.
- **task_hard**: All fields must be correct including the exact reply template mapped from category and action.

## Reward Function

Continuous per-step reward with partial credit:

| Component | Correct | Reward |
|-----------|---------|--------|
| Category | Yes | +0.3 |
| Priority | Yes | +0.2 |
| Action | Yes | +0.2 |
| Reply Template | Yes | +0.3 |

### Penalties

- Wrong classification (category mismatch): -0.2
- Unnecessary escalation: -0.3

Rewards are computed at every step, not only at episode end.

## Agent Architecture

The inference agent uses a **hybrid approach** combining rule-based classification with optional LLM integration:

| Component | Method | Logic |
|-----------|--------|-------|
| Category | Rule-based | Keyword matching with prioritized categories |
| Priority | Rule-based | Urgency keyword detection |
| Action | Rule-based | Category + priority based logic |
| Reply Template | Rule-mapped | Direct mapping from category to template |

### Category Detection Priority

1. Complaint (emotional indicators)
2. Billing (financial indicators)
3. Shipping (delivery indicators)
4. Account (login/access indicators)
5. Sales (pricing/business indicators)
6. Technical (problem indicators)
7. Other (default)

### Reply Template Mapping

```
billing → billing_refund / billing_invoice
technical → tech_troubleshoot / escalate_specialist
sales → sales_pricing
account → account_unlock
complaint → complaint_apology / escalate_specialist
shipping → shipping_update
other → archive_no_reply
```

### Optional LLM Integration

The agent supports OpenAI-compatible LLM APIs via environment variables:
- `API_BASE_URL`: Base URL for the LLM API
- `MODEL_NAME`: Model identifier (e.g., `mistralai/Mistral-7B-Instruct-v0.2`)
- `HF_TOKEN`: API key for authentication

When these are provided, the agent can use LLM for enhanced classification.

## Setup Instructions

### Local Run

```bash
pip install -r requirements.txt
python inference.py
```

### Docker Run

```bash
docker build -t email-triage-env .
docker run --rm --env-file .env email-triage-env
```

## Environment Variables

Create a `.env` file with the following variables:

| Variable | Description | Required |
|----------|-------------|----------|
| `API_BASE_URL` | Base URL for OpenAI-compatible API | Optional |
| `MODEL_NAME` | LLM model name for inference | Optional |
| `HF_TOKEN` | API token for authentication | Optional |

Example `.env`:
```
API_BASE_URL=https://api-inference.huggingface.co/models
MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
HF_TOKEN=your_token_here
```

## Hugging Face Spaces Deployment

This project is configured for Hugging Face Spaces deployment:

1. **Space Configuration**: Uses Docker SDK (specified in YAML header)
2. **Entry Point**: `inference.py` runs automatically on container start
3. **Environment Variables**: Add `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` in Space Settings > Secrets
4. **Build Process**: Hugging Face automatically builds the Docker image and runs the environment

To deploy:
1. Create a new Space on Hugging Face
2. Select "Docker" as the SDK
3. Push this repository to the Space
4. Add environment variables in Space Settings
5. The environment will automatically build and run

## Inference Process

The `inference.py` script performs the following:

1. **Load Environment Variables**: Reads `.env` file for API configuration
2. **Initialize Agent**: Creates a HybridEmailAgent (rule-based with optional LLM)
3. **Run Tasks**: Executes task_easy, task_medium, and task_hard sequentially
4. **Environment Loop**: For each task:
   - Calls `env.reset()` to initialize
   - Iterates through all 30 emails
   - Calls `agent.decide_action()` for each email
   - Calls `env.step(action)` to execute and get reward
   - Tracks component accuracy (category, priority, action, reply)
5. **Print Results**: Outputs step rewards, cumulative rewards, and final normalized scores

## Example Output

```
Email Triage OpenEnv Inference
API_BASE_URL set: True
MODEL_NAME: mistralai/Mistral-7B-Instruct-v0.2
HF_TOKEN set: True

=== Running task_easy ===
Step 01 | email=E001 | reward=+0.30 | cat=+0.30 pri=+0.00 act=+0.00 rep=+0.00 penalties={} | ok(cat=1 pri=0 act=1 rep=0)
Step 02 | email=E002 | reward=+0.30 | cat=+0.30 pri=+0.00 act=+0.00 rep=+0.00 penalties={} | ok(cat=1 pri=1 act=1 rep=0)
...
Final score (task_easy): 0.9333
Cumulative reward (task_easy): 8.0000
Average reward (task_easy): 0.2667
Agent calls: 30

=== Running task_medium ===
...
Final score (task_medium): 0.9000

=== Running task_hard ===
...
Final score (task_hard): 0.8917

=== Final Summary ===
Task 1 (Easy):
- Final Score: 0.9333
- Category Accuracy: 93.33%

Task 2 (Medium):
- Final Score: 0.9000
- Priority Accuracy: 86.67%
- Action Accuracy: 90.00%

Task 3 (Hard):
- Final Score: 0.8917
- Reply Accuracy: 86.67%
```

## System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Email Dataset  │────▶│   Environment   │────▶│   Observation   │
│   (30 emails)   │     │  (EmailTriage)  │     │  (email_text)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Final Score    │◀────│     Grader      │◀────│     Reward      │
│    (0-1.0)      │     │  (Deterministic)│     │  (Step-wise)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        ▲
                                                        │
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Reply Template │◀────│     Action      │◀────│     Agent       │
│ (category-based)│     │ (reply/escalate)│     │ (Hybrid: Rules  │
└─────────────────┘     └─────────────────┘     │   + Optional LLM)│
                                                └─────────────────┘
```

## Baseline Scores

| Task | Score | Category Accuracy | Priority Accuracy | Action Accuracy | Reply Accuracy |
|------|-------|-------------------|-------------------|-----------------|----------------|
| Easy | 0.9333 | 93.33% | - | - | - |
| Medium | 0.9000 | 93.33% | 86.67% | 90.00% | - |
| Hard | 0.8917 | 93.33% | 86.67% | 90.00% | 86.67% |

## Conclusion

This project is an **OpenEnv AI agent evaluation environment** designed for benchmarking email triage agents. It provides:

- A realistic simulation of customer support workflows
- Deterministic, reproducible evaluation metrics
- Multiple difficulty levels to test agent capabilities
- Easy deployment via Docker and Hugging Face Spaces
- Compatibility with both rule-based and LLM-based agents

The environment is production-ready and suitable for research, competition, or production agent evaluation.

---

**License**: MIT  
**Author**: OpenEnv Contributors  
**Repository**: https://huggingface.co/spaces/YOUR_USERNAME/email-triage-env

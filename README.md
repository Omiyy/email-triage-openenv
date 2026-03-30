# Email Triage OpenEnv Environment

Production-ready OpenEnv-style environment for real-world email triage simulation.

It includes:
- Typed Pydantic models for observation, action, state, and reward
- Gym-compatible loop: `reset()`, `step(action)`, `state()`
- Deterministic rule-based grading with a normalized final score in `[0.0, 1.0]`
- Three progressively harder tasks
- Synthetic dataset of 30 fully labeled support emails

## Project Structure

```text
email-triage-env/
├── inference.py
├── openenv.yaml
├── Dockerfile
├── README.md
├── requirements.txt
├── src/
│   ├── env.py
│   ├── models.py
│   ├── dataset.py
│   ├── tasks.py
│   ├── rewards.py
│   ├── graders.py
```

## Environment Design

### Observation Space

Each step provides:
- `email_id`: unique email id
- `email_text`: raw email body
- `task_id`: active task identifier
- `valid_categories`, `valid_priorities`, `valid_actions`

### Action Space

Agent action (`Action` model):
- `category`: one of `billing|technical|sales|account|complaint|shipping|other`
- `priority`: one of `low|medium|high|urgent`
- `action`: one of `reply|escalate|archive`
- `reply_template`: template key string

### State Space

`state()` returns:
- task id
- current index
- total emails
- cumulative reward
- last reward
- done flag

## Tasks

1. `task_easy` (Easy): category classification only
2. `task_medium` (Medium): category + priority
3. `task_hard` (Hard): category + priority + action + reply template

## Reward Function

Continuous per-step reward with partial credit:

- Correct category: `+0.3`
- Correct priority: `+0.2`
- Correct action: `+0.2`
- Correct reply template: `+0.3`

Penalties:
- Wrong classification (category mismatch): `-0.2`
- Unnecessary escalation: `-0.3`

Rewards are computed at every step, not only at episode end.

## Grader

Deterministic scoring in `src/graders.py`:

- Final score is normalized: `correct_decisions / total_possible_decisions`
- Always in `[0.0, 1.0]`
- No external LLM used for grading

## Setup

### Local Python

```bash
pip install -r requirements.txt
python inference.py
```

Optional env vars for OpenAI-compatible endpoint:

```bash
export API_BASE_URL="https://your-openai-compatible-endpoint/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your_token"
python inference.py
```

If `API_BASE_URL` is not set, inference uses deterministic local heuristics.

### Docker

```bash
docker build -t email-triage-env .
docker run --rm \
  -e API_BASE_URL="https://your-openai-compatible-endpoint/v1" \
  -e MODEL_NAME="gpt-4o-mini" \
  -e HF_TOKEN="your_token" \
  email-triage-env
```

It also runs without external endpoint:

```bash
docker run --rm email-triage-env
```

## How Inference Works

`inference.py`:
- Reads `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
- Creates OpenAI client if endpoint is configured
- Runs all 3 tasks sequentially
- Prints per-step reward breakdown
- Prints final score and cumulative reward per task

## Example Output

```text
Email Triage OpenEnv Inference
API_BASE_URL set: False
MODEL_NAME: gpt-4o-mini
HF_TOKEN set: False

=== Running task_easy ===
Step 01 | email=E001 | reward=+0.30 | cat=+0.30 pri=+0.00 act=+0.00 rep=+0.00 penalties={}
...
Final score (task_easy): 0.9333
Cumulative reward (task_easy): 6.8000

=== Running task_medium ===
...
Final score (task_medium): 0.8500

=== Running task_hard ===
...
Final score (task_hard): 0.7750
```

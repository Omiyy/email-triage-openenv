from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt


def plot_rewards(step_rewards: Sequence[float], title: str = "Reward vs Steps") -> None:
    steps = list(range(1, len(step_rewards) + 1))
    plt.figure()
    plt.plot(steps, step_rewards, marker="o")
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.tight_layout()


def plot_cumulative_rewards(cumulative_rewards: Sequence[float], title: str = "Cumulative Reward vs Steps") -> None:
    steps = list(range(1, len(cumulative_rewards) + 1))
    plt.figure()
    plt.plot(steps, cumulative_rewards, marker="o")
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Cumulative Reward")
    plt.grid(True)
    plt.tight_layout()


def plot_task_scores(task_scores: dict[str, float]) -> None:
    plt.figure()
    labels = list(task_scores.keys())
    values = [task_scores[label] for label in labels]
    plt.bar(labels, values)
    plt.title("Final Score per Task")
    plt.xlabel("Task")
    plt.ylabel("Final Score")
    plt.ylim(0.0, 1.0)
    plt.grid(axis="y")
    plt.tight_layout()


def show_plots() -> None:
    plt.show()

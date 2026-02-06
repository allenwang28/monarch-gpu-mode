"""
RL primitives for async training.

Provides common dataclasses used across RL training:
- Trajectory: A single rollout from generation
- TrainMetrics: Metrics from a training step
"""

from dataclasses import dataclass


@dataclass
class Trajectory:
    """A single trajectory from generation."""
    task_question: str
    task_answer: int | str
    response_text: str
    reward: float
    is_correct: bool
    num_turns: int
    num_tool_calls: int
    generator_id: int
    policy_version: int  # Which version of weights was used
    model_only_text: str = ""  # Model-generated text only (no tool results)


@dataclass
class TrainMetrics:
    """Metrics from a training step."""
    step: int
    loss: float
    batch_size: int
    avg_reward: float
    policy_version: int

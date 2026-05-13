"""Reward helpers for RoboTwin outcome RL."""

from __future__ import annotations


def binary_success_reward(success: bool) -> float:
    return 1.0 if bool(success) else 0.0

#!/usr/bin/env python3
"""Train the strict-artifact offline GRPO smoke adapter.

This command validates the Phase 4 data/loss/optimizer/checkpoint path on a
collected `grpo_groups.jsonl`. It does not yet update the LingBot-VA actor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wan_va.rl.trainer import OfflineGrpoTrainer, OfflineGrpoTrainerConfig


def run_smoke_training(
    groups_jsonl: Path,
    output_dir: Path,
    *,
    steps: int = 10,
    learning_rate: float = 1e-3,
    clip_low: float = 0.2,
    clip_high: float = 0.28,
    device: str = "cpu",
    seed: int = 0,
) -> dict:
    config = OfflineGrpoTrainerConfig(
        groups_jsonl=groups_jsonl.expanduser(),
        output_dir=output_dir.expanduser(),
        steps=steps,
        learning_rate=learning_rate,
        clip_low=clip_low,
        clip_high=clip_high,
        device=device,
        seed=seed,
    )
    result = OfflineGrpoTrainer(config).train()
    return result.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict-artifact offline GRPO smoke training.")
    parser.add_argument("--groups-jsonl", type=Path, required=True, help="Path to groups/grpo_groups.jsonl.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for metrics and checkpoint.")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--clip-low", type=float, default=0.2)
    parser.add_argument("--clip-high", type=float, default=0.28)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    result = run_smoke_training(
        args.groups_jsonl,
        args.output_dir,
        steps=args.steps,
        learning_rate=args.learning_rate,
        clip_low=args.clip_low,
        clip_high=args.clip_high,
        device=args.device,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train LingBot-VA action modules with real denoising-transition replay.

Unlike `train_offline_grpo_smoke.py`, this command loads the LingBot
transformer and recomputes current transition log probabilities from saved
replay context. Input artifacts must contain `replay_context` and per-transition
`replay_input` fields.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path

try:
    from tools._repo_root import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _repo_root import ensure_repo_root_on_path

REPO_ROOT = ensure_repo_root_on_path()


def _git_commit(repo_root: Path = REPO_ROOT) -> str | None:
    for env_key in ("GIT_COMMIT", "SUBMIT_GIT_COMMIT"):
        value = os.environ.get(env_key)
        if value and value != "unknown":
            return value
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def run_actor_replay_training(
    *,
    groups_jsonl: Path,
    output_dir: Path,
    model_path: Path,
    config_name: str = "robotwin_grpo_train",
    steps: int = 1,
    learning_rate: float = 5e-5,
    clip_low: float = 0.2,
    clip_high: float = 0.28,
    device: str = "cuda",
    dtype: str = "bfloat16",
    seed: int = 0,
    trainable_mode: str = "action_heads",
    trainable_param_patterns: tuple[str, ...] = (),
    frozen_param_patterns: tuple[str, ...] = (),
    action_num_inference_steps: int | None = None,
    action_snr_shift: float | None = None,
    logprob_reduction: str = "mean",
    logprob_std_floor: float | None = 0.1,
    progress_every: int = 0,
    opts: list[str] | None = None,
) -> dict:
    import torch

    from wan_va.configs import VA_CONFIGS
    from wan_va.configs.runtime import apply_cli_overrides, apply_env_overrides
    from wan_va.modules.utils import load_transformer
    from wan_va.rl.actor_replay import (
        ActorReplayGrpoTrainer,
        ActorReplayTrainerConfig,
        torch_dtype_from_string,
    )

    config = deepcopy(VA_CONFIGS[config_name])
    apply_env_overrides(config)
    apply_cli_overrides(config, opts or [])

    model_path = model_path.expanduser()
    transformer_path = model_path / "transformer"
    if not transformer_path.exists():
        raise FileNotFoundError(f"missing transformer checkpoint directory: {transformer_path}")

    torch_dtype = torch_dtype_from_string(dtype)
    target_device = torch.device(device)
    transformer = load_transformer(
        str(transformer_path),
        torch_dtype=torch_dtype,
        torch_device=target_device,
        attn_mode=getattr(config, "attn_mode", "torch"),
    )

    trainer_config = ActorReplayTrainerConfig(
        groups_jsonl=groups_jsonl.expanduser(),
        output_dir=output_dir.expanduser(),
        model_path=str(model_path),
        config_name=config_name,
        git_commit=_git_commit(),
        steps=steps,
        learning_rate=learning_rate,
        clip_low=clip_low,
        clip_high=clip_high,
        device=device,
        dtype=dtype,
        seed=seed,
        trainable_mode=trainable_mode,
        trainable_param_patterns=trainable_param_patterns or tuple(getattr(config, "trainable_param_patterns", ())),
        frozen_param_patterns=frozen_param_patterns or tuple(getattr(config, "frozen_param_patterns", ())),
        action_num_inference_steps=int(action_num_inference_steps or getattr(config, "action_num_inference_steps", 50)),
        action_snr_shift=float(action_snr_shift if action_snr_shift is not None else getattr(config, "action_snr_shift", 1.0)),
        logprob_reduction=logprob_reduction,
        logprob_std_floor=_normalize_logprob_std_floor(logprob_std_floor),
        progress_every=progress_every,
    )
    result = ActorReplayGrpoTrainer(trainer_config, transformer=transformer).train()
    return result.to_dict()


def _normalize_logprob_std_floor(value: float | None) -> float | None:
    if value is None:
        return None
    if value <= 0.0:
        return None
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real LingBot actor replay GRPO training.")
    parser.add_argument("--groups-jsonl", type=Path, required=True, help="Path to groups/grpo_groups.jsonl.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for metrics and checkpoint.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(os.environ.get("WAN_VA_MODEL_PATH", "")),
        help="LingBot-VA checkpoint root containing transformer/.",
    )
    parser.add_argument("--config-name", default="robotwin_grpo_train")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--clip-low", type=float, default=0.2)
    parser.add_argument("--clip-high", type=float, default=0.28)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trainable-mode", default="action_heads")
    parser.add_argument("--trainable-param-pattern", action="append", default=[])
    parser.add_argument("--frozen-param-pattern", action="append", default=[])
    parser.add_argument("--action-num-inference-steps", type=int, default=None)
    parser.add_argument("--action-snr-shift", type=float, default=None)
    parser.add_argument("--logprob-reduction", choices=("sum", "mean"), default="mean")
    parser.add_argument(
        "--logprob-std-floor",
        type=float,
        default=0.1,
        help="Training-time std floor for replay logprob stability; set <=0 to disable.",
    )
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--opts", nargs="*", default=[], help="Config overrides as key=value.")
    args = parser.parse_args()

    if not str(args.model_path):
        raise SystemExit("--model-path is required or WAN_VA_MODEL_PATH must be set")

    result = run_actor_replay_training(
        groups_jsonl=args.groups_jsonl,
        output_dir=args.output_dir,
        model_path=args.model_path,
        config_name=args.config_name,
        steps=args.steps,
        learning_rate=args.learning_rate,
        clip_low=args.clip_low,
        clip_high=args.clip_high,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        trainable_mode=args.trainable_mode,
        trainable_param_patterns=tuple(args.trainable_param_pattern),
        frozen_param_patterns=tuple(args.frozen_param_pattern),
        action_num_inference_steps=args.action_num_inference_steps,
        action_snr_shift=args.action_snr_shift,
        logprob_reduction=args.logprob_reduction,
        logprob_std_floor=args.logprob_std_floor,
        progress_every=args.progress_every,
        opts=args.opts,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Diagnose real actor replay against stored strict GRPO transitions."""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path

import torch

try:
    from tools._repo_root import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _repo_root import ensure_repo_root_on_path

ensure_repo_root_on_path()


def _tensor(value, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    return tensor.to(device=device, dtype=dtype)


def _normalize_logprob_std_floor(value: float | None) -> float | None:
    if value is None:
        return None
    if value <= 0.0:
        return None
    return float(value)


def _transition_std_for_diagnosis(transition, *, device, floor: float | None):
    std = _tensor(transition["transition_std"], device=device, dtype=torch.float32)
    if floor is None:
        return std
    return std.clamp_min(float(floor))


def _logprob_for_mean(mean, transition, *, device, dtype, logprob_std_floor: float | None = None):
    from wan_va.rl.denoising_replay import compute_gaussian_transition_logprob

    return compute_gaussian_transition_logprob(
        transition_mean=mean,
        action_xt_next=_tensor(transition["action_xt_next"], device=device, dtype=dtype),
        transition_std=_transition_std_for_diagnosis(transition, device=device, floor=logprob_std_floor),
        logprob_mask=_tensor(transition["logprob_mask"], device=device, dtype=torch.bool),
    ).logprob_sum.detach().cpu().float().flatten()


def diagnose_actor_replay(
    *,
    groups_jsonl: Path,
    model_path: Path,
    output_json: Path,
    config_name: str,
    device: str,
    dtype: str,
    action_num_inference_steps: int,
    logprob_std_floor: float | None,
    max_examples: int,
) -> dict:
    from wan_va.configs import VA_CONFIGS
    from wan_va.configs.runtime import apply_env_overrides
    from wan_va.modules.utils import load_transformer
    from wan_va.rl.actor_replay import LingBotActionReplayPolicy, iter_actor_replay_examples, torch_dtype_from_string
    from wan_va.utils.scheduler import FlowMatchScheduler

    config = deepcopy(VA_CONFIGS[config_name])
    apply_env_overrides(config)

    torch_dtype = torch_dtype_from_string(dtype)
    target_device = torch.device(device)
    transformer = load_transformer(
        str(model_path.expanduser() / "transformer"),
        torch_dtype=torch_dtype,
        torch_device=target_device,
        attn_mode=getattr(config, "attn_mode", "torch"),
    )
    transformer.to(device=target_device, dtype=torch_dtype).eval()

    action_snr_shift = float(getattr(config, "action_snr_shift", 1.0))
    scheduler = FlowMatchScheduler(shift=action_snr_shift, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(action_num_inference_steps)
    policy = LingBotActionReplayPolicy(
        transformer,
        action_scheduler=scheduler,
        device=target_device,
        dtype=torch_dtype,
    )

    normalized_std_floor = _normalize_logprob_std_floor(logprob_std_floor)
    examples = []
    with torch.no_grad():
        for idx, example in enumerate(iter_actor_replay_examples(groups_jsonl.expanduser())):
            if idx >= max_examples:
                break
            transition = example.transition
            replay_mean = policy.predict_transition_mean(example)
            stored_mean = _tensor(transition["transition_mean"], device=target_device, dtype=torch_dtype)
            action_next = _tensor(transition["action_xt_next"], device=target_device, dtype=torch_dtype)
            mask = _tensor(transition["logprob_mask"], device=target_device, dtype=torch.bool)
            std = _tensor(transition["transition_std"], device=target_device, dtype=torch.float32)
            old_logprob = torch.as_tensor(transition["old_logprob_sum"], dtype=torch.float32).flatten()
            if "old_logprob_mean" in transition:
                old_logprob_mean = torch.as_tensor(transition["old_logprob_mean"], dtype=torch.float32).flatten()
            else:
                old_count = torch.as_tensor(transition["old_logprob_count"], dtype=torch.float32).flatten().clamp_min(1.0)
                old_logprob_mean = old_logprob / old_count

            replay_logprob = _logprob_for_mean(replay_mean, transition, device=target_device, dtype=torch_dtype)
            stored_logprob = _logprob_for_mean(stored_mean, transition, device=target_device, dtype=torch_dtype)
            trainer_replay_logprob = _logprob_for_mean(
                replay_mean,
                transition,
                device=target_device,
                dtype=torch_dtype,
                logprob_std_floor=normalized_std_floor,
            )
            trainer_stored_logprob = _logprob_for_mean(
                stored_mean,
                transition,
                device=target_device,
                dtype=torch_dtype,
                logprob_std_floor=normalized_std_floor,
            )
            mask_count = int(mask.sum().detach().cpu())
            replay_logprob_mean = replay_logprob / max(mask_count, 1)
            stored_logprob_mean = stored_logprob / max(mask_count, 1)
            trainer_replay_logprob_mean = trainer_replay_logprob / max(mask_count, 1)
            trainer_stored_logprob_mean = trainer_stored_logprob / max(mask_count, 1)
            candidate_logprobs = {"default": replay_logprob.tolist()}
            candidate_minus_old = {"default": (replay_logprob - old_logprob).tolist()}
            candidate_mean_minus_old = {"default": (replay_logprob_mean - old_logprob_mean).tolist()}
            candidate_mean_deltas = {}
            if bool(example.replay_context.get("use_cfg", False)):
                # Re-run raw CFG branches to catch positive/negative branch-order
                # mismatches without changing the trainer path.
                from wan_va.rl.actor_replay import _restore_action_noise_layout, _to_tensor, restore_transformer_cache

                cache_name = str(example.replay_context.get("cache_name") or "pos")
                restore_transformer_cache(
                    transformer,
                    example.replay_context["transformer_cache"],
                    cache_name=cache_name,
                    device=target_device,
                    dtype=torch_dtype,
                )
                input_dict = policy._build_transformer_input(example.replay_context, transition, transition["replay_input"])
                raw_pred = transformer(input_dict, update_cache=0, cache_name=cache_name, action_mode=True)
                raw_pred = _restore_action_noise_layout(raw_pred, int(example.replay_context["frame_chunk_size"]))
                action_xt = _to_tensor(transition["action_xt"], device=target_device, dtype=torch_dtype)
                timestep = _to_tensor(transition["timestep"], device=target_device, dtype=torch.float32)
                branch_preds = {
                    "cfg_positive": raw_pred[:1],
                    "cfg_negative": raw_pred[1:],
                }
                scale = float(example.replay_context.get("action_guidance_scale", 1.0))
                branch_preds["cfg_guided"] = raw_pred[1:] + scale * (raw_pred[:1] - raw_pred[1:])
                for name, pred in branch_preds.items():
                    mean = scheduler.step(pred, timestep, action_xt, return_dict=False)
                    logprob = _logprob_for_mean(mean, transition, device=target_device, dtype=torch_dtype)
                    candidate_logprobs[name] = logprob.tolist()
                    candidate_minus_old[name] = (logprob - old_logprob).tolist()
                    candidate_mean_minus_old[name] = (logprob / max(mask_count, 1) - old_logprob_mean).tolist()
                    delta = (mean.float() - stored_mean.float()).masked_select(mask)
                    candidate_mean_deltas[name] = {
                        "mean_abs": float(delta.abs().mean().detach().cpu()),
                        "max_abs": float(delta.abs().max().detach().cpu()),
                    }

            mean_delta = (replay_mean.float() - stored_mean.float()).masked_select(mask)
            action_delta = (action_next.float() - stored_mean.float()).masked_select(mask)
            examples.append(
                {
                    "artifact_path": example.ref.artifact_path,
                    "sample_idx": example.ref.sample_idx,
                    "reward": example.ref.reward,
                    "advantage": example.ref.advantage,
                    "denoising_step_index": int(transition["denoising_step_index"]),
                    "use_cfg": bool(example.replay_context.get("use_cfg", False)),
                    "action_guidance_scale": float(example.replay_context.get("action_guidance_scale", 1.0)),
                    "context_action_num_inference_steps": int(
                        example.replay_context.get("action_num_inference_steps", -1)
                    ),
                    "config_action_snr_shift": action_snr_shift,
                    "mask_count": mask_count,
                    "transition_std": float(std.detach().cpu()),
                    "trainer_logprob_std_floor": normalized_std_floor,
                    "old_logprob_sum": old_logprob.tolist(),
                    "old_logprob_mean": old_logprob_mean.tolist(),
                    "stored_recomputed_logprob_sum": stored_logprob.tolist(),
                    "stored_recomputed_logprob_mean": stored_logprob_mean.tolist(),
                    "replay_logprob_sum": replay_logprob.tolist(),
                    "replay_logprob_mean": replay_logprob_mean.tolist(),
                    "trainer_stored_logprob_sum": trainer_stored_logprob.tolist(),
                    "trainer_stored_logprob_mean": trainer_stored_logprob_mean.tolist(),
                    "trainer_replay_logprob_sum": trainer_replay_logprob.tolist(),
                    "trainer_replay_logprob_mean": trainer_replay_logprob_mean.tolist(),
                    "trainer_replay_minus_stored": (trainer_replay_logprob - trainer_stored_logprob).tolist(),
                    "trainer_replay_mean_minus_stored": (
                        trainer_replay_logprob_mean - trainer_stored_logprob_mean
                    ).tolist(),
                    "candidate_replay_logprob_sum": candidate_logprobs,
                    "candidate_replay_minus_old": candidate_minus_old,
                    "candidate_replay_mean_minus_old": candidate_mean_minus_old,
                    "candidate_mean_deltas": candidate_mean_deltas,
                    "stored_recomputed_minus_old": (stored_logprob - old_logprob).tolist(),
                    "stored_recomputed_mean_minus_old": (stored_logprob_mean - old_logprob_mean).tolist(),
                    "replay_minus_old": (replay_logprob - old_logprob).tolist(),
                    "replay_mean_minus_old": (replay_logprob_mean - old_logprob_mean).tolist(),
                    "mean_abs_replay_minus_stored_mean": float(mean_delta.abs().mean().detach().cpu()),
                    "max_abs_replay_minus_stored_mean": float(mean_delta.abs().max().detach().cpu()),
                    "mean_abs_action_next_minus_stored_mean": float(action_delta.abs().mean().detach().cpu()),
                    "max_abs_action_next_minus_stored_mean": float(action_delta.abs().max().detach().cpu()),
                }
            )

    result = {
        "groups_jsonl": str(groups_jsonl),
        "model_path": str(model_path),
        "device": device,
        "dtype": dtype,
        "action_num_inference_steps": action_num_inference_steps,
        "logprob_std_floor": normalized_std_floor,
        "example_count": len(examples),
        "examples": examples,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose real actor replay against stored strict GRPO transitions.")
    parser.add_argument("--groups-jsonl", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=Path(os.environ.get("WAN_VA_MODEL_PATH", "")))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--config-name", default="robotwin_grpo_train")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--action-num-inference-steps", type=int, default=10)
    parser.add_argument(
        "--logprob-std-floor",
        type=float,
        default=0.1,
        help="Trainer-style std floor for additional logprob diagnostics; set <=0 for raw artifact std only.",
    )
    parser.add_argument("--max-examples", type=int, default=1)
    args = parser.parse_args()

    if not str(args.model_path):
        raise SystemExit("--model-path is required or WAN_VA_MODEL_PATH must be set")
    result = diagnose_actor_replay(
        groups_jsonl=args.groups_jsonl,
        model_path=args.model_path,
        output_json=args.output_json,
        config_name=args.config_name,
        device=args.device,
        dtype=args.dtype,
        action_num_inference_steps=args.action_num_inference_steps,
        logprob_std_floor=args.logprob_std_floor,
        max_examples=args.max_examples,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Real actor replay utilities for denoising-step GRPO.

This module is intentionally separate from the smoke trainer. The smoke trainer
validates data/loss/checkpoint plumbing with a tiny scalar policy. The classes
here replay saved LingBot-VA action denoising inputs through the current
transformer, recompute transition log probabilities, and can backpropagate into
real actor parameters.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch

from wan_va.utils.scheduler import FlowMatchScheduler
from wan_va.utils.trainable import configure_trainable_parameters

from .dataset import (
    GrpoTransitionRef,
    iter_strict_artifact_transitions,
    load_strict_artifact,
    read_transition_refs,
    resolve_replay_context,
)
from .denoising_replay import TransitionLogprob, compute_gaussian_transition_logprob
from .grpo_loss import compute_clipped_grpo_loss


REPLAY_CONTEXT_SCHEMA_VERSION = 1
DEFAULT_CACHE_NAME = "pos"


class MissingReplayContextError(ValueError):
    """Raised when an artifact cannot support real actor replay."""


def load_actor_replay_checkpoint_into_transformer(
    transformer: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Load a real actor replay checkpoint's trainable weights into a transformer."""

    path = Path(checkpoint_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"actor replay checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=map_location)
    state_dict = checkpoint.get("trainable_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(f"actor replay checkpoint has no trainable_state_dict: {path}")

    load_result = transformer.load_state_dict(state_dict, strict=False)
    unexpected = list(load_result.unexpected_keys)
    if unexpected:
        raise ValueError(
            "actor replay checkpoint contains unexpected transformer keys: "
            + ", ".join(unexpected[:10])
        )
    tensor_count = sum(1 for value in state_dict.values() if torch.is_tensor(value))
    param_count = sum(int(value.numel()) for value in state_dict.values() if torch.is_tensor(value))
    return {
        "checkpoint_path": str(path),
        "tensor_count": tensor_count,
        "param_count": param_count,
        "missing_key_count": len(load_result.missing_keys),
    }


@dataclass(frozen=True)
class ActorReplayExample:
    ref: GrpoTransitionRef
    transition: dict
    replay_context: dict


@dataclass(frozen=True)
class ActorReplayTrainerConfig:
    groups_jsonl: Path
    output_dir: Path
    steps: int = 1
    learning_rate: float = 5e-5
    clip_low: float = 0.2
    clip_high: float = 0.28
    device: str = "cuda"
    dtype: str = "bfloat16"
    seed: int = 0
    trainable_mode: str = "action_heads"
    trainable_param_patterns: tuple[str, ...] = ()
    frozen_param_patterns: tuple[str, ...] = ()
    action_num_inference_steps: int = 50
    action_snr_shift: float = 1.0
    logprob_reduction: str = "sum"
    logprob_std_floor: float | None = None
    progress_every: int = 0
    cache_name: str = DEFAULT_CACHE_NAME


@dataclass(frozen=True)
class ActorReplayTrainingResult:
    transition_count: int
    steps: int
    final_loss: float
    final_ratio_mean: float
    checkpoint_path: str
    metrics_path: str
    trainable_param_count: int
    total_param_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def torch_dtype_from_string(value: str) -> torch.dtype:
    normalized = str(value).strip().lower()
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype {value!r}; expected bf16, fp16, or fp32")
    return mapping[normalized]


def tensor_tree_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: tensor_tree_to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_tree_to_cpu(item) for item in value]
    return value


def tensor_tree_nbytes(value) -> int:
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(tensor_tree_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_tree_nbytes(item) for item in value)
    return 0


def check_replay_context_tensor_budget(
    replay_context: dict,
    max_gb: float | None,
    *,
    label: str = "replay_context",
) -> int:
    """Return context tensor bytes, or fail before writing an oversized context."""

    tensor_bytes = tensor_tree_nbytes(replay_context)
    if max_gb is None or max_gb <= 0:
        return tensor_bytes
    max_bytes = int(float(max_gb) * 1024**3)
    if tensor_bytes > max_bytes:
        raise ValueError(
            f"{label} tensor storage is {tensor_bytes / 1024**3:.3f} GiB, "
            f"exceeding strict_grpo_replay_context_max_gb={float(max_gb):.3f}. "
            "Lower STRICT_GRPO_CAPTURE_MAX_CHUNKS/GROUP_SIZE, increase the reviewed budget, "
            "or disable replay-context capture for this run."
        )
    return tensor_bytes


def select_cache_kv_batch(cache_snapshot: Sequence[dict], batch_index: int = 0) -> list[dict]:
    """Clone one CFG branch from attention k/v tensors without slicing masks."""

    selected = []
    for cache in cache_snapshot:
        selected_cache = {}
        for key, value in cache.items():
            if (
                key in {"k", "v"}
                and torch.is_tensor(value)
                and value.ndim > 0
                and value.shape[0] > batch_index
            ):
                selected_cache[key] = value.narrow(0, batch_index, 1).clone()
            else:
                selected_cache[key] = value
        selected.append(selected_cache)
    return selected


def snapshot_transformer_cache(transformer, cache_name: str = DEFAULT_CACHE_NAME) -> list[dict]:
    """Clone the transformer's self-attention KV cache for later replay.

    The replay trainer restores this snapshot before each denoising transition.
    This is required because LingBot-VA action generation is stateful: the action
    transformer attends to cached observation/video/action context.
    """

    snapshots: list[dict] = []
    for block_idx, block in enumerate(transformer.blocks):
        cache = getattr(block.attn1, "attn_caches", None)
        if cache is None or cache_name not in cache or cache[cache_name] is None:
            raise MissingReplayContextError(f"missing transformer cache {cache_name!r} at block {block_idx}")
        snapshots.append(tensor_tree_to_cpu(cache[cache_name]))
    return snapshots


def restore_transformer_cache(
    transformer,
    cache_snapshot: Sequence[dict],
    *,
    cache_name: str = DEFAULT_CACHE_NAME,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> None:
    if len(cache_snapshot) != len(transformer.blocks):
        raise ValueError(f"cache snapshot has {len(cache_snapshot)} blocks, model has {len(transformer.blocks)} blocks")
    for block, cache in zip(transformer.blocks, cache_snapshot, strict=True):
        restored = {}
        for key, value in cache.items():
            if torch.is_tensor(value):
                target_dtype = dtype if key in {"k", "v"} and dtype is not None else value.dtype
                restored[key] = value.to(device=device, dtype=target_dtype)
            else:
                restored[key] = value
        block.attn1.attn_caches[cache_name] = restored


def build_replay_context(
    *,
    transformer,
    cache_name: str,
    action_input_template: dict,
    negative_prompt_embeds: torch.Tensor | None,
    use_cfg: bool,
    action_guidance_scale: float,
    action_num_inference_steps: int,
    frame_chunk_size: int,
) -> dict:
    """Build a chunk-level replay context from live inference state."""

    transformer_cache = snapshot_transformer_cache(transformer, cache_name=cache_name)
    action_uses_cfg = bool(use_cfg and action_guidance_scale > 1.0)
    pruned_to_conditional = bool(use_cfg and not action_uses_cfg)
    if pruned_to_conditional:
        transformer_cache = select_cache_kv_batch(transformer_cache, batch_index=0)

    return {
        "schema_version": REPLAY_CONTEXT_SCHEMA_VERSION,
        "cache_name": cache_name,
        "transformer_cache": transformer_cache,
        "grid_id": tensor_tree_to_cpu(action_input_template["grid_id"]),
        "text_emb": tensor_tree_to_cpu(action_input_template["text_emb"]),
        "negative_text_emb": (
            None
            if negative_prompt_embeds is None or not action_uses_cfg
            else tensor_tree_to_cpu(negative_prompt_embeds)
        ),
        "use_cfg": action_uses_cfg,
        "cfg_pruned_to_conditional": pruned_to_conditional,
        "action_guidance_scale": float(action_guidance_scale),
        "action_num_inference_steps": int(action_num_inference_steps),
        "frame_chunk_size": int(frame_chunk_size),
    }


def build_transition_replay_input(action_input: dict) -> dict:
    """Save the exact per-transition action model input needed for replay."""

    return {
        # `noisy_latents` is the same denoising state already stored as
        # transition["action_xt"]. Keeping only timesteps avoids duplicating a
        # large tensor while preserving frame-specific action-conditioning
        # timesteps that cannot always be reconstructed from the scalar t.
        "timesteps": tensor_tree_to_cpu(action_input["timesteps"]),
    }


def iter_actor_replay_examples(
    groups_jsonl: Path,
    *,
    loader: Callable[[Path], dict] | None = None,
    require_replay_context: bool = True,
) -> Iterable[ActorReplayExample]:
    for ref in read_transition_refs(groups_jsonl.expanduser()):
        artifact = load_strict_artifact(Path(ref.artifact_path), loader=loader)
        replay_context = resolve_replay_context(artifact, Path(ref.artifact_path), loader=loader)
        if replay_context is None:
            if require_replay_context:
                raise MissingReplayContextError(
                    "strict artifact does not contain replay_context; "
                    f"real actor replay cannot use old smoke-only artifact: {ref.artifact_path}"
                )
            continue
        for transition in iter_strict_artifact_transitions(artifact):
            if "replay_input" not in transition:
                raise MissingReplayContextError(
                    "strict artifact transition does not contain replay_input; "
                    f"cannot recompute current actor logprob: {ref.artifact_path}"
                )
            yield ActorReplayExample(ref=ref, transition=transition, replay_context=replay_context)


def count_actor_replay_transition_items(groups_jsonl: Path, *, loader: Callable[[Path], dict] | None = None) -> int:
    """Count replay transition items without loading external replay contexts."""

    count = 0
    for ref in read_transition_refs(groups_jsonl.expanduser()):
        artifact_path = Path(ref.artifact_path)
        artifact = load_strict_artifact(artifact_path, loader=loader)
        if "replay_context" not in artifact and "replay_context_path" not in artifact:
            raise MissingReplayContextError(
                "strict artifact does not contain replay_context; "
                f"real actor replay cannot use old smoke-only artifact: {ref.artifact_path}"
            )
        for transition in iter_strict_artifact_transitions(artifact):
            if "replay_input" not in transition:
                raise MissingReplayContextError(
                    "strict artifact transition does not contain replay_input; "
                    f"cannot recompute current actor logprob: {ref.artifact_path}"
                )
            count += _stored_batch_size(transition["old_logprob_sum"])
    return count


class LingBotActionReplayPolicy(torch.nn.Module):
    """Replay saved denoising transitions through the current LingBot transformer."""

    def __init__(
        self,
        transformer: torch.nn.Module,
        *,
        action_scheduler: FlowMatchScheduler,
        device: torch.device,
        dtype: torch.dtype,
        transition_std_floor: float | None = None,
        cache_name: str = DEFAULT_CACHE_NAME,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.action_scheduler = action_scheduler
        self.device = device
        self.dtype = dtype
        self.transition_std_floor = transition_std_floor
        self.cache_name = cache_name

    def forward_example(self, example: ActorReplayExample) -> torch.Tensor:
        return self.forward_transition_logprob(example).logprob_sum

    def forward_transition_logprob(self, example: ActorReplayExample) -> TransitionLogprob:
        transition = example.transition
        transition_mean = self.predict_transition_mean(example)
        return compute_gaussian_transition_logprob(
            transition_mean=transition_mean,
            action_xt_next=_to_tensor(transition["action_xt_next"], device=self.device, dtype=self.dtype),
            transition_std=_floored_transition_std(
                transition["transition_std"],
                floor=self.transition_std_floor,
                device=self.device,
            ),
            logprob_mask=_to_tensor(transition["logprob_mask"], device=self.device, dtype=torch.bool),
        )

    def predict_transition_mean(self, example: ActorReplayExample) -> torch.Tensor:
        transition = example.transition
        context = example.replay_context
        replay_input = transition["replay_input"]
        cache_name = str(context.get("cache_name") or self.cache_name)

        restore_transformer_cache(
            self.transformer,
            context["transformer_cache"],
            cache_name=cache_name,
            device=self.device,
            dtype=self.dtype,
        )
        input_dict = self._build_transformer_input(context, transition, replay_input)
        action_noise_pred = self.transformer(
            input_dict,
            update_cache=0,
            cache_name=cache_name,
            action_mode=True,
        )
        frame_chunk_size = int(context["frame_chunk_size"])
        action_noise_pred = _restore_action_noise_layout(action_noise_pred, frame_chunk_size)

        use_cfg = bool(context.get("use_cfg", False))
        action_guidance_scale = float(context.get("action_guidance_scale", 1.0))
        if use_cfg and action_noise_pred.shape[0] != 2:
            raise ValueError(f"CFG replay expected batch=2, got {action_noise_pred.shape[0]}")
        if use_cfg and action_guidance_scale > 1.0:
            action_noise_pred = action_noise_pred[1:] + action_guidance_scale * (
                action_noise_pred[:1] - action_noise_pred[1:]
            )
        elif use_cfg:
            action_noise_pred = action_noise_pred[:1]

        action_xt = _to_tensor(transition["action_xt"], device=self.device, dtype=self.dtype)
        timestep = _to_tensor(transition["timestep"], device=self.device, dtype=torch.float32)
        return self.action_scheduler.step(action_noise_pred, timestep, action_xt, return_dict=False)

    def _build_transformer_input(self, context: dict, transition: dict, replay_input: dict) -> dict:
        noisy_latents_source = replay_input.get("noisy_latents", transition["action_xt"])
        noisy_latents = _to_tensor(noisy_latents_source, device=self.device, dtype=self.dtype)
        timesteps = _to_tensor(replay_input["timesteps"], device=self.device, dtype=torch.float32)
        grid_id = _to_tensor(context["grid_id"], device=self.device, dtype=torch.long)
        text_emb = _to_tensor(context["text_emb"], device=self.device, dtype=self.dtype)

        if bool(context.get("use_cfg", False)):
            negative_text_emb = context.get("negative_text_emb")
            if negative_text_emb is None:
                raise MissingReplayContextError("replay_context has use_cfg=True but missing negative_text_emb")
            return {
                "noisy_latents": noisy_latents.repeat(2, 1, 1, 1, 1),
                "timesteps": timesteps[None].repeat(2, 1),
                "grid_id": grid_id[None].repeat(2, 1, 1),
                "text_emb": torch.cat(
                    [text_emb, _to_tensor(negative_text_emb, device=self.device, dtype=self.dtype)],
                    dim=0,
                ),
            }
        return {
            "noisy_latents": noisy_latents,
            "timesteps": timesteps[None] if timesteps.ndim == 1 else timesteps,
            "grid_id": grid_id[None] if grid_id.ndim == 2 else grid_id,
            "text_emb": text_emb,
        }


class ActorReplayGrpoTrainer:
    """GRPO trainer that updates real LingBot transformer parameters."""

    def __init__(
        self,
        config: ActorReplayTrainerConfig,
        *,
        transformer: torch.nn.Module,
    ) -> None:
        if config.steps <= 0:
            raise ValueError("steps must be positive")
        if config.logprob_reduction not in {"sum", "mean"}:
            raise ValueError("logprob_reduction must be 'sum' or 'mean'")
        if config.logprob_std_floor is not None and config.logprob_std_floor <= 0.0:
            raise ValueError("logprob_std_floor must be positive when set")
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = torch_dtype_from_string(config.dtype)
        torch.manual_seed(config.seed)

        self.groups_jsonl = config.groups_jsonl.expanduser()
        self.transition_item_count = count_actor_replay_transition_items(self.groups_jsonl)
        if self.transition_item_count <= 0:
            raise ValueError(f"no actor replay transition items found in {config.groups_jsonl}")

        self.transformer = transformer.to(device=self.device, dtype=self.dtype)
        self.trainable_summary = configure_trainable_parameters(self.transformer, config)
        # Replay should be deterministic with respect to the saved denoising
        # state. Gradients still flow through trainable parameters in eval mode.
        self.transformer.eval()
        self.action_scheduler = FlowMatchScheduler(
            shift=config.action_snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.action_scheduler.set_timesteps(config.action_num_inference_steps)
        self.policy = LingBotActionReplayPolicy(
            self.transformer,
            action_scheduler=self.action_scheduler,
            device=self.device,
            dtype=self.dtype,
            transition_std_floor=config.logprob_std_floor,
            cache_name=config.cache_name,
        )
        params = [param for param in self.transformer.parameters() if param.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=config.learning_rate)

    def train(self) -> ActorReplayTrainingResult:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        history: list[dict[str, float]] = []
        last_metrics: dict[str, float] | None = None

        for step in range(1, self.config.steps + 1):
            self.optimizer.zero_grad(set_to_none=True)
            metrics = self._backward_dataset_loss()
            metrics.update(self._gradient_metrics())
            self._assert_nonzero_finite_gradients(metrics)
            before_step = self._trainable_param_snapshot()
            self.optimizer.step()
            metrics.update(self._parameter_update_metrics(before_step))
            last_metrics = metrics
            history.append({"step": step, **metrics})

        if last_metrics is None:
            raise RuntimeError("training loop did not run")

        checkpoint_path = self.config.output_dir / "checkpoint.pt"
        metrics_path = self.config.output_dir / "metrics.json"
        torch.save(
            {
                "trainable_state_dict": self._trainable_state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": {
                    **asdict(self.config),
                    "groups_jsonl": str(self.config.groups_jsonl),
                    "output_dir": str(self.config.output_dir),
                },
                "trainable_summary": asdict(self.trainable_summary),
                "steps_completed": self.config.steps,
                "history": history,
            },
            checkpoint_path,
        )
        result = ActorReplayTrainingResult(
            transition_count=self.transition_item_count,
            steps=self.config.steps,
            final_loss=float(last_metrics["loss"]),
            final_ratio_mean=float(last_metrics["ratio_mean"]),
            checkpoint_path=str(checkpoint_path),
            metrics_path=str(metrics_path),
            trainable_param_count=self.trainable_summary.trainable_params,
            total_param_count=self.trainable_summary.total_params,
        )
        metrics_path.write_text(
            json.dumps(
                {
                    "result": result.to_dict(),
                    "trainable_summary": asdict(self.trainable_summary),
                    "history": history,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return result

    def _backward_dataset_loss(self) -> dict[str, float]:
        metric_sums = {
            "loss": 0.0,
            "ratio_mean": 0.0,
            "clip_fraction": 0.0,
            "advantage_mean": 0.0,
            "logratio_mean": 0.0,
            "logratio_clamp_fraction": 0.0,
        }
        logratio_min: float | None = None
        logratio_max: float | None = None
        ratio_min: float | None = None
        ratio_max: float | None = None
        processed = 0
        progress_every = max(0, int(self.config.progress_every))
        next_progress = progress_every

        for example in iter_actor_replay_examples(self.groups_jsonl):
            logprob = self.policy.forward_transition_logprob(example)
            if self.config.logprob_reduction == "mean":
                new_logprob = logprob.logprob_mean.flatten()
            else:
                new_logprob = logprob.logprob_sum.flatten()
            batch_size = int(new_logprob.numel())
            old_logprob = self._old_logprob_for_reduction(example.transition, batch_size)
            advantages = torch.full((batch_size,), float(example.ref.advantage), dtype=torch.float32, device=self.device)
            loss, metrics = compute_clipped_grpo_loss(
                new_logprob_sum=new_logprob,
                old_logprob_sum=old_logprob,
                advantages=advantages,
                clip_low=self.config.clip_low,
                clip_high=self.config.clip_high,
            )
            weight = batch_size / self.transition_item_count
            (loss * weight).backward()
            for key in metric_sums:
                metric_sums[key] += float(metrics[key].cpu()) * batch_size
            current_min = float(metrics["ratio_min"].cpu())
            current_max = float(metrics["ratio_max"].cpu())
            ratio_min = current_min if ratio_min is None else min(ratio_min, current_min)
            ratio_max = current_max if ratio_max is None else max(ratio_max, current_max)
            current_logratio_min = float(metrics["logratio_min"].cpu())
            current_logratio_max = float(metrics["logratio_max"].cpu())
            logratio_min = current_logratio_min if logratio_min is None else min(logratio_min, current_logratio_min)
            logratio_max = current_logratio_max if logratio_max is None else max(logratio_max, current_logratio_max)
            processed += batch_size
            if progress_every and (processed >= next_progress or processed == self.transition_item_count):
                print(
                    f"actor replay progress: {processed}/{self.transition_item_count} transition items",
                    flush=True,
                )
                while next_progress <= processed:
                    next_progress += progress_every

        return {
            "loss": metric_sums["loss"] / self.transition_item_count,
            "ratio_mean": metric_sums["ratio_mean"] / self.transition_item_count,
            "ratio_min": float(ratio_min if ratio_min is not None else 0.0),
            "ratio_max": float(ratio_max if ratio_max is not None else 0.0),
            "clip_fraction": metric_sums["clip_fraction"] / self.transition_item_count,
            "advantage_mean": metric_sums["advantage_mean"] / self.transition_item_count,
            "logratio_mean": metric_sums["logratio_mean"] / self.transition_item_count,
            "logratio_min": float(logratio_min if logratio_min is not None else 0.0),
            "logratio_max": float(logratio_max if logratio_max is not None else 0.0),
            "logratio_clamp_fraction": metric_sums["logratio_clamp_fraction"] / self.transition_item_count,
        }

    def _old_logprob_for_reduction(self, transition: dict, batch_size: int) -> torch.Tensor:
        if self.config.logprob_std_floor is not None:
            logprob = compute_gaussian_transition_logprob(
                transition_mean=_to_tensor(transition["transition_mean"], device=self.device, dtype=self.dtype),
                action_xt_next=_to_tensor(transition["action_xt_next"], device=self.device, dtype=self.dtype),
                transition_std=_floored_transition_std(
                    transition["transition_std"],
                    floor=self.config.logprob_std_floor,
                    device=self.device,
                ),
                logprob_mask=_to_tensor(transition["logprob_mask"], device=self.device, dtype=torch.bool),
            )
            if self.config.logprob_reduction == "mean":
                return _as_batch_vector(logprob.logprob_mean, batch_size, self.device)
            return _as_batch_vector(logprob.logprob_sum, batch_size, self.device)
        if self.config.logprob_reduction == "sum":
            return _as_batch_vector(transition["old_logprob_sum"], batch_size, self.device)
        if "old_logprob_mean" in transition:
            return _as_batch_vector(transition["old_logprob_mean"], batch_size, self.device)
        old_sum = _as_batch_vector(transition["old_logprob_sum"], batch_size, self.device)
        old_count = _as_batch_vector(transition["old_logprob_count"], batch_size, self.device).clamp_min(1.0)
        return old_sum / old_count

    def _gradient_metrics(self) -> dict[str, float]:
        total_norm = torch.zeros((), device=self.device)
        grad_tensors = 0
        grad_params = 0
        for param in self.transformer.parameters():
            if not param.requires_grad or param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                raise ValueError("non-finite gradient detected during actor replay GRPO")
            total_norm = total_norm + param.grad.detach().float().pow(2).sum()
            grad_tensors += 1
            grad_params += param.numel()
        return {
            "grad_norm": float(total_norm.sqrt().detach().cpu()),
            "grad_tensor_count": float(grad_tensors),
            "grad_param_count": float(grad_params),
        }

    def _trainable_param_snapshot(self) -> dict[str, torch.Tensor]:
        return {
            name: param.detach().clone()
            for name, param in self.transformer.named_parameters()
            if param.requires_grad
        }

    def _parameter_update_metrics(self, before_step: dict[str, torch.Tensor]) -> dict[str, float]:
        total_norm = torch.zeros((), device=self.device)
        max_abs = torch.zeros((), device=self.device)
        update_params = 0
        for name, param in self.transformer.named_parameters():
            if not param.requires_grad:
                continue
            before = before_step[name]
            delta = (param.detach() - before).float()
            total_norm = total_norm + delta.pow(2).sum()
            if delta.numel():
                max_abs = torch.maximum(max_abs, delta.abs().max())
            update_params += param.numel()
        return {
            "param_update_norm": float(total_norm.sqrt().detach().cpu()),
            "param_update_max": float(max_abs.detach().cpu()),
            "param_update_param_count": float(update_params),
        }

    def _assert_nonzero_finite_gradients(self, metrics: dict[str, float]) -> None:
        if int(metrics["grad_tensor_count"]) == 0 or metrics["grad_norm"] <= 0.0:
            self._write_failure_diagnostics(metrics)
            raise ValueError(
                "real actor replay produced zero gradients for trainable parameters "
                f"(grad_norm={metrics['grad_norm']:.6g}, "
                f"logratio_min={metrics.get('logratio_min', 0.0):.6g}, "
                f"logratio_max={metrics.get('logratio_max', 0.0):.6g}, "
                f"logratio_clamp_fraction={metrics.get('logratio_clamp_fraction', 0.0):.6g})"
            )

    def _write_failure_diagnostics(self, metrics: dict[str, float]) -> None:
        diagnostics_path = self.config.output_dir / "failure_diagnostics.json"
        diagnostics_path.write_text(
            json.dumps(
                {
                    "metrics": metrics,
                    "trainable_summary": asdict(self.trainable_summary),
                    "transition_count": self.transition_item_count,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def _trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: param.detach().cpu()
            for name, param in self.transformer.named_parameters()
            if param.requires_grad
        }


def _to_tensor(value, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    return tensor.to(device=device, dtype=dtype)


def _as_batch_vector(value, batch_size: int, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device).flatten()
    if tensor.numel() == 1:
        return tensor.repeat(batch_size)
    if tensor.numel() != batch_size:
        raise ValueError(f"expected scalar or {batch_size} values, got shape {tuple(tensor.shape)}")
    return tensor


def _floored_transition_std(
    value,
    *,
    floor: float | None,
    device: torch.device,
) -> torch.Tensor:
    std = _to_tensor(value, device=device, dtype=torch.float32)
    if floor is None:
        return std
    return std.clamp_min(float(floor))


def _stored_batch_size(value) -> int:
    return max(int(torch.as_tensor(value).flatten().numel()), 1)


def _restore_action_noise_layout(action_noise_pred: torch.Tensor, frame_chunk_size: int) -> torch.Tensor:
    if action_noise_pred.ndim != 3:
        raise ValueError(f"expected transformer action output with 3 dims, got shape {tuple(action_noise_pred.shape)}")
    if frame_chunk_size <= 0:
        raise ValueError("frame_chunk_size must be positive")
    batch_size, token_count, channel_count = action_noise_pred.shape
    if token_count % frame_chunk_size != 0:
        raise ValueError(
            f"cannot reshape {token_count} action tokens into frame_chunk_size={frame_chunk_size}"
        )
    action_slots = token_count // frame_chunk_size
    return (
        action_noise_pred.reshape(batch_size, frame_chunk_size, action_slots, channel_count)
        .permute(0, 3, 1, 2)
        .unsqueeze(-1)
        .contiguous()
    )

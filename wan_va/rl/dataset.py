"""Offline dataset utilities for denoising-step GRPO training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable


STRICT_ARTIFACT_SCHEMA_SINGLE = 1
STRICT_ARTIFACT_SCHEMA_TRAJECTORY = 2
STRICT_ARTIFACT_SCOPE_SINGLE = "first_action_denoising_step"
STRICT_ARTIFACT_SCOPE_TRAJECTORY = "action_denoising_trajectory"

REQUIRED_STRICT_ARTIFACT_KEYS = frozenset(
    {
        "schema_version",
        "scope",
        "sampling_seed",
        "frame_st_id",
        "timestep",
        "action_xt",
        "action_xt_next",
        "transition_mean",
        "transition_std",
        "old_logprob_sum",
        "old_logprob_mean",
        "old_logprob_count",
        "logprob_mask",
    }
)

REQUIRED_STRICT_TRAJECTORY_ARTIFACT_KEYS = frozenset(
    {
        "schema_version",
        "scope",
        "sampling_seed",
        "frame_st_id",
        "num_transitions",
        "transitions",
    }
)

REQUIRED_STRICT_TRANSITION_KEYS = frozenset(
    {
        "timestep",
        "action_xt",
        "action_xt_next",
        "transition_mean",
        "transition_std",
        "old_logprob_sum",
        "old_logprob_mean",
        "old_logprob_count",
        "logprob_mask",
    }
)

REQUIRED_REPLAY_CONTEXT_KEYS = frozenset(
    {
        "schema_version",
        "cache_name",
        "transformer_cache",
        "grid_id",
        "text_emb",
        "use_cfg",
        "action_guidance_scale",
        "action_num_inference_steps",
        "frame_chunk_size",
    }
)

REQUIRED_REPLAY_INPUT_KEYS = frozenset({"timesteps"})
OPTIONAL_REPLAY_INPUT_KEYS = frozenset({"noisy_latents"})
REPLAY_CONTEXT_INLINE_KEY = "replay_context"
REPLAY_CONTEXT_PATH_KEY = "replay_context_path"


@dataclass(frozen=True)
class GrpoTransitionRef:
    task: str
    group_id: str
    sample_idx: int
    reward: float
    advantage: float
    record_path: str
    artifact_path: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DatasetIssue:
    severity: str
    code: str
    message: str
    group_id: str = ""
    sample_idx: int | None = None
    artifact_path: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DatasetValidationReport:
    transition_count: int
    issues: tuple[DatasetIssue, ...]

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def ok(self) -> bool:
        return self.error_count == 0

    def to_dict(self) -> dict:
        return {
            "transition_count": self.transition_count,
            "error_count": self.error_count,
            "ok": self.ok,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def read_grpo_group_dicts(path: Path) -> Iterable[dict]:
    with path.expanduser().open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_transition_refs(path: Path) -> Iterable[GrpoTransitionRef]:
    for group in read_grpo_group_dicts(path):
        task = str(group.get("task", ""))
        group_id = str(group["group_id"])
        for sample in group.get("samples", []):
            for artifact_path in sample.get("strict_grpo_artifact_paths", []) or []:
                yield GrpoTransitionRef(
                    task=task,
                    group_id=group_id,
                    sample_idx=int(sample["sample_idx"]),
                    reward=float(sample["reward"]),
                    advantage=float(sample["advantage"]),
                    record_path=str(sample["record_path"]),
                    artifact_path=str(artifact_path),
                )


def load_strict_artifact(path: Path, *, loader: Callable[[Path], dict] | None = None) -> dict:
    expanded = path.expanduser()
    if loader is None:
        import torch

        data = torch.load(expanded, map_location="cpu")
    else:
        data = loader(expanded)

    if not isinstance(data, dict):
        raise ValueError(f"strict artifact must be a dict: {expanded}")
    _validate_strict_artifact_schema(data, str(expanded))
    return data


def load_replay_context(path: Path, *, loader: Callable[[Path], dict] | None = None) -> dict:
    expanded = path.expanduser()
    if loader is None:
        import torch

        data = torch.load(expanded, map_location="cpu")
    else:
        data = loader(expanded)

    if not isinstance(data, dict):
        raise ValueError(f"replay context must be a dict: {expanded}")
    return data


def resolve_replay_context(
    data: dict,
    artifact_path: Path | str,
    *,
    loader: Callable[[Path], dict] | None = None,
) -> dict | None:
    """Return inline or externally referenced replay context for an artifact."""

    replay_context = data.get(REPLAY_CONTEXT_INLINE_KEY)
    if replay_context is not None:
        if not isinstance(replay_context, dict):
            raise ValueError(f"strict artifact {artifact_path} field replay_context must be a dict")
        return replay_context

    context_path_value = data.get(REPLAY_CONTEXT_PATH_KEY)
    if context_path_value is None:
        return None
    if not isinstance(context_path_value, str) or not context_path_value:
        raise ValueError(f"strict artifact {artifact_path} field replay_context_path must be a non-empty string")

    context_path = Path(context_path_value).expanduser()
    if not context_path.is_absolute():
        context_path = Path(artifact_path).expanduser().parent / context_path
    return load_replay_context(context_path, loader=loader)


def _validate_strict_artifact_schema(data: dict, path_label: str) -> None:
    schema_version = data.get("schema_version")
    scope = data.get("scope")

    if schema_version == STRICT_ARTIFACT_SCHEMA_SINGLE and scope in {None, STRICT_ARTIFACT_SCOPE_SINGLE}:
        missing = sorted(REQUIRED_STRICT_ARTIFACT_KEYS - set(data))
        if missing:
            raise ValueError(f"strict artifact {path_label} missing required strict artifact keys: {missing}")
        _validate_strict_transition(data, f"{path_label}:transition")
        return

    if schema_version == STRICT_ARTIFACT_SCHEMA_TRAJECTORY and scope in {None, STRICT_ARTIFACT_SCOPE_TRAJECTORY}:
        missing = sorted(REQUIRED_STRICT_TRAJECTORY_ARTIFACT_KEYS - set(data))
        if missing:
            raise ValueError(f"strict artifact {path_label} missing required trajectory artifact keys: {missing}")
        transitions = data["transitions"]
        if not isinstance(transitions, list):
            raise ValueError(f"strict artifact {path_label} field transitions must be a list")
        if not transitions:
            raise ValueError(f"strict artifact {path_label} field transitions must not be empty")
        try:
            expected_count = int(data["num_transitions"])
        except Exception as exc:
            raise ValueError(f"strict artifact {path_label} field num_transitions must be an integer") from exc
        if expected_count != len(transitions):
            raise ValueError(
                f"strict artifact {path_label} num_transitions={expected_count} "
                f"does not match transitions length {len(transitions)}"
            )
        for index, transition in enumerate(transitions):
            if not isinstance(transition, dict):
                raise ValueError(f"strict artifact {path_label} transition {index} must be a dict")
            _validate_strict_transition(transition, f"{path_label}:transition[{index}]")
        return

    raise ValueError(
        f"strict artifact {path_label} has unsupported schema/scope: "
        f"schema_version={schema_version!r}, scope={scope!r}"
    )


def _validate_strict_transition(data: dict, path_label: str) -> None:
    missing = sorted(REQUIRED_STRICT_TRANSITION_KEYS - set(data))
    if missing:
        raise ValueError(f"strict artifact {path_label} missing required transition keys: {missing}")
    _validate_strict_artifact_shapes(data, path_label)
    _validate_strict_artifact_values(data, path_label)


def iter_strict_artifact_transitions(data: dict) -> Iterable[dict]:
    """Yield normalized transition dictionaries from v1 or v2 strict artifacts."""

    schema_version = data.get("schema_version")
    scope = data.get("scope")
    if schema_version == STRICT_ARTIFACT_SCHEMA_SINGLE and scope == STRICT_ARTIFACT_SCOPE_SINGLE:
        yield data
        return

    if schema_version == STRICT_ARTIFACT_SCHEMA_TRAJECTORY and scope == STRICT_ARTIFACT_SCOPE_TRAJECTORY:
        for index, transition in enumerate(data.get("transitions", [])):
            normalized = dict(transition)
            normalized.setdefault("schema_version", schema_version)
            normalized.setdefault("scope", scope)
            normalized.setdefault("sampling_seed", data.get("sampling_seed"))
            normalized.setdefault("frame_st_id", data.get("frame_st_id"))
            normalized.setdefault("denoising_step_index", index)
            yield normalized
        return

    raise ValueError(f"unsupported strict artifact schema/scope: schema_version={schema_version!r}, scope={scope!r}")


def count_strict_artifact_transitions(data: dict) -> int:
    """Return the number of denoising transitions represented by a strict artifact."""

    schema_version = data.get("schema_version")
    scope = data.get("scope")
    if schema_version == STRICT_ARTIFACT_SCHEMA_SINGLE and scope == STRICT_ARTIFACT_SCOPE_SINGLE:
        return 1
    if schema_version == STRICT_ARTIFACT_SCHEMA_TRAJECTORY and scope == STRICT_ARTIFACT_SCOPE_TRAJECTORY:
        return len(data.get("transitions", []))
    raise ValueError(f"unsupported strict artifact schema/scope: schema_version={schema_version!r}, scope={scope!r}")


def _validate_strict_artifact_shapes(data: dict, path_label: str) -> None:
    state_keys = ("action_xt", "action_xt_next", "transition_mean", "logprob_mask")
    state_shapes = {key: _shape_of(data[key]) for key in state_keys}
    known_state_shapes = {shape for shape in state_shapes.values() if shape is not None}
    if len(known_state_shapes) > 1:
        raise ValueError(f"strict artifact {path_label} has incompatible state tensor shapes: {state_shapes}")

    state_shape = next(iter(known_state_shapes), None)
    if state_shape is not None and len(state_shape) == 0:
        raise ValueError(f"strict artifact {path_label} state tensors must have a batch dimension")
    batch_size = None if state_shape is None else state_shape[0]

    for key in ("transition_std", "old_logprob_sum", "old_logprob_mean", "old_logprob_count"):
        shape = _shape_of(data[key])
        if shape is None or batch_size is None:
            continue
        if shape not in {(), (batch_size,)}:
            raise ValueError(
                f"strict artifact {path_label} field {key} must be scalar or batch vector of length {batch_size}; "
                f"got shape {shape}"
            )


def _validate_strict_artifact_values(data: dict, path_label: str) -> None:
    for key in ("action_xt", "action_xt_next", "transition_mean", "old_logprob_sum", "old_logprob_mean", "old_logprob_count"):
        _validate_tensor_finite(data[key], path_label=path_label, key=key)
    _validate_tensor_finite(data["transition_std"], path_label=path_label, key="transition_std", positive=True)


def _shape_of(value: object) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(dim) for dim in shape)
    except TypeError:
        return None


def _validate_tensor_finite(value: object, *, path_label: str, key: str, positive: bool = False) -> None:
    try:
        import torch
    except ImportError:
        return

    try:
        tensor = torch.as_tensor(value)
    except Exception:
        return
    if tensor.numel() == 0:
        raise ValueError(f"strict artifact {path_label} field {key} must not be empty")
    numeric = tensor.float()
    if not torch.isfinite(numeric).all():
        raise ValueError(f"strict artifact {path_label} field {key} contains non-finite values")
    if positive and not (numeric > 0).all():
        raise ValueError(f"strict artifact {path_label} field {key} must be positive")


def validate_transition_refs(
    refs: Iterable[GrpoTransitionRef],
    *,
    require_existing_artifacts: bool = True,
) -> DatasetValidationReport:
    items = list(refs)
    issues: list[DatasetIssue] = []
    for ref in items:
        if require_existing_artifacts and not Path(ref.artifact_path).expanduser().exists():
            issues.append(
                DatasetIssue(
                    severity="error",
                    code="missing_transition_artifact",
                    message=f"transition artifact does not exist: {ref.artifact_path}",
                    group_id=ref.group_id,
                    sample_idx=ref.sample_idx,
                    artifact_path=ref.artifact_path,
                )
            )
    return DatasetValidationReport(transition_count=len(items), issues=tuple(issues))


def inspect_strict_artifacts(
    refs: Iterable[GrpoTransitionRef],
    *,
    loader: Callable[[Path], dict] | None = None,
    require_replay_context: bool = False,
) -> DatasetValidationReport:
    items = list(refs)
    issues: list[DatasetIssue] = []
    transition_count = 0
    for ref in items:
        try:
            artifact = load_strict_artifact(Path(ref.artifact_path), loader=loader)
            if require_replay_context:
                _validate_actor_replay_fields(artifact, ref.artifact_path, loader=loader)
            transition_count += count_strict_artifact_transitions(artifact)
        except Exception as exc:
            issues.append(
                DatasetIssue(
                    severity="error",
                    code="invalid_transition_artifact",
                    message=str(exc),
                    group_id=ref.group_id,
                    sample_idx=ref.sample_idx,
                    artifact_path=ref.artifact_path,
                )
            )
    return DatasetValidationReport(transition_count=transition_count, issues=tuple(issues))


def _validate_actor_replay_fields(
    data: dict,
    path_label: str,
    *,
    loader: Callable[[Path], dict] | None = None,
) -> None:
    replay_context = resolve_replay_context(data, path_label, loader=loader)
    if not isinstance(replay_context, dict):
        raise ValueError(f"strict artifact {path_label} missing replay_context required for actor replay")

    missing_context = sorted(REQUIRED_REPLAY_CONTEXT_KEYS - set(replay_context))
    if missing_context:
        raise ValueError(f"strict artifact {path_label} replay_context missing keys: {missing_context}")

    transformer_cache = replay_context["transformer_cache"]
    if not isinstance(transformer_cache, list) or not transformer_cache:
        raise ValueError(f"strict artifact {path_label} replay_context transformer_cache must be a non-empty list")

    if bool(replay_context.get("use_cfg", False)) and replay_context.get("negative_text_emb") is None:
        raise ValueError(f"strict artifact {path_label} replay_context has use_cfg=True but missing negative_text_emb")

    for key in ("action_guidance_scale", "action_num_inference_steps", "frame_chunk_size"):
        try:
            float(replay_context[key])
        except Exception as exc:
            raise ValueError(f"strict artifact {path_label} replay_context field {key} must be numeric") from exc

    for index, transition in enumerate(iter_strict_artifact_transitions(data)):
        replay_input = transition.get("replay_input")
        if not isinstance(replay_input, dict):
            raise ValueError(f"strict artifact {path_label} transition {index} missing replay_input")
        missing_input = sorted(REQUIRED_REPLAY_INPUT_KEYS - set(replay_input))
        if missing_input:
            raise ValueError(f"strict artifact {path_label} transition {index} replay_input missing keys: {missing_input}")
        _validate_tensor_finite(replay_input["timesteps"], path_label=path_label, key=f"replay_input[{index}].timesteps")
        if "noisy_latents" in replay_input:
            _validate_tensor_finite(
                replay_input["noisy_latents"],
                path_label=path_label,
                key=f"replay_input[{index}].noisy_latents",
            )

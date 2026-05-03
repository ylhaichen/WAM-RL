# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import ast
import os
from pathlib import Path


ENV_TO_CONFIG = {
    "WAN_VA_MODEL_PATH": "wan22_pretrained_model_name_or_path",
    "WAN_VA_DATASET_PATH": "dataset_path",
    "WAN_VA_EMPTY_EMB_PATH": "empty_emb_path",
    "WAN_VA_SAVE_ROOT": "save_root",
    "WAN_VA_ENABLE_WANDB": "enable_wandb",
    "WAN_VA_TRAINABLE_MODE": "trainable_mode",
    "WAN_VA_TRAINABLE_PATTERNS": "trainable_param_patterns",
    "WAN_VA_FROZEN_PATTERNS": "frozen_param_patterns",
}


def _parse_value(value):
    if not isinstance(value, str):
        return value
    lowered = value.strip().lower()
    if lowered in {"true", "yes", "1", "on"}:
        return True
    if lowered in {"false", "no", "0", "off"}:
        return False
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _set_dotted(config, key, value):
    parts = key.split(".")
    cur = config
    for part in parts[:-1]:
        cur = cur[part]
    cur[parts[-1]] = value


def apply_env_overrides(config):
    for env_name, config_key in ENV_TO_CONFIG.items():
        value = os.environ.get(env_name)
        if value is not None:
            _set_dotted(config, config_key, _parse_value(value))

    dataset_path = os.environ.get("WAN_VA_DATASET_PATH")
    if dataset_path and not os.environ.get("WAN_VA_EMPTY_EMB_PATH"):
        config.empty_emb_path = str(Path(dataset_path) / "empty_emb.pt")

    return config


def normalize_overrides(items):
    if not items:
        return []

    out = []
    pending_key = None
    for item in items:
        if "=" in item and pending_key is None:
            key, value = item.split("=", 1)
            out.append((key.lstrip("-"), value))
        elif item.startswith("--") and pending_key is None:
            pending_key = item.lstrip("-")
        elif pending_key is not None:
            out.append((pending_key, item))
            pending_key = None
        else:
            raise ValueError(f"Malformed override: {item}")

    if pending_key is not None:
        raise ValueError(f"Override {pending_key} is missing a value")

    return out


def apply_cli_overrides(config, items):
    seen_keys = set()
    for key, value in normalize_overrides(items):
        _set_dotted(config, key, _parse_value(value))
        seen_keys.add(key)

    if "dataset_path" in seen_keys and "empty_emb_path" not in seen_keys:
        config.empty_emb_path = str(Path(config.dataset_path) / "empty_emb.pt")

    return config


def require_existing_path(config, key, label):
    value = getattr(config, key, None)
    if value is None:
        raise ValueError(f"{label} is not configured")
    path = Path(str(value)).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"{label} does not exist: {path}. Set {key} or the matching WAN_VA_* env var."
        )
    return path

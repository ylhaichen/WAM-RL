#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def _check_path(path, label):
    if path is None:
        print(f"[missing] {label}: not configured")
        return False
    path = Path(path).expanduser()
    if path.exists():
        print(f"[ok] {label}: {path}")
        return True
    print(f"[missing] {label}: {path}")
    return False


def _check_attn_mode(model_path):
    if model_path is None:
        return False
    config_path = Path(model_path).expanduser() / "transformer" / "config.json"
    if not config_path.exists():
        print(f"[missing] transformer config: {config_path}")
        return False
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    print(f"[ok] attn_mode: {config.get('attn_mode')!r}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Check local LingBot-VA paths.")
    parser.add_argument("--model-path")
    parser.add_argument("--dataset-path")
    parser.add_argument("--robotwin-root")
    args = parser.parse_args()

    ok = True
    ok &= _check_path(args.model_path, "model path")
    _check_attn_mode(args.model_path)
    if args.dataset_path:
        ok &= _check_path(args.dataset_path, "dataset path")
        ok &= _check_path(Path(args.dataset_path).expanduser() / "empty_emb.pt", "empty_emb.pt")
    if args.robotwin_root:
        ok &= _check_path(args.robotwin_root, "RoboTwin root")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

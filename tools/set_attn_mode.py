#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Set LingBot-VA transformer attn_mode.")
    parser.add_argument("model_path", help="Checkpoint root containing transformer/config.json")
    parser.add_argument("attn_mode", choices=("flex", "torch", "flashattn"))
    args = parser.parse_args()

    config_path = Path(args.model_path).expanduser() / "transformer" / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing transformer config: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    old_mode = config.get("attn_mode")
    config["attn_mode"] = args.attn_mode

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    print(f"{config_path}: attn_mode {old_mode!r} -> {args.attn_mode!r}")


if __name__ == "__main__":
    main()

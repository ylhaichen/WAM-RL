import json

from tools.build_grpo_groups import build_groups_from_roots, write_group_outputs


def _write_rollout(root, task, group_id, sample_idx, reward):
    rollout_dir = root / "rollouts" / task
    rollout_dir.mkdir(parents=True, exist_ok=True)
    path = rollout_dir / f"{group_id}_episode_{sample_idx:06d}.json"
    path.write_text(
        json.dumps(
            {
                "task_name": task,
                "seed": 10000,
                "episode_index": sample_idx,
                "success": reward > 0.0,
                "reward": reward,
                "group_id": group_id,
                "sample_idx": sample_idx,
                "group_size": 2,
                "sampling_seed": 900000 + sample_idx,
                "strict_grpo_ready": True,
                "strict_grpo_artifact_paths": [str(root / f"{group_id}_strict_{sample_idx}.pt")],
            }
        ),
        encoding="utf-8",
    )


def test_build_grpo_groups_tool_writes_jsonl_and_summary(tmp_path):
    _write_rollout(tmp_path, "open_microwave", "mixed", 0, 0.0)
    _write_rollout(tmp_path, "open_microwave", "mixed", 1, 1.0)
    _write_rollout(tmp_path, "open_microwave", "all_success", 0, 1.0)
    _write_rollout(tmp_path, "open_microwave", "all_success", 1, 1.0)

    result = build_groups_from_roots([tmp_path], expected_group_size=2)
    out_jsonl = tmp_path / "groups" / "grpo_groups.jsonl"
    out_summary = tmp_path / "groups" / "grpo_summary.json"
    write_group_outputs(result, out_jsonl=out_jsonl, out_summary=out_summary)

    lines = out_jsonl.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    group = json.loads(lines[0])
    assert group["group_id"] == "mixed"
    assert [sample["advantage"] for sample in group["samples"]] == [-1.0, 1.0]

    summary = json.loads(out_summary.read_text(encoding="utf-8"))
    assert summary["total_groups"] == 2
    assert summary["mixed_groups"] == 1
    assert summary["skipped_all_success"] == 1
    assert summary["mixed_group_ratio"] == 0.5


def test_build_grpo_groups_tool_writes_manifest_with_validation(tmp_path):
    for group_id, sample_idx in (("mixed", 0), ("mixed", 1)):
        (tmp_path / f"{group_id}_strict_{sample_idx}.pt").write_bytes(b"pt")
    _write_rollout(tmp_path, "open_microwave", "mixed", 0, 0.0)
    _write_rollout(tmp_path, "open_microwave", "mixed", 1, 1.0)

    result = build_groups_from_roots([tmp_path], expected_group_size=2)
    out_jsonl = tmp_path / "groups" / "grpo_groups.jsonl"
    out_summary = tmp_path / "groups" / "grpo_summary.json"
    out_manifest = tmp_path / "groups" / "grpo_manifest.json"
    write_group_outputs(
        result,
        out_jsonl=out_jsonl,
        out_summary=out_summary,
        out_manifest=out_manifest,
        roots=[tmp_path],
        expected_group_size=2,
        require_strict_artifacts=True,
        require_existing_artifacts=True,
    )

    manifest = json.loads(out_manifest.read_text(encoding="utf-8"))
    assert manifest["validation"]["ok"] is True
    assert manifest["record_count"] == 2
    assert manifest["outputs"]["summary_json"].endswith("grpo_summary.json")

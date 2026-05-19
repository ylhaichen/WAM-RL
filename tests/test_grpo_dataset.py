import json

from wan_va.rl.dataset import (
    REQUIRED_STRICT_ARTIFACT_KEYS,
    STRICT_ARTIFACT_SCOPE_TRAJECTORY,
    count_strict_artifact_transitions,
    inspect_strict_artifacts,
    iter_strict_artifact_transitions,
    load_strict_artifact,
    read_transition_refs,
    validate_transition_refs,
)


class ShapeOnly:
    def __init__(self, shape):
        self.shape = shape


def _strict_artifact_with_shapes(
    *,
    state_shape=(1, 2, 3),
    next_shape=None,
    mask_shape=None,
    vector_shape=(1,),
):
    next_shape = state_shape if next_shape is None else next_shape
    mask_shape = state_shape if mask_shape is None else mask_shape
    artifact = {
        **{key: object() for key in REQUIRED_STRICT_ARTIFACT_KEYS},
        "schema_version": 1,
        "scope": "first_action_denoising_step",
    }
    artifact.update(
        {
            "action_xt": ShapeOnly(state_shape),
            "action_xt_next": ShapeOnly(next_shape),
            "transition_mean": ShapeOnly(state_shape),
            "logprob_mask": ShapeOnly(mask_shape),
            "transition_std": ShapeOnly(()),
            "old_logprob_sum": ShapeOnly(vector_shape),
            "old_logprob_mean": ShapeOnly(vector_shape),
            "old_logprob_count": ShapeOnly(vector_shape),
        }
    )
    return artifact


def _strict_trajectory_artifact_with_shapes(transition_count=2, **kwargs):
    transitions = []
    for index in range(transition_count):
        transition = _strict_artifact_with_shapes(**kwargs)
        transition.pop("schema_version", None)
        transition.pop("scope", None)
        transition.pop("sampling_seed", None)
        transition.pop("frame_st_id", None)
        transition["denoising_step_index"] = index
        transitions.append(transition)
    return {
        "schema_version": 2,
        "scope": STRICT_ARTIFACT_SCOPE_TRAJECTORY,
        "sampling_seed": 123,
        "frame_st_id": 0,
        "num_transitions": transition_count,
        "transitions": transitions,
    }


def test_read_transition_refs_flattens_group_samples(tmp_path):
    path = tmp_path / "grpo_groups.jsonl"
    path.write_text(
        json.dumps(
            {
                "group_id": "g0",
                "task": "open_microwave",
                "samples": [
                    {
                        "sample_idx": 0,
                        "reward": 0.0,
                        "advantage": -1.0,
                        "record_path": "/tmp/r0.json",
                        "strict_grpo_artifact_paths": ["/tmp/a0.pt"],
                    },
                    {
                        "sample_idx": 1,
                        "reward": 1.0,
                        "advantage": 1.0,
                        "record_path": "/tmp/r1.json",
                        "strict_grpo_artifact_paths": ["/tmp/a1.pt", "/tmp/a1b.pt"],
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    refs = list(read_transition_refs(path))

    assert [(ref.sample_idx, ref.artifact_path, ref.advantage) for ref in refs] == [
        (0, "/tmp/a0.pt", -1.0),
        (1, "/tmp/a1.pt", 1.0),
        (1, "/tmp/a1b.pt", 1.0),
    ]


def test_load_strict_artifact_validates_required_keys(tmp_path):
    artifact = {
        **{key: object() for key in REQUIRED_STRICT_ARTIFACT_KEYS},
        "schema_version": 1,
        "scope": "first_action_denoising_step",
    }
    loaded = load_strict_artifact(tmp_path / "strict.pt", loader=lambda path: artifact)

    assert loaded is artifact


def test_load_strict_artifact_accepts_trajectory_schema(tmp_path):
    artifact = _strict_trajectory_artifact_with_shapes(transition_count=3)

    loaded = load_strict_artifact(tmp_path / "strict.pt", loader=lambda path: artifact)

    assert loaded is artifact
    assert count_strict_artifact_transitions(loaded) == 3
    transitions = list(iter_strict_artifact_transitions(loaded))
    assert len(transitions) == 3
    assert transitions[0]["scope"] == STRICT_ARTIFACT_SCOPE_TRAJECTORY
    assert transitions[0]["sampling_seed"] == 123


def test_load_strict_artifact_rejects_missing_keys(tmp_path):
    try:
        load_strict_artifact(tmp_path / "strict.pt", loader=lambda path: {"schema_version": 1})
    except ValueError as exc:
        assert "missing required strict artifact keys" in str(exc)
    else:
        raise AssertionError("expected missing-key ValueError")


def test_load_strict_artifact_rejects_trajectory_count_mismatch(tmp_path):
    artifact = _strict_trajectory_artifact_with_shapes(transition_count=2)
    artifact["num_transitions"] = 3

    try:
        load_strict_artifact(tmp_path / "strict.pt", loader=lambda path: artifact)
    except ValueError as exc:
        assert "does not match transitions length" in str(exc)
    else:
        raise AssertionError("expected count-mismatch ValueError")


def test_load_strict_artifact_rejects_incompatible_state_shapes(tmp_path):
    try:
        load_strict_artifact(
            tmp_path / "strict.pt",
            loader=lambda path: _strict_artifact_with_shapes(next_shape=(1, 2, 4)),
        )
    except ValueError as exc:
        assert "incompatible state tensor shapes" in str(exc)
    else:
        raise AssertionError("expected incompatible-shape ValueError")


def test_load_strict_artifact_rejects_bad_logprob_vector_shape(tmp_path):
    try:
        load_strict_artifact(
            tmp_path / "strict.pt",
            loader=lambda path: _strict_artifact_with_shapes(vector_shape=(2,)),
        )
    except ValueError as exc:
        assert "old_logprob_sum" in str(exc)
    else:
        raise AssertionError("expected bad-vector-shape ValueError")


def test_validate_transition_refs_reports_missing_files(tmp_path):
    path = tmp_path / "grpo_groups.jsonl"
    path.write_text(
        json.dumps(
            {
                "group_id": "g0",
                "task": "open_microwave",
                "samples": [
                    {
                        "sample_idx": 0,
                        "reward": 1.0,
                        "advantage": 1.0,
                        "record_path": "/tmp/r0.json",
                        "strict_grpo_artifact_paths": [str(tmp_path / "missing.pt")],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = validate_transition_refs(read_transition_refs(path), require_existing_artifacts=True)

    assert report.ok is False
    assert report.error_count == 1
    assert report.issues[0].code == "missing_transition_artifact"


def test_inspect_strict_artifacts_reports_schema_errors(tmp_path):
    path = tmp_path / "grpo_groups.jsonl"
    artifact_path = tmp_path / "strict.pt"
    artifact_path.write_bytes(b"pt")
    path.write_text(
        json.dumps(
            {
                "group_id": "g0",
                "task": "open_microwave",
                "samples": [
                    {
                        "sample_idx": 0,
                        "reward": 1.0,
                        "advantage": 1.0,
                        "record_path": "/tmp/r0.json",
                        "strict_grpo_artifact_paths": [str(artifact_path)],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = inspect_strict_artifacts(
        read_transition_refs(path),
        loader=lambda _: {"schema_version": 1},
    )

    assert report.ok is False
    assert report.issues[0].code == "invalid_transition_artifact"


def test_inspect_strict_artifacts_counts_trajectory_transitions(tmp_path):
    path = tmp_path / "grpo_groups.jsonl"
    artifact_path = tmp_path / "strict.pt"
    artifact_path.write_bytes(b"pt")
    path.write_text(
        json.dumps(
            {
                "group_id": "g0",
                "task": "open_microwave",
                "samples": [
                    {
                        "sample_idx": 0,
                        "reward": 1.0,
                        "advantage": 1.0,
                        "record_path": "/tmp/r0.json",
                        "strict_grpo_artifact_paths": [str(artifact_path)],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = inspect_strict_artifacts(
        read_transition_refs(path),
        loader=lambda _: _strict_trajectory_artifact_with_shapes(transition_count=4),
    )

    assert report.ok is True
    assert report.transition_count == 4

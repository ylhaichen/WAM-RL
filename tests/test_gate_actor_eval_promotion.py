import json
import subprocess
import sys

from tools.gate_actor_eval_promotion import gate_actor_eval_promotion


def _comparison(*, matched=20, improved=4, regressed=1, candidate="actor"):
    return {
        "matched_episode_count": matched,
        "pairwise_vs_first": [
            {
                "baseline": "baseline",
                "candidate": candidate,
                "matched_episode_count": matched,
                "improved_count": improved,
                "regressed_count": regressed,
                "same_success_count": 10,
                "same_failure_count": 5,
                "net_improvement_count": improved - regressed,
            }
        ],
    }


def _repeatability(*, matched=20, flip_rate=0.05):
    return {
        "matched_episode_count": matched,
        "flip_rate": flip_rate,
        "flipped_count": int(round(matched * flip_rate)),
    }


def test_gate_blocks_tiny_repeatability_dominated_eval():
    decision = gate_actor_eval_promotion(
        _comparison(matched=2, improved=0, regressed=2),
        _repeatability(matched=2, flip_rate=0.5),
    )

    assert decision["decision"] == "blocked"
    assert any("matched eval episodes 2 < required 10" in item for item in decision["blockers"])
    assert any("flip_rate 0.5 > allowed 0.1" in item for item in decision["blockers"])
    assert decision["metrics"]["net_improvement_count"] == -2


def test_gate_promotes_when_eval_exceeds_repeatability_noise():
    decision = gate_actor_eval_promotion(
        _comparison(matched=40, improved=8, regressed=1),
        _repeatability(matched=30, flip_rate=0.03),
    )

    assert decision["decision"] == "promote"
    assert decision["blockers"] == []
    assert decision["warnings"] == ["candidate has 1 matched regressions"]
    assert decision["metrics"]["net_improvement_rate"] == 7 / 40


def test_gate_requires_candidate_when_multiple_pairwise_entries():
    comparison = _comparison(candidate="actor_a")
    comparison["pairwise_vs_first"].append(dict(comparison["pairwise_vs_first"][0], candidate="actor_b"))

    try:
        gate_actor_eval_promotion(comparison, _repeatability())
    except ValueError as exc:
        assert "pass --candidate" in str(exc)
    else:
        raise AssertionError("expected candidate-selection ValueError")

    decision = gate_actor_eval_promotion(comparison, _repeatability(), candidate="actor_b")
    assert decision["candidate"] == "actor_b"


def test_gate_cli_writes_json_and_markdown(tmp_path):
    comparison = tmp_path / "comparison.json"
    repeatability = tmp_path / "repeatability.json"
    out_json = tmp_path / "gate.json"
    out_md = tmp_path / "gate.md"
    comparison.write_text(json.dumps(_comparison()) + "\n", encoding="utf-8")
    repeatability.write_text(json.dumps(_repeatability()) + "\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "tools/gate_actor_eval_promotion.py",
            "--comparison",
            str(comparison),
            "--baseline-repeatability",
            str(repeatability),
            "--out-json",
            str(out_json),
            "--out-markdown",
            str(out_md),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(out_json.read_text())["decision"] == "promote"
    assert "Actor Eval Promotion Gate" in out_md.read_text()

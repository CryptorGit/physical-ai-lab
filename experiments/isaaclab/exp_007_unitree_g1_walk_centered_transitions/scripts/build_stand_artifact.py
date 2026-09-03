"""Build the reference-only Stage 1 artifact after a verified formal PASS."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
RESULT = REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage1_stand_formal"
ARTIFACT = REPO / "artifacts/exp_007_unitree_g1_walk_centered_transitions/stand_home_state_v1"
CHECKPOINT_RELATIVE = Path(
    "logs/rsl_rl/physical_ai_g1_flat_run/"
    "2026-07-17_21-40-39_stage2_1024_750/model_4246.pt"
)
EXPECTED_SHA256 = "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(name: str):
    return json.loads((RESULT / name).read_text(encoding="utf-8"))


def write_json(name: str, value) -> None:
    (ARTIFACT / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    gate = load("gate.json")
    summary = load("summary.json")
    routing = load("routing_preflight.json")
    if gate["status"] != "PASS" or not gate["eligible_for_stage2"]:
        raise RuntimeError("Stage 1 formal gate is not eligible for artifact creation")
    checkpoint = REPO / CHECKPOINT_RELATIVE
    if sha256(checkpoint) != EXPECTED_SHA256:
        raise RuntimeError("checkpoint hash changed after formal evaluation")
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    action_order = json.loads(
        (REPO / "results/exp_007_unitree_g1_walk_centered_transitions/"
        "stage0_expert_audit/action_order.json").read_text(encoding="utf-8")
    )
    write_json("capability_entry.json", {
        "STAND": {
            "status": "PASS",
            "role": "HOME_STATE",
            "expert": "stage2_model_4246",
            "artifact": "stand_home_state_v1",
        }
    })
    write_json("formal_metrics.json", {
        "stage": 1,
        "status": gate["status"],
        "eligible_for_stage2": gate["eligible_for_stage2"],
        "episodes": 50,
        "metrics": gate["metrics"],
        "thresholds": gate["thresholds"],
        "retention": summary["retention"],
        "failure_counts": load("failure_counts.json"),
    })
    write_json("evaluation_config.json", summary["evaluation_config"])
    write_json("checkpoint_reference.json", {
        "path": CHECKPOINT_RELATIVE.as_posix(),
        "sha256": EXPECTED_SHA256,
        "checkpoint_copied": False,
        "checkpoint_in_artifact": False,
        "source": "exp_005 Stage 2 model_4246",
    })
    write_json("action_order.json", action_order)
    write_json("routing_contract.json", {
        "active_state": "STAND",
        "action": "frozen_stage2_expert_action",
        "router_invoked": False,
        "run_expert_contribution": "BITWISE_ZERO",
        "run_residual": "BITWISE_ZERO",
        "transition_bridge_contribution": "BITWISE_ZERO",
        "transition_bridge_invoked": False,
        "crouch_offset": "BITWISE_ZERO",
        "step_over_offset": "BITWISE_ZERO",
        "land_offset": "BITWISE_ZERO",
        "other_scripted_offset": "BITWISE_ZERO",
        "formal_preflight": routing,
    })
    (ARTIFACT / "reproduction_command.txt").write_text(
        "cd \"$HOME\\workspace\\physical-ai-lab\"\n"
        ".\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions"
        "\\scripts\\evaluate_stand.ps1\n",
        encoding="utf-8",
    )
    source_files = [
        EXP / "scripts/evaluate_stand.py",
        EXP / "src/g1_walk_centered/experts/adapters.py",
        EXP / "src/g1_walk_centered/experts/walk_expert.py",
        EXP / "src/g1_walk_centered/tasks/evaluation.py",
    ]
    write_json("source_revision.json", {
        "evaluation_parent_git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "stage1_commit_message": "Formalize exp_007 STAND home state",
        "evaluated_worktree_files": {
            str(path.relative_to(REPO)).replace("\\", "/"): sha256(path)
            for path in source_files
        },
        "note": "Formal evaluation ran from these exact worktree file hashes before the PASS-only commit.",
    })
    artifact_files = sorted(path for path in ARTIFACT.iterdir() if path.name != "SHA256SUMS")
    lines = [f"{sha256(path)}  {path.name}" for path in artifact_files]
    (ARTIFACT / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"artifact={ARTIFACT}")
    print(f"files={len(artifact_files)} checkpoint_copied=false")


if __name__ == "__main__":
    main()

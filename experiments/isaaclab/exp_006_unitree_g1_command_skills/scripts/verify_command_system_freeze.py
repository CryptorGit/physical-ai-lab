"""Static final-freeze checks for exp_006 command_system_v1."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
sys.path.insert(0, str(EXP / "src"))

from g1_command_skills.command_system import (  # noqa: E402
    CommandSystemRouter, ControllerState, SUPPORTED_TRANSITIONS,
)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path):
    digest = hashlib.sha256(path.read_bytes())
    return digest.hexdigest()


def main():
    artifact = REPO / "artifacts/exp_006_unitree_g1_command_skills/command_system_v1"
    manifest = load(artifact / "capability_manifest.json")
    graph = load(artifact / "transition_graph.json")
    routing = load(artifact / "controller_routing_config.json")
    formal = load(artifact / "formal_sequence_results.json")
    protected = load(artifact / "protected_tensor_hashes.json")
    unsupported = load(artifact / "unsupported_request_safety_results.json")
    sha_mismatches = []
    for line in (artifact / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        actual = sha256(artifact / name)
        if actual != expected:
            sha_mismatches.append({"file": name, "expected": expected, "actual": actual})
    graph_edges = sorted(f"{source.value}->{target.value}" for source, target in SUPPORTED_TRANSITIONS)
    router = CommandSystemRouter(ControllerState.RUN)
    trainable_values = [value for value in vars(router).values() if hasattr(value, "requires_grad")]
    checks = {
        "sha256_all_match": not sha_mismatches,
        "manifest_pass": manifest["status"] == "PASS",
        "stop_is_prototype": manifest["skills"]["STOP"]["status"] == "PROTOTYPE",
        "land_not_supported": manifest["skills"]["LAND"]["status"] == "NOT_SUPPORTED",
        "land_drop_is_observation_only": manifest["skills"]["LAND"]["observation_is_not_a_skill_claim"] is True,
        "crouch_range_exact": manifest["skills"]["CROUCH_SHALLOW"]["supported_depth_m"] == [0.08, 0.10],
        "step_over_not_supported": manifest["skills"]["STEP_OVER"]["status"] == "NOT_SUPPORTED",
        "transition_graph_matches_code": sorted(graph["supported"]) == graph_edges,
        "unsupported_graph_matches_manifest": sorted(graph["unsupported"]) == sorted(manifest["unsupported_transitions"]),
        "formal_gates_pass": all(formal[name]["gate_pass"] for name in ("stand", "run_turn_run", "stand_crouch_stand")),
        "unsupported_requests_pass": unsupported["pass_rate"] == 1.0 and unsupported["all_actions_bitwise_unchanged"],
        "protected_58_tensor_hash": protected["tensor_hash_verified"] and protected["protected_actor_route_hash"] == "c903898bd8422b00d8187d9f86a7c1fc8ed33b9d2d54aa734aa8eba70db47f2a",
        "action_equivalence": protected["action_equivalence_verified"] and protected["stop_action_immutability_verified"],
        "observation_dimension_152": routing["observation_dimension"] == 152,
        "policy_skill_one_hot_6": routing["policy_skill_one_hot_dimension"] == 6,
        "stand_external_controller": routing["stand_is_external_controller_state"] is True,
        "router_parameter_free": not trainable_values,
    }
    report = {"schema_version": 1, "checks": checks, "sha_mismatches": sha_mismatches, "passed": all(checks.values())}
    output = REPO / "results/exp_006_unitree_g1_command_skills/command_system_v1/freeze_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

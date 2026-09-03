"""Fail-closed validation for the frozen POST_RUN_WALK Pilot 1 config."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import yaml

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
CFG_PATH = EXP / "configs/stage0_post_run_walk_pilot1.yaml"
OUT = REPO / "results/exp_010_unitree_g1_post_run_walk_attractor/stage0_prepilot_protocol"
EXPECTED = {
    "stand": (
        "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt",
        "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621",
    ),
    "stand_to_walk": (
        "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-47-23_stage3_stand_to_walk_pilot1_validrun_1024_100/model_0.pt",
        "511b3c832e2c7bc54f348391db79d4ac57d1ceda10a6907549a50da093a9c36e",
    ),
    "walk": (
        "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100/model_100.pt",
        "9098eaea68d1e5f818b87a499f53c17ccc97c3dbb1afa7a7604ed3dcaa3ef7fa",
    ),
    "run_low": (
        "logs/rsl_rl/physical_ai_g1_command_skills/2026-07-19_22-02-41_pilot_turn90_right_residual_from_model0/model_0.pt",
        "60675a79187ef00935069f3bf85293e72477fa1d408b80d3325232d805b0d266",
    ),
    "walk_to_run": (
        "results/exp_007_unitree_g1_walk_centered_transitions/stage7r8_walk_to_run_pilot2_saturation/checkpoints/model_100.pt",
        "d94a94409ed9651734ae8ebc85313b827a809de139135115e833c76ca3bb9fd0",
    ),
    "stage8c_model_10": (
        "results/exp_007_unitree_g1_walk_centered_transitions/stage8c_run_to_walk_pilot1_execution/checkpoints/model_10.pt",
        "f54ead0da2a192e238e1fd6dbcb48670fb785f7ef7e7766c64d0dfbf06eba263",
    ),
}


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    hashes = {name: file_sha(REPO / path) for name, (path, _) in EXPECTED.items()}
    required_rewards = {
        "speed_tracking",
        "heading_tracking",
        "upright",
        "stable_support",
        "excessive_flight",
        "dangerous_slip",
        "impact",
        "long_dwell_saturation",
        "fall",
        "action_rate",
    }
    forbidden = {"walk_action_alignment", "walk_imitation", "completion_bonus"}
    flattened = json.dumps(cfg)
    checks = {
        "all_protected_checkpoints_exist": all((REPO / path).is_file() for path, _ in EXPECTED.values()),
        "all_protected_checkpoint_hashes_match": all(
            hashes[name] == expected for name, (_, expected) in EXPECTED.items()
        ),
        "source_speeds_exact": cfg["source"]["speeds_mps"] == [2.6, 2.8],
        "source_probabilities_exact": cfg["source"]["probabilities"] == [0.5, 0.5],
        "target_speed_exact": cfg["target"]["speed_mps"] == 1.2,
        "hold_eight_seconds": cfg["target"]["success_hold_seconds"] == 8.0,
        "physical_envs_1024": cfg["experiment"]["physical_envs"] == 1024,
        "cohort_512": cfg["experiment"]["cohort_size"] == 512,
        "iterations_100": cfg["experiment"]["iterations"] == 100,
        "maximum_two_pilots": cfg["experiment"]["maximum_pilots"] == 2,
        "actor_152_to_37": cfg["actor"]["observation_dimension"] == 152
        and cfg["actor"]["action_dimension"] == 37,
        "full_actor_not_residual": cfg["actor"]["trainable"] == "all_post_run_walk_actor_parameters",
        "parent_model_10": cfg["actor"]["parent_sha256"] == EXPECTED["stage8c_model_10"][1],
        "reward_terms_exact": set(cfg["reward"]) == required_rewards,
        "no_walk_imitation_reward": not any(term in flattened for term in forbidden),
        "source_no_grad_no_storage": not cfg["source"]["source_preparation_gradients"]
        and not cfg["source"]["source_preparation_storage"],
        "in_place_history_contract": cfg["runtime"]["in_place_env_id_cohort"]
        and cfg["runtime"]["preserve_previous_action"]
        and cfg["runtime"]["preserve_contact_history"]
        and cfg["runtime"]["preserve_sensor_history"],
        "no_copy_setter_teleport": not cfg["runtime"]["state_copy_allowed"]
        and not cfg["runtime"]["setter_allowed"]
        and not cfg["runtime"]["teleport_allowed"],
        "runtime_overrides_disabled": not cfg["runtime"]["cli_overrides_allowed"]
        and not cfg["runtime"]["environment_overrides_allowed"],
        "production_disabled": not cfg["runtime"]["production_enablement"],
    }
    config_sha = digest(cfg)
    reward_sha = digest({"reward": cfg["reward"], "thresholds": cfg["reward_thresholds"]})
    status = "FROZEN_READY_FOR_PILOT1" if all(checks.values()) else "FREEZE_FAILED"
    result = {
        "status": status,
        "checks": checks,
        "config_path": str(CFG_PATH.relative_to(REPO)).replace("\\", "/"),
        "config_sha256": config_sha,
        "reward_sha256": reward_sha,
        "protected_hashes": hashes,
        "starting_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "config_validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (OUT / "frozen_protocol_hashes.json").write_text(
        json.dumps(
            {
                "config_sha256": config_sha,
                "reward_sha256": reward_sha,
                "parent_checkpoint_sha256": hashes["stage8c_model_10"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (OUT / "freeze_declaration.json").write_text(
        json.dumps(
            {
                "classification": status,
                "pilot_iterations_executed": 0,
                "optimizer_updates": 0,
                "actor_architecture": "PostRunWalkExpert152",
                "original_walk_basin_required": False,
                "capability_manifest_changed": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    if status != "FROZEN_READY_FOR_PILOT1":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

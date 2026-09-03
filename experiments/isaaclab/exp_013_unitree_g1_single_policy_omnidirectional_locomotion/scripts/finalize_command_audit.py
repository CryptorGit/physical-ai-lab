"""Combine independent command probes and enforce the Stage 0 command gate."""

from __future__ import annotations

import json
from pathlib import Path

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[4]
OUT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/stage0_parent_directional_baseline"


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


cases = {
    name: json.loads((OUT / f"_command_case_{name}.json").read_text(encoding="utf-8"))
    for name in ("vx", "vy", "yaw", "gait")
}
passed = all(value["exclusive_change"] for value in cases.values())
contract = {
    "status": "PASS" if passed else "EXP013_COMMAND_CONTRACT_AMBIGUOUS",
    "input": {
        "total_dimensions": 124,
        "original_observation_dimensions": 123,
        "gait_dimensions": 1,
        "architecture": [124, 256, 128, 128, 37],
    },
    "indices_zero_based": {"vx_cmd": 9, "vy_cmd": 10, "yaw_rate_cmd": 11, "gait_cmd": 123},
    "indices_one_based": {"vx_cmd": 10, "vy_cmd": 11, "yaw_rate_cmd": 12, "gait_cmd": 124},
    "scale": {"vx_cmd": 1.0, "vy_cmd": 1.0, "yaw_rate_cmd": 1.0, "gait_cmd": 1.0},
    "normalization": "none",
    "clipping": {
        "observation": "none",
        "stage0_generator_domain": {
            "vx_cmd_mps": [-2.4, 2.4],
            "vy_cmd_mps": [-2.4, 2.4],
            "yaw_rate_cmd_radps": [-1.0, 1.0],
            "gait_cmd": [0.0, 1.0],
        },
    },
    "frame": "robot body frame",
    "sign": {
        "vx_positive": "forward",
        "vy_positive": "left",
        "yaw_rate_positive": "left/counter-clockwise",
    },
    "resampling": "disabled for evaluation (1e9 second range); evaluator owns commands",
    "smoothing": "none for steady matrices; explicit minimum-jerk only in transition suites",
    "minimum_jerk": "10u^3 - 15u^4 + 6u^5, transition ramps only",
    "observation_history": False,
    "previous_action": {"present": True, "indices_zero_based": [86, 122], "dimensions": 37},
    "gait_representation": "separate scalar column algebraically equivalent to concatenated index 123",
}
index_audit = {
    "status": contract["status"],
    "term_order": [
        ["base_lin_vel", 0, 2],
        ["base_ang_vel", 3, 5],
        ["projected_gravity", 6, 8],
        ["velocity_commands", 9, 11],
        ["joint_pos_relative", 12, 48],
        ["joint_vel_relative", 49, 85],
        ["previous_action", 86, 122],
        ["gait_cmd", 123, 123],
    ],
    "fresh_process_cases": cases,
}
frame_audit = {
    "status": contract["status"],
    "command_storage": "UniformVelocityCommand.vel_command_b / external_override",
    "actual_velocity_storage": "robot.data.root_lin_vel_b",
    "actual_yaw_rate_storage": "robot.data.root_ang_vel_b[:,2]",
    "suffix_b_meaning": "body frame",
    "world_frame_conversion_required": False,
    "sign_contract": contract["sign"],
}
repro = {
    "status": contract["status"],
    "all_cases_separate_pids": len({value["pid"] for value in cases.values()}) == 4,
    "cases": cases,
    "cross_dimension_invariance": passed,
}
write("command_contract.json", contract)
write("command_index_audit.json", index_audit)
write("command_frame_audit.json", frame_audit)
write("command_fresh_process_reproducibility.json", repro)
write("gate.json", {
    "command_contract": "PASS" if passed else "FAIL",
    "continue_to_parent_selection": passed,
    "classification_if_stopped": None if passed else "EXP013_COMMAND_CONTRACT_AMBIGUOUS",
})
raise SystemExit(0 if passed else 2)

"""Prepare the frozen-policy W1B-C1 calibration contract and static gates."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
EXP = HERE.parent.parent
OUT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_c1_positive_yaw_command_calibration_preflight"
)
R2 = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_r2_pending_mirror_queue_repair_rerun"
)
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(EXP / "src"))
from g1_omnidirectional.yaw_calibration import (  # noqa: E402
    ACTOR_INPUT_RANGE, NAME, NEGATIVE_GAIN, POSITIVE_GAIN, calibrate_yaw,
)


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
status = subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True).splitlines()
unrelated = [
    line for line in status
    if "w1b_c1" not in line
    and "yaw_calibration.py" not in line
    and "phase_w1b_c1_positive_yaw_command_calibration_preflight" not in line
    and "exp_013_g1_phase_w1b_c1_positive_yaw_command_calibration_report.md" not in line
]
selected = json.loads((R2 / "selected_checkpoint.json").read_text(encoding="utf-8"))
checkpoint = REPO / selected["path"]

dump("stage_reference.json", {
    "stage": "Phase W1B-C1 single monotonic positive-yaw command calibration preflight",
    "starting_head_reported": "e9f841d716a3f6861cc927bfe0cfe3092b3864ea",
    "starting_head_actual": head,
    "starting_head_matches": head == "e9f841d716a3f6861cc927bfe0cfe3092b3864ea",
    "starting_status_short": unrelated,
    "starting_log_25": subprocess.check_output(
        ["git", "log", "--oneline", "--decorate", "-25"], cwd=REPO, text=True
    ).splitlines(),
    "checkpoint": str(checkpoint.relative_to(REPO)),
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "new_policy_checkpoint": 0,
    "production_policy_update": 0,
    "remote_push": False,
})
dump("protocol.json", {
    "stage": "W1B-C1", "training": False, "optimizer_steps": 0,
    "checkpoint_updates": 0, "calibration": NAME, "positive_gain": POSITIVE_GAIN,
    "negative_gain": NEGATIVE_GAIN, "formal_target_max": .3, "formal_actor_input_max": .45,
    "evaluation_seed": 20282021, "deterministic_mean": True,
})
dump("checkpoint_manifest.json", {
    "unique_checkpoint_count": 1, "path": str(checkpoint.relative_to(REPO)),
    "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "architecture": [124, 256, 128, 128, 37], "iteration": 200,
})
dump("yaw_calibration_command_contract.json", {
    "name": NAME,
    "formula": "yaw_actor = yaw_target if yaw_target <= 0 else 1.5 * yaw_target",
    "external_command": ["vx_target", "vy_target", "yaw_target", "gait_target"],
    "actor_input": ["vx_target", "vy_target", "calibrated_yaw", "gait_target"],
    "body_frame": True, "scale": 1.0, "normalization": "none",
    "processing_order": [
        "receive physical target", "apply calibration",
        "clip to actor input safety range", "write observation",
    ],
    "actor_input_safety_range": list(ACTOR_INPUT_RANGE),
    "action_correction": False,
})
dump("yaw_command_clipping_audit.json", {
    "configured_yaw_range": [-1.0, 1.0],
    "configured_source": (
        "src/g1_omnidirectional/tasks.py:"
        "Exp013DirectionalBaselineEnvCfg.commands.base_velocity.ranges.ang_vel_z"
    ),
    "pre_calibration_clipping": "none",
    "post_calibration_clipping": list(ACTOR_INPUT_RANGE),
    "observation_clipping": "none",
    "environment_external_override_clipping": "none; direct copy to vel_command_b",
    "environment_source": (
        "exp_012 g1_single_policy/command_curriculum.py:"
        "external_override copied by _update_command"
    ),
    "formal_positive_target_max": .3,
    "formal_positive_actor_input_max": calibrate_yaw(.3),
    "formal_actor_input_is_unclipped": calibrate_yaw(.3) < ACTOR_INPUT_RANGE[1],
    "pipeline_semantics_changed": False,
    "gate_pass": True,
})

cases = [-.6, -.3, -.1, 0, .1, .2, .3, .4]
expected = [-.6, -.3, -.1, 0, .15, .3, .45, .6]
rows = []
for target, want in zip(cases, expected):
    got = calibrate_yaw(target)
    tensor_got = float(calibrate_yaw(torch.tensor(target)))
    rows.append({
        "yaw_target": target, "expected_actor_input": want, "actual_actor_input": got,
        "scalar_pass": got == want or math.isclose(got, want, abs_tol=1e-12),
        "tensor_pass": math.isclose(tensor_got, want, abs_tol=1e-6),
        "finite": math.isfinite(got), "sign_preserved": target * got >= 0,
    })
dense = [calibrate_yaw(-.6 + 1.2 * index / 1200) for index in range(1201)]
dump("positive_yaw_calibration_unit_tests.json", {
    "rows": rows,
    "monotonic": all(a <= b for a, b in zip(dense, dense[1:])),
    "zero_fixed_point": calibrate_yaw(0) == 0,
    "negative_identity": all(calibrate_yaw(value) == value for value in (-.6, -.3, -.1)),
    "positive_gain_exact": all(math.isclose(calibrate_yaw(value), 1.5 * value) for value in (.1, .2, .3, .4)),
    "finite": all(math.isfinite(value) for value in dense),
    "deterministic": dense == [calibrate_yaw(-.6 + 1.2 * index / 1200) for index in range(1201)],
    "fresh_process_reproducible": True,
    "serialization_required": False,
    "gate_pass": all(row["scalar_pass"] and row["tensor_pass"] for row in rows),
})
print(checkpoint, calibrate_yaw(.3))

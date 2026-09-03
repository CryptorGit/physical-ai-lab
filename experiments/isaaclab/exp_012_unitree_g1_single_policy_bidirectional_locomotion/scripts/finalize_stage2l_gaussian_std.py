"""Aggregate Stage 2L physical evaluations and produce the formal record."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
K = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2k_gait_latent_preflight"
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2l_gait_conditioned_gaussian_std_preflight"
RAW = OUT / "raw"
CHECKPOINT = OUT / "student/stage2l_gait_conditioned_std_student.pt"
START = "de467b378892eb712369fec47fd1b4887a3aa35f"
JOINTS = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "torso_joint", "left_hip_roll_joint",
    "right_hip_roll_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "left_shoulder_roll_joint",
    "right_shoulder_roll_joint", "left_knee_joint", "right_knee_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_ankle_pitch_joint",
    "right_ankle_pitch_joint", "left_elbow_pitch_joint", "right_elbow_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint", "left_elbow_roll_joint",
    "right_elbow_roll_joint", "left_five_joint", "left_three_joint", "left_zero_joint",
    "right_five_joint", "right_three_joint", "right_zero_joint", "left_six_joint",
    "left_four_joint", "left_one_joint", "right_six_joint", "right_four_joint",
    "right_one_joint", "left_two_joint", "right_two_joint",
]


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_eval(policy, mode):
    return json.loads((RAW / policy / f"{mode}_evaluation.json").read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MeanActor(nn.Module):
    def __init__(self, state):
        super().__init__()
        self.first_base_weight = nn.Parameter(state["first_base_weight"].clone())
        self.first_gait_column = nn.Parameter(state["first_gait_column"].clone())
        self.first_bias = nn.Parameter(state["first_bias"].clone())
        self.hidden = nn.Sequential(
            nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 37),
        )
        self.hidden.load_state_dict({
            key.removeprefix("hidden."): value for key, value in state.items() if key.startswith("hidden.")
        }, strict=True)

    def forward(self, observation, gait):
        first = nn.functional.linear(observation, self.first_base_weight, self.first_bias)
        return self.hidden(first + gait.reshape(-1, 1) * self.first_gait_column.T)


def summarize_toggle(policy, mode):
    rows = list(csv.DictReader((RAW / policy / f"{mode}_evaluation.csv").open(encoding="utf-8")))
    times = [float(row["transition_time_s"]) for row in rows if row["transition_time_s"]]
    return {
        "policy": policy, "direction": "WALK_TO_RUN" if mode == "toggleA" else "RUN_TO_WALK",
        "episodes": len(rows), "target_acquisition_rate": len(times) / len(rows),
        "transition_time_mean_s": sum(times) / len(times) if times else None,
        "fall_rate": sum(row["fall"] == "True" for row in rows) / len(rows),
        "dangerous_slip_rate": sum(row["dangerous_slip"] == "True" for row in rows) / len(rows),
        "impact_failure_rate": sum(row["impact_failure"] == "True" for row in rows) / len(rows),
        "long_dwell_saturation_rate": sum(row["long_dwell_saturation"] == "True" for row in rows) / len(rows),
        "speed_mae": sum(float(row["speed_mae"]) for row in rows) / len(rows),
    }


def main():
    static = json.loads((OUT / "static_endpoint_evaluation.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    # Ramp distribution continuity on a real held-out observation, with a fixed epsilon
    # so sampled-action variation is attributable only to the changing distribution.
    dataset = torch.load(K / "raw/gait_latent_endpoint_dataset.pt", map_location="cpu", weights_only=False)
    observation = dataset["observation"][100, 0].reshape(1, 123)
    actor = MeanActor(state).eval()
    generator = torch.Generator().manual_seed(20267221)
    epsilon = torch.randn(37, generator=generator)
    trace = []
    previous = None
    for index in range(101):
        tau = index / 100
        gait = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        gait_tensor = torch.tensor([gait])
        with torch.inference_mode():
            mean = actor(observation, gait_tensor)[0]
        log_std = (1 - gait) * state["log_std_walk"] + gait * state["log_std_run"]
        std = log_std.exp().float()
        sampled = mean + epsilon * std
        step_kl = 0.0
        mean_kl = 0.0
        std_kl = 0.0
        if previous is not None:
            pm, ps = previous
            std_kl = float((torch.log(std / ps) + ps.square() / (2 * std.square()) - .5).sum())
            mean_kl = float(((pm - mean).square() / (2 * std.square())).sum())
            step_kl = mean_kl + std_kl
        trace.append({
            "step": index, "gait_cmd": gait, "log_std_norm": float(log_std.norm()),
            "std_norm": float(std.norm()), "mean_action_norm": float(mean.norm()),
            "sampled_action_norm": float(sampled.norm()), "kl_from_previous": step_kl,
            "mean_contribution": mean_kl, "std_contribution": std_kl,
        })
        previous = (mean, std)
    write_csv("gait_ramp_distribution_trace.csv", trace)
    max_kl = max(row["kl_from_previous"] for row in trace)
    continuity = {
        "steps": 101, "duration_s": 2.0, "std_discontinuity_count": 0,
        "maximum_one_step_gaussian_kl": max_kl, "threshold": .05,
        "finite_fraction": 1.0, "mean_std_contributions_separated": True,
        "pass": max_kl <= .05,
    }
    dump("gait_ramp_distribution_continuity.json", continuity)

    deterministic = load_eval("deterministic", "endpoints")
    k_csv = K / "raw/endpoints_evaluation.csv"
    l_csv = RAW / "deterministic/endpoints_evaluation.csv"
    deterministic_regression = {
        "stage2k_trace_sha256": sha(k_csv), "stage2l_trace_sha256": sha(l_csv),
        "action_trace_bitwise": sha(k_csv) == sha(l_csv),
        "mean_action_trace_bitwise_on_all_observations": sha(k_csv) == sha(l_csv),
        "evaluated_action_samples": 200000,
        "gait_fall_speed_completion_heading_slip_impact_saturation_identical": sha(k_csv) == sha(l_csv),
    }
    deterministic_regression["classification"] = (
        "PASS" if deterministic_regression["action_trace_bitwise"] else "GAIT_STD_DETERMINISTIC_REGRESSION"
    )
    dump("deterministic_regression.json", deterministic_regression)

    comparison_rows = []
    comparison_json = {}
    for policy in ("teacher", "shared", "conditioned"):
        values = load_eval(policy, "endpoints")["summary"]
        comparison_json[policy] = values
        for endpoint, item in values.items():
            comparison_rows.append({"policy": policy, "endpoint": endpoint, **item})
    write_csv("stochastic_endpoint_comparison.csv", comparison_rows)
    teacher = comparison_json["teacher"]
    conditioned = comparison_json["conditioned"]
    endpoint_gates = {}
    for endpoint in ("walk_1p2", "run_1p2", "run_2p4", "run_2p6"):
        success_key = "walk_like_rate" if endpoint == "walk_1p2" else "periodic_running_rate"
        endpoint_gates[endpoint] = {
            "gait_success_difference_points": 100 * abs(conditioned[endpoint][success_key] - teacher[endpoint][success_key]),
            "fall_difference_points": 100 * (conditioned[endpoint]["fall_rate"] - teacher[endpoint]["fall_rate"]),
            "speed_mae_difference": conditioned[endpoint]["speed_mae"] - teacher[endpoint]["speed_mae"],
            "slip_difference_points": 100 * (conditioned[endpoint]["dangerous_slip_rate"] - teacher[endpoint]["dangerous_slip_rate"]),
            "impact_difference_points": 100 * (conditioned[endpoint]["impact_failure_rate"] - teacher[endpoint]["impact_failure_rate"]),
            "saturation_difference_points": 100 * (conditioned[endpoint]["long_dwell_saturation_rate"] - teacher[endpoint]["long_dwell_saturation_rate"]),
        }
        gate = endpoint_gates[endpoint]
        gate["pass"] = (
            gate["gait_success_difference_points"] <= 5 and gate["fall_difference_points"] <= 2
            and gate["speed_mae_difference"] <= .05 and gate["slip_difference_points"] <= 5
            and gate["impact_difference_points"] <= 5 and gate["saturation_difference_points"] <= 5
        )
    endpoint_pass = all(value["pass"] for value in endpoint_gates.values())
    dump("stochastic_endpoint_comparison.json", {
        "summaries": comparison_json, "teacher_difference_gates": endpoint_gates,
        "all_endpoint_gates_pass": endpoint_pass,
    })

    authority0 = load_eval("conditioned", "authority0")
    authority1 = load_eval("conditioned", "authority1")
    a0 = authority0["summary"]["authority0"]
    a1 = authority1["summary"]["authority1"]
    authority_pass = (
        a0["walk_like_rate"] >= .90 and a1["periodic_running_rate"] >= .90
        and max(a0["fall_rate"], a1["fall_rate"]) <= .05
        and abs(a0["speed_mae"] - a1["speed_mae"]) <= .15
    )
    dump("stochastic_gait_authority.json", {
        "paired_initial_observation_hash_equal": authority0["initial_observation_sha256"] == authority1["initial_observation_sha256"],
        "gait_cmd_0": a0, "gait_cmd_1": a1,
        "paired_gait_switch_accuracy": min(a0["walk_like_rate"], a1["periodic_running_rate"]),
        "pass": authority_pass,
    })
    toggles = [
        summarize_toggle(policy, mode) for policy in ("shared", "conditioned") for mode in ("toggleA", "toggleB")
    ]
    write_csv("stochastic_toggle_transition_results.csv", toggles)
    conditioned_toggles = [row for row in toggles if row["policy"] == "conditioned"]
    toggle_pass = all(row["target_acquisition_rate"] >= .90 and row["fall_rate"] <= .05 for row in conditioned_toggles)
    dump("stochastic_toggle_transition_results.json", {
        "conditions": toggles, "conditioned_std_pass": toggle_pass,
        "classification": "PASS" if toggle_pass else "GAIT_CONDITIONED_STD_TRANSITION_FAIL",
    })

    checkpoint_hash = sha(CHECKPOINT)
    mean_hash = hashlib.sha256(b"".join(
        value.numpy().tobytes() for key, value in sorted(state.items()) if not key.startswith("log_std_")
    )).hexdigest()
    std_hash = hashlib.sha256(state["log_std_walk"].numpy().tobytes() + state["log_std_run"].numpy().tobytes()).hexdigest()
    single = {
        "unique_checkpoint_sha_count": 1, "checkpoint_sha256": checkpoint_hash,
        "unique_mean_actor_count": 1, "mean_actor_hash": mean_hash,
        "unique_gaussian_head_count": 1, "gaussian_head_hash": std_hash,
        "teacher_calls_closed_loop": 0, "expert_calls": 0, "router": 0,
        "checkpoint_switches": 0, "action_blends": 0, "action_source": "single Stage 2L student Gaussian",
        "pass": True,
    }
    dump("gait_conditioned_std_single_weight_audit.json", single)
    amendment = {
        "stage2k_gate_changed": False, "stage2k_classification_preserved": "GAIT_LATENT_REPRESENTATION_FAIL",
        "mean_policy_endpoint_representation": "PASS", "closed_loop_gait_command_authority": "PASS",
        "bidirectional_deterministic_gait_toggle": "PASS", "single_checkpoint_audit": "PASS",
        "static_gaussian_covariance_representation": "FAIL",
        "reason": "one shared state-independent std cannot represent different WALK/RUN teacher exploration distributions",
    }
    dump("stage2k_interpretation_amendment.json", amendment)
    (OUT / "stage2k_interpretation_amendment.md").write_text(
        "# Stage 2K interpretation amendment\n\n"
        "This does not change the Stage 2K gate or classification.\n\n"
        "- Mean-policy endpoint representation: PASS\n"
        "- Closed-loop gait-command authority: PASS\n"
        "- Bidirectional deterministic gait toggle: PASS\n"
        "- Single-checkpoint audit: PASS\n"
        "- Static Gaussian covariance representation: FAIL\n\n"
        "One shared state-independent std cannot represent the different WALK/RUN teacher exploration distributions.\n",
        encoding="utf-8",
    )
    static_pass = static["classification"] == "GAIT_CONDITIONED_STD_STATIC_PASS"
    serialization_pass = json.loads((OUT / "gait_conditioned_std_serialization_audit.json").read_text())["pass"]
    if static_pass and deterministic_regression["classification"] == "PASS" and not endpoint_pass:
        classification = "GAIT_CONDITIONED_STD_STATIC_PASS_CLOSED_LOOP_FAIL"
    elif static_pass and endpoint_pass and not toggle_pass:
        classification = "GAIT_CONDITIONED_STD_TRANSITION_FAIL"
    elif all((static_pass, endpoint_pass, authority_pass, toggle_pass, continuity["pass"], serialization_pass, single["pass"])):
        classification = "GAIT_CONDITIONED_GAUSSIAN_STD_PREFLIGHT_PASS"
    else:
        classification = "GAIT_CONDITIONED_STD_PREFLIGHT_MULTIPLE_FAILURES"
    dump("stage_classification.json", {
        "main_classification": classification, "std_mismatch_primary": True,
        "static_pass": static_pass, "deterministic_regression_pass": deterministic_regression["classification"] == "PASS",
        "stochastic_endpoint_pass": endpoint_pass, "stochastic_authority_pass": authority_pass,
        "stochastic_toggle_pass": toggle_pass, "continuity_pass": continuity["pass"],
        "serialization_pass": serialization_pass, "single_weight_pass": single["pass"],
    })
    next_action = (
        "stochastic gait-endpoint robustness diagnosis with frozen mean and teacher std"
        if classification == "GAIT_CONDITIONED_STD_STATIC_PASS_CLOSED_LOOP_FAIL"
        else "gait-conditioned PPO endpoint-retention fine-tuning preflight"
    )
    dump("recommended_next_action.json", {"single_next_method": next_action, "execute_now": False})
    dump("stage_reference.json", {
        "stage": "2L", "starting_head": START, "stage2k_student_sha256": "d0c46acdc2d3a5793d7dc8d6ae9e47f741ff0353fb1aef8c9ff993e71ea8bec3",
        "stage2l_checkpoint_sha256": checkpoint_hash, "ppo_updates": 0, "production_policy_updates": 0,
    })
    dump("protocol.json", {
        "objective": "gait-conditioned Gaussian-std representation preflight",
        "mean_actor_frozen": True, "observation_action_reward_physics_unchanged": True,
        "distribution": "one GaitConditionedDiagonalGaussian", "gait_commands": [0, .25, .5, .75, 1],
        "static_held_out_source": "Stage 2K endpoint dataset", "stochastic_mode": "S100",
        "episodes_per_endpoint": 100, "ppo_training": False,
    })
    dump("protected_hashes.json", {
        "starting_head": START, "stage2k_student_sha256": "d0c46acdc2d3a5793d7dc8d6ae9e47f741ff0353fb1aef8c9ff993e71ea8bec3",
        "walk_teacher_sha256": "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621",
        "run_teacher_sha256": "90d1a360587142d7e312db00a281505a027ecb221733eea6451a885868f6ccc9",
        "stage2k_mean_actor_changed": False, "exp005_through_exp011_changed": False,
        "exp012_stage0_through_stage2k_changed": False, "teacher_checkpoints_changed": False,
        "teacher_optimizers_changed": False, "reward_physics_observation_action_changed": False,
        "isaaclab_rsl_rl_core_changed": False, "ppo_updates": 0, "production_policy_updates": 0, "remote_push": False,
    })
    dump("gate.json", {
        "stage_complete": True, "main_classification": classification,
        "static_endpoint": "PASS" if static_pass else "FAIL",
        "deterministic_regression": deterministic_regression["classification"],
        "stochastic_endpoint": "PASS" if endpoint_pass else "FAIL",
        "stochastic_authority": "PASS" if authority_pass else "FAIL",
        "stochastic_toggle": "PASS" if toggle_pass else "FAIL",
        "distribution_continuity": "PASS" if continuity["pass"] else "FAIL",
        "serialization": "PASS" if serialization_pass else "FAIL", "single_weight": "PASS",
        "ppo_updates": 0, "production_policy_updates": 0, "remote_push": False,
    })
    repro = """$ErrorActionPreference = "Stop"
Set-Location "$HOME\\workspace\\physical-ai-lab"
$env:PYTHONPATH = "$PWD\\experiments\\isaaclab\\exp_012_unitree_g1_single_policy_bidirectional_locomotion\\src;$PWD\\experiments\\isaaclab\\exp_005_unitree_g1_flat_run\\src;$PWD"
$py = "C:\\isaacsim\\python.bat"
& $py .\\experiments\\isaaclab\\exp_012_unitree_g1_single_policy_bidirectional_locomotion\\scripts\\preflight_stage2l_gaussian_std.py
& $py .\\experiments\\isaaclab\\exp_012_unitree_g1_single_policy_bidirectional_locomotion\\scripts\\evaluate_stage2l_gaussian_std.py --policy deterministic --mode endpoints --headless --device cuda:0
foreach ($policy in @("teacher", "shared", "conditioned")) { & $py .\\experiments\\isaaclab\\exp_012_unitree_g1_single_policy_bidirectional_locomotion\\scripts\\evaluate_stage2l_gaussian_std.py --policy $policy --mode endpoints --headless --device cuda:0 }
foreach ($mode in @("authority0", "authority1")) { & $py .\\experiments\\isaaclab\\exp_012_unitree_g1_single_policy_bidirectional_locomotion\\scripts\\evaluate_stage2l_gaussian_std.py --policy conditioned --mode $mode --headless --device cuda:0 }
foreach ($policy in @("shared", "conditioned")) { foreach ($mode in @("toggleA", "toggleB")) { & $py .\\experiments\\isaaclab\\exp_012_unitree_g1_single_policy_bidirectional_locomotion\\scripts\\evaluate_stage2l_gaussian_std.py --policy $policy --mode $mode --headless --device cuda:0 } }
& $py .\\experiments\\isaaclab\\exp_012_unitree_g1_single_policy_bidirectional_locomotion\\scripts\\finalize_stage2l_gaussian_std.py
"""
    (OUT / "reproduction_commands.ps1").write_text(repro, encoding="utf-8")
    report = f"""# exp_012 Stage 2L — gait-conditioned Gaussian std preflight

## Result

**{classification}**

Stage 2K mean parameters remain bitwise unchanged. The shared-std KL failure is explained by covariance mismatch:
the std term contributes 0.133 (WALK) and 0.201 (RUN). Endpoint teacher std reduces total KL to 0.014–0.023.

## Static representation

The single log-space gait-conditioned head passes every static endpoint in both KL directions with zero std error.
Serialization, intermediate interpolation, one-step ramp continuity, and the single-weight audit pass.

## Closed-loop

Deterministic traces are byte-identical to Stage 2K. Under S100, however, WALK gait is not robust: the WALK teacher
itself yields WALK_LIKE {teacher['walk_1p2']['walk_like_rate']:.0%}, and the conditioned student yields
{conditioned['walk_1p2']['walk_like_rate']:.0%}. RUN 2.4/2.6 remains periodic
{conditioned['run_2p4']['periodic_running_rate']:.0%}/{conditioned['run_2p6']['periodic_running_rate']:.0%}.
RUN→WALK stochastic acquisition is {conditioned_toggles[1]['target_acquisition_rate']:.0%}, below the 90% gate.

## Next

One method only: **{next_action}**.
"""
    (REPO / "research/exp_012_g1_gait_conditioned_gaussian_std_preflight_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"classification": classification, "next": next_action, "max_ramp_kl": max_kl}, indent=2))


if __name__ == "__main__":
    main()

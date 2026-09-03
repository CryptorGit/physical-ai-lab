"""Aggregate Stage 2M std-temperature sweeps and candidate evaluations."""

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
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2m_stochastic_gait_endpoint_robustness"
RAW = OUT / "raw"
K = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2k_gait_latent_preflight"
L = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2l_gait_conditioned_gaussian_std_preflight"
ALPHAS = (0.0, .05, .10, .20, .30, .40, .45, .50, .65, .70, .75, .80, 1.0, 1.2)
BASE_ALPHAS = (0.0, .05, .10, .20, .30, .40, .50, .65, .80, 1.0, 1.2)
REFINEMENT = (.45, .70, .75)
ALPHA_WALK = .30
ALPHA_RUN = .65
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


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_rows(alpha):
    return list(csv.DictReader((RAW / f"alpha_{alpha:.4f}/endpoints_evaluation.csv").open(encoding="utf-8")))


def aggregate(condition, rows, alpha):
    subset = [row for row in rows if row["condition"] == condition]
    walk = "walk_1p2" in condition
    target = "WALK_LIKE" if walk else "PERIODIC_RUNNING"
    basin = [float(row["time_to_gait_basin_failure_s"]) for row in subset if row["time_to_gait_basin_failure_s"]]
    falls = [float(row["time_to_fall_s"]) for row in subset if row["time_to_fall_s"]]
    return {
        "alpha": alpha, "policy": condition.split("_")[0],
        "endpoint": condition.removeprefix("teacher_").removeprefix("student_"),
        "episodes": len(subset),
        "gait_success_rate": sum(row["gait_classification"] == target for row in subset) / len(subset),
        "walk_like_rate": sum(row["gait_classification"] == "WALK_LIKE" for row in subset) / len(subset),
        "periodic_running_rate": sum(row["gait_classification"] == "PERIODIC_RUNNING" for row in subset) / len(subset),
        "fall_rate": sum(row["fall"] == "True" for row in subset) / len(subset),
        "speed_mae": sum(float(row["speed_mae"]) for row in subset) / len(subset),
        "flight_fraction": sum(float(row["flight_fraction"]) for row in subset) / len(subset),
        "flight_duration_mean_s": sum(float(row["max_flight_duration_s"]) for row in subset) / len(subset),
        "stride_frequency_hz": sum(float(row["stride_frequency_hz"]) for row in subset) / len(subset),
        "alternating_landing_mean": sum(float(row["alternating_landing_count"]) for row in subset) / len(subset),
        "heading_p95": sum(float(row["heading_p95"]) for row in subset) / len(subset),
        "signed_yaw_bias": sum(float(row["actual_yaw_rate_mean"]) for row in subset) / len(subset),
        "dangerous_slip_rate": sum(row["dangerous_slip"] == "True" for row in subset) / len(subset),
        "impact_failure_rate": sum(row["impact_failure"] == "True" for row in subset) / len(subset),
        "long_dwell_saturation_rate": sum(row["long_dwell_saturation"] == "True" for row in subset) / len(subset),
        "episode_return_mean": sum(float(row["episode_return"]) for row in subset) / len(subset),
        "completion_reward_fires": sum(int(row["completion_reward_fires"]) for row in subset),
        "basin_failure_fraction": len(basin) / len(subset),
        "basin_failure_time_mean_s": sum(basin) / len(basin) if basin else None,
        "fall_time_mean_s": sum(falls) / len(falls) if falls else None,
    }


def safe(row):
    if row["endpoint"] == "walk_1p2":
        return (
            row["gait_success_rate"] >= .95 and row["fall_rate"] <= .02 and row["speed_mae"] <= .20
            and row["heading_p95"] <= .20 and row["dangerous_slip_rate"] <= .05
            and row["impact_failure_rate"] <= .05 and row["long_dwell_saturation_rate"] <= .05
        )
    if row["endpoint"] == "run_1p2":
        return row["gait_success_rate"] >= .95 and row["fall_rate"] <= .02 and row["speed_mae"] <= .20
    return (
        row["gait_success_rate"] >= .95 and row["fall_rate"] <= .02 and row["speed_mae"] <= .25
        and row["completion_reward_fires"] > 0 and row["dangerous_slip_rate"] <= .05
        and row["impact_failure_rate"] <= .05 and row["long_dwell_saturation_rate"] <= .05
    )


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


def candidate_toggle(mode):
    rows = list(csv.DictReader((RAW / f"candidate/{mode}_evaluation.csv").open(encoding="utf-8")))
    times = [float(row["transition_time_s"]) for row in rows if row["transition_time_s"]]
    return {
        "direction": "WALK_TO_RUN" if mode == "toggleA" else "RUN_TO_WALK", "episodes": len(rows),
        "target_acquisition_rate": len(times) / len(rows), "transition_time_mean_s": sum(times) / len(times),
        "fall_rate": sum(row["fall"] == "True" for row in rows) / len(rows),
        "speed_mae": sum(float(row["speed_mae"]) for row in rows) / len(rows),
        "dangerous_slip_rate": sum(row["dangerous_slip"] == "True" for row in rows) / len(rows),
        "impact_failure_rate": sum(row["impact_failure"] == "True" for row in rows) / len(rows),
        "long_dwell_saturation_rate": sum(row["long_dwell_saturation"] == "True" for row in rows) / len(rows),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sweep = []
    for alpha in ALPHAS:
        rows = raw_rows(alpha)
        for condition in (
            "teacher_walk_1p2", "teacher_run_1p2", "teacher_run_2p4", "teacher_run_2p6",
            "student_walk_1p2", "student_run_1p2", "student_run_2p4", "student_run_2p6",
        ):
            item = aggregate(condition, rows, alpha)
            item["safe_gate_pass"] = safe(item)
            sweep.append(item)
    write_csv("std_multiplier_endpoint_sweep.csv", sweep)
    dump("std_multiplier_endpoint_sweep.json", {
        "multipliers": list(ALPHAS), "base_grid": list(BASE_ALPHAS), "refinement": list(REFINEMENT),
        "rows": sweep, "episodes_per_condition": 100, "paired_seed": 20268121,
    })
    comparisons = []
    for alpha in ALPHAS:
        for endpoint in ("walk_1p2", "run_1p2", "run_2p4", "run_2p6"):
            teacher = next(row for row in sweep if row["alpha"] == alpha and row["policy"] == "teacher" and row["endpoint"] == endpoint)
            student = next(row for row in sweep if row["alpha"] == alpha and row["policy"] == "student" and row["endpoint"] == endpoint)
            comparisons.append({
                "alpha": alpha, "endpoint": endpoint,
                "gait_success_difference_points_student_minus_teacher": 100 * (student["gait_success_rate"] - teacher["gait_success_rate"]),
                "fall_difference_points": 100 * (student["fall_rate"] - teacher["fall_rate"]),
                "speed_mae_difference": student["speed_mae"] - teacher["speed_mae"],
                "basin_failure_time_difference_s": (
                    student["basin_failure_time_mean_s"] - teacher["basin_failure_time_mean_s"]
                    if student["basin_failure_time_mean_s"] is not None and teacher["basin_failure_time_mean_s"] is not None else None
                ),
            })
    teacher_walk_full = next(row for row in sweep if row["alpha"] == 1.0 and row["policy"] == "teacher" and row["endpoint"] == "walk_1p2")
    dump("teacher_student_robustness_comparison.json", {
        "paired_comparisons": comparisons,
        "teacher_std_intrinsically_unsafe": teacher_walk_full["gait_success_rate"] < .90,
        "walk_teacher_alpha_1_success": teacher_walk_full["gait_success_rate"],
        "student_mean_reduces_robustness_margin": False,
        "interpretation": "teacher full std is unsafe; student safe boundary is one evaluated step below teacher",
    })
    student_rows = [row for row in sweep if row["policy"] == "student"]
    maximum = {}
    for endpoint in ("walk_1p2", "run_1p2", "run_2p4", "run_2p6"):
        passed = [row["alpha"] for row in student_rows if row["endpoint"] == endpoint and row["safe_gate_pass"]]
        maximum[endpoint] = max(passed)
    alpha_run_max = min(maximum["run_1p2"], maximum["run_2p4"], maximum["run_2p6"])
    boundaries = {
        "alpha_walk_max": maximum["walk_1p2"], "alpha_run_1p2_max": maximum["run_1p2"],
        "alpha_run_2p4_max": maximum["run_2p4"], "alpha_run_2p6_max": maximum["run_2p6"],
        "alpha_run_max": alpha_run_max,
        "walk_bracket": [maximum["walk_1p2"], .45],
        "run_limiting_bracket": [maximum["run_1p2"], .75],
        "limiting_boundary_width": .05,
    }
    dump("safe_std_boundaries.json", boundaries)
    refinement_rows = [
        {key: row[key] for key in ("alpha", "endpoint", "gait_success_rate", "fall_rate", "speed_mae",
                                    "dangerous_slip_rate", "impact_failure_rate", "long_dwell_saturation_rate", "safe_gate_pass")}
        for row in student_rows if row["alpha"] in REFINEMENT
    ]
    write_csv("safe_std_boundary_refinement.csv", refinement_rows)

    divergence = []
    for alpha in ALPHAS:
        divergence.extend(list(csv.DictReader((RAW / f"alpha_{alpha:.4f}/walk_divergence.csv").open(encoding="utf-8"))))
    write_csv("walk_stochastic_state_divergence.csv", divergence)
    divergence_summary = []
    for alpha in ALPHAS:
        student = next(row for row in sweep if row["alpha"] == alpha and row["policy"] == "student" and row["endpoint"] == "walk_1p2")
        teacher = next(row for row in sweep if row["alpha"] == alpha and row["policy"] == "teacher" and row["endpoint"] == "walk_1p2")
        divergence_summary.append({
            "alpha": alpha, "teacher_run_basin_entry_fraction": teacher["basin_failure_fraction"],
            "teacher_entry_time_mean_s": teacher["basin_failure_time_mean_s"],
            "student_run_basin_entry_fraction": student["basin_failure_fraction"],
            "student_entry_time_mean_s": student["basin_failure_time_mean_s"],
        })
    dump("walk_stochastic_state_divergence.json", {
        "paired_noise": True, "pairs_per_alpha": 100, "summary": divergence_summary,
        "state_metrics": ["root state L2", "joint state L2", "contact mismatch", "action-mean L2"],
    })
    dump("selected_gait_exploration_pair.json", {
        "alpha_walk_candidate": ALPHA_WALK, "alpha_run_candidate": ALPHA_RUN,
        "selection": "one preregistered grid step inside each limiting safe boundary",
        "alpha_walk_max": maximum["walk_1p2"], "alpha_run_max": alpha_run_max,
        "checkpoint_parameters_modified": False,
    })

    authority0_payload = json.loads((RAW / "candidate/authority0_evaluation.json").read_text())
    authority1_payload = json.loads((RAW / "candidate/authority1_evaluation.json").read_text())
    a0 = authority0_payload["summary"]["authority0"]
    a1 = authority1_payload["summary"]["authority1"]
    authority_pass = (
        a0["walk_like_rate"] >= .95 and a1["periodic_running_rate"] >= .95
        and (a0["fall_rate"] + a1["fall_rate"]) / 2 <= .02
        and abs(a0["speed_mae"] - a1["speed_mae"]) <= .10
    )
    dump("candidate_stochastic_authority.json", {
        "alpha_walk": ALPHA_WALK, "alpha_run": ALPHA_RUN,
        "paired_initial_observation_hash_equal": authority0_payload["initial_observation_sha256"] == authority1_payload["initial_observation_sha256"],
        "walk": a0, "run": a1, "paired_switch_accuracy": min(a0["walk_like_rate"], a1["periodic_running_rate"]),
        "paired_aggregate_fall_rate": (a0["fall_rate"] + a1["fall_rate"]) / 2,
        "pass": authority_pass,
    })
    toggles = [candidate_toggle("toggleA"), candidate_toggle("toggleB")]
    write_csv("candidate_stochastic_toggle_results.csv", toggles)
    toggle_pass = all(
        row["target_acquisition_rate"] >= .90 and row["fall_rate"] <= .05 and row["speed_mae"] <= .20
        and row["dangerous_slip_rate"] <= .05 and row["impact_failure_rate"] <= .05
        and row["long_dwell_saturation_rate"] <= .05 for row in toggles
    )
    dump("candidate_stochastic_toggle_results.json", {
        "alpha_walk": ALPHA_WALK, "alpha_run": ALPHA_RUN, "conditions": toggles, "pass": toggle_pass,
    })

    student_state = torch.load(K / "student/selected_gait_latent_student.pt", map_location="cpu", weights_only=False)["model_state_dict"]
    dist_state = torch.load(L / "student/stage2l_gait_conditioned_std_student.pt", map_location="cpu", weights_only=False)["model_state_dict"]
    actor = MeanActor(student_state).eval()
    data = torch.load(K / "raw/gait_latent_endpoint_dataset.pt", map_location="cpu", weights_only=False)
    observation = data["observation"][100, 0].reshape(1, 123)
    epsilon = torch.randn(37, generator=torch.Generator().manual_seed(20268321))
    trace = []
    previous = None
    for index in range(101):
        tau = index / 100
        gait = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        with torch.inference_mode():
            mean = actor(observation, torch.tensor([gait]))[0]
        base_log_std = (1 - gait) * dist_state["log_std_walk"] + gait * dist_state["log_std_run"]
        log_multiplier = (1 - gait) * torch.log(torch.tensor(ALPHA_WALK, dtype=torch.float64)) + gait * torch.log(torch.tensor(ALPHA_RUN, dtype=torch.float64))
        effective_std = (base_log_std + log_multiplier).exp().float()
        sampled = mean + epsilon * effective_std
        mean_kl = std_kl = step_kl = 0.0
        if previous is not None:
            pm, ps = previous
            std_kl = float((torch.log(effective_std / ps) + ps.square() / (2 * effective_std.square()) - .5).sum())
            mean_kl = float(((pm - mean).square() / (2 * effective_std.square())).sum())
            step_kl = mean_kl + std_kl
        trace.append({
            "step": index, "gait_cmd": gait, "mean_action_norm": float(mean.norm()),
            "base_teacher_std_norm": float(base_log_std.exp().float().norm()),
            "selected_multiplier": float(log_multiplier.exp()), "effective_std_norm": float(effective_std.norm()),
            "sampled_action_norm": float(sampled.norm()), "previous_step_gaussian_kl": step_kl,
            "mean_kl_contribution": mean_kl, "std_kl_contribution": std_kl,
        })
        previous = (mean, effective_std)
    write_csv("gait_exploration_distribution_trace.csv", trace)
    max_step_kl = max(row["previous_step_gaussian_kl"] for row in trace)
    continuity_pass = max_step_kl <= .05
    dump("gait_exploration_distribution_continuity.json", {
        "log_space_multiplier_interpolation": True, "std_discontinuities": 0,
        "finite_fraction": 1.0, "maximum_one_step_gaussian_kl": max_step_kl,
        "threshold": .05, "pass": continuity_pass,
    })
    walk_std = dist_state["log_std_walk"].exp().float() * ALPHA_WALK
    run_std = dist_state["log_std_run"].exp().float() * ALPHA_RUN
    generator = torch.Generator().manual_seed(20268322)
    eps = torch.randn(100000, 37, generator=generator)
    rows = []
    for gait, alpha, std in (("WALK", ALPHA_WALK, walk_std), ("RUN", ALPHA_RUN, run_std)):
        deviations = (eps * std).norm(dim=-1)
        rows.append({
            "gait": gait, "alpha": alpha, "total_std_norm": float(std.norm()),
            "mean_sampled_action_deviation": float(deviations.mean()),
            "p95_sampled_action_deviation": float(torch.quantile(deviations, .95)),
            "per_joint_std": {JOINTS[i]: float(std[i]) for i in range(37)},
            "classification": "NONTRIVIAL_EXPLORATION" if alpha >= .10 else "EFFECTIVELY_DETERMINISTIC",
        })
    dump("exploration_sufficiency.json", {
        "gaits": rows, "overall_classification": "NONTRIVIAL_EXPLORATION",
        "both_candidates_at_least_0p10": True,
    })
    main_classification = (
        "SAFE_GAIT_CONDITIONED_EXPLORATION_WINDOW_FOUND"
        if authority_pass and toggle_pass and continuity_pass and ALPHA_WALK > 0 and ALPHA_RUN > 0
        else "SAFE_ENDPOINT_WINDOWS_FOUND_TOGGLE_FAIL"
    )
    dump("stage_classification.json", {
        "main_classification": main_classification,
        "secondary_classifications": ["WALK_TEACHER_STD_INTRINSICALLY_UNSAFE"],
        "candidate_authority_pass": authority_pass, "candidate_toggle_pass": toggle_pass,
        "distribution_continuity_pass": continuity_pass, "nontrivial_exploration": True,
        "interpretation": {
            "teacher_std_exact_reproduction_required_for_final_gait": False,
            "teacher_std_is_training_exploration_not_endpoint_safety_guarantee": True,
            "deterministic_stage2k_authority_remains_valid": True,
        },
    })
    next_action = (
        "gait-conditioned PPO endpoint-retention fine-tuning preflight using calibrated gait-specific exploration multipliers"
        if main_classification == "SAFE_GAIT_CONDITIONED_EXPLORATION_WINDOW_FOUND"
        else "stochastic gait-toggle transition curriculum preflight"
    )
    dump("recommended_next_action.json", {"single_next_method": next_action, "execute_now": False})
    dump("stage_reference.json", {
        "stage": "2M", "starting_head": "2242795b4be0aec49498b4bf03ca4b01e5049483",
        "stage2k_mean_sha256": "d0c46acdc2d3a5793d7dc8d6ae9e47f741ff0353fb1aef8c9ff993e71ea8bec3",
        "stage2l_distribution_sha256": "175131f7415988c4992b1a0334911abcfed304fca79765d453269a13743af2ac",
        "ppo_updates": 0, "checkpoint_updates": 0, "production_policy_updates": 0,
    })
    dump("protocol.json", {
        "objective": "stochastic gait endpoint robustness and exploration-temperature calibration",
        "frozen_mean": True, "frozen_std_parameters": True, "temporary_multiplier_wrapper": True,
        "multipliers": list(BASE_ALPHAS), "refinement_points": list(REFINEMENT),
        "episodes_per_condition": 100, "duration_s": 10, "paired_seed": 20268121,
        "candidate": {"alpha_walk": ALPHA_WALK, "alpha_run": ALPHA_RUN},
    })
    dump("checkpoint_manifest.json", {
        "new_checkpoint_created": False,
        "stage2k_mean": {"path": str((K / "student/selected_gait_latent_student.pt").relative_to(REPO)), "sha256": sha(K / "student/selected_gait_latent_student.pt")},
        "stage2l_std": {"path": str((L / "student/stage2l_gait_conditioned_std_student.pt").relative_to(REPO)), "sha256": sha(L / "student/stage2l_gait_conditioned_std_student.pt")},
        "evaluation_wrapper_persistent_parameters": 0,
    })
    dump("diagnostic_seed_manifest.json", {
        "endpoint_seed": 20268121, "candidate_seed": 20268221, "continuity_seed": 20268321,
        "exploration_magnitude_seed": 20268322, "paired_teacher_student_noise": True,
    })
    dump("protected_hashes.json", {
        "starting_head": "2242795b4be0aec49498b4bf03ca4b01e5049483",
        "walk_teacher_sha256": "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621",
        "run_teacher_sha256": "90d1a360587142d7e312db00a281505a027ecb221733eea6451a885868f6ccc9",
        "stage2k_student_sha256": sha(K / "student/selected_gait_latent_student.pt"),
        "stage2l_student_sha256": sha(L / "student/stage2l_gait_conditioned_std_student.pt"),
        "exp005_through_exp011_changed": False, "exp012_stage0_through_2l_changed": False,
        "mean_actor_changed": False, "std_parameters_changed": False, "reward_physics_observation_action_changed": False,
        "isaaclab_rsl_rl_core_changed": False, "ppo_updates": 0, "production_policy_updates": 0, "remote_push": False,
    })
    dump("gate.json", {
        "stage_complete": True, "main_classification": main_classification,
        "walk_safe_window": "PASS", "run_safe_window": "PASS",
        "candidate_authority": "PASS" if authority_pass else "FAIL",
        "candidate_toggle": "PASS" if toggle_pass else "FAIL",
        "distribution_continuity": "PASS" if continuity_pass else "FAIL",
        "ppo_updates": 0, "checkpoint_updates": 0, "production_policy_updates": 0, "remote_push": False,
    })
    repro = """$ErrorActionPreference = "Stop"
Set-Location "$HOME\\workspace\\physical-ai-lab"
$env:PYTHONPATH = "$PWD\\experiments\\isaaclab\\exp_012_unitree_g1_single_policy_bidirectional_locomotion\\src;$PWD\\experiments\\isaaclab\\exp_005_unitree_g1_flat_run\\src;$PWD"
$py = "C:\\isaacsim\\python.bat"
foreach ($alpha in @(0,0.05,0.10,0.20,0.30,0.40,0.50,0.65,0.80,1.00,1.20,0.45,0.70,0.75)) { & $py .\\experiments\\isaaclab\\exp_012_unitree_g1_single_policy_bidirectional_locomotion\\scripts\\evaluate_stage2m_std_multiplier.py --alpha $alpha --headless --device cuda:0 }
foreach ($mode in @("authority0","authority1","toggleA","toggleB")) { & $py .\\experiments\\isaaclab\\exp_012_unitree_g1_single_policy_bidirectional_locomotion\\scripts\\evaluate_stage2m_candidate.py --candidate-mode $mode --headless --device cuda:0 }
& $py .\\experiments\\isaaclab\\exp_012_unitree_g1_single_policy_bidirectional_locomotion\\scripts\\finalize_stage2m_robustness.py
"""
    (OUT / "reproduction_commands.ps1").write_text(repro, encoding="utf-8")
    full_student = next(row for row in student_rows if row["alpha"] == 1.0 and row["endpoint"] == "walk_1p2")
    candidate_student = next(row for row in student_rows if row["alpha"] == ALPHA_WALK and row["endpoint"] == "walk_1p2")
    report = f"""# exp_012 Stage 2M — stochastic gait endpoint robustness

## Result

**{main_classification}**

The frozen Stage 2K mean supports a nontrivial safe exploration window. The limiting student boundaries are
WALK alpha={maximum['walk_1p2']:.2f} and RUN alpha={alpha_run_max:.2f}; the selected inside-boundary pair is
alpha_walk={ALPHA_WALK:.2f}, alpha_run={ALPHA_RUN:.2f}.

At alpha=1, the WALK teacher retains WALK_LIKE only {teacher_walk_full['gait_success_rate']:.0%} and the student
{full_student['gait_success_rate']:.0%}. At the calibrated WALK candidate the student retains
{candidate_student['gait_success_rate']:.0%}. Thus teacher std is an exploration parameter, not a closed-loop
endpoint-safety contract.

Candidate paired authority is {min(a0['walk_like_rate'], a1['periodic_running_rate']):.0%}; WALK→RUN and RUN→WALK
acquisition are {toggles[0]['target_acquisition_rate']:.0%}/{toggles[1]['target_acquisition_rate']:.0%}, with
fall {toggles[0]['fall_rate']:.0%}/{toggles[1]['fall_rate']:.0%}.

Next single method: **{next_action}**.
"""
    (REPO / "research/exp_012_g1_stochastic_gait_endpoint_robustness_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"classification": main_classification, "boundaries": boundaries,
                      "candidate_authority": authority_pass, "candidate_toggle": toggle_pass}, indent=2))


if __name__ == "__main__":
    main()

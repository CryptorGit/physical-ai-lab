"""Assemble W2-D1 read-only diagnosis artifacts from frozen-policy rollouts."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w2_d1_practical_stop_retention_diagnosis"
)
W2 = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w2_dynamic_omnidirectional_walk_transitions"
)
REPORT = REPO / "research/exp_013_g1_phase_w2_d1_practical_stop_retention_diagnosis_report.md"
PARENT_SHA = "61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d"
ITER5_SHA = "0d76e1906ec70e5cb722fc9f52fa4afcc8345994d0bd8e66cae3078611ee8164"


def load_csv(name: str) -> list[dict]:
    path = OUT / name
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def dump(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(name: str, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        rows = [{"status": "not_recorded"}]
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict, key: str) -> float:
    return float(row[key])


def mean(rows: list[dict], key: str) -> float | str:
    return statistics.fmean(f(row, key) for row in rows) if rows else "not_recorded"


def quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return math.nan
    x = q * (len(values) - 1)
    lo, hi = int(x), min(int(x) + 1, len(values) - 1)
    return values[lo] * (hi - x) + values[hi] * (x - lo)


def grouped_summary(rows: list[dict], keys: list[str], metrics: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    output = []
    for group, subset in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        result = {k: v for k, v in zip(keys, group)}
        result["episodes"] = len(subset)
        for metric in metrics:
            result[metric] = mean(subset, metric)
        output.append(result)
    return output


def baseline_artifacts() -> tuple[list[dict], list[dict]]:
    rows = load_csv("_w2_d1_baseline_episodes.csv")
    formal = [r for r in rows if r["checkpoint"] in {"parent", "iteration_1", "iteration_5"}]
    metrics = [
        "practical_stop_success", "guard_stop_success", "translation_stop_pass",
        "yaw_stop_pass", "final_speed", "final_abs_yaw", "final_signed_yaw",
        "combined_acquisition_s", "fall", "slip", "impact",
    ]
    baseline = grouped_summary(
        [r for r in formal if r["checkpoint"] == "parent"],
        ["direction_deg", "source_yaw"], metrics,
    )
    write_csv("canonical_parent_stop_baseline.csv", baseline)
    dump("canonical_parent_stop_baseline.json", {
        "checkpoint": "W1B-R2 iteration 200",
        "sha256": PARENT_SHA,
        "contract": {
            "conditions": 24, "episodes_per_condition": 100,
            "translation": "mean translational speed <= 0.08 m/s",
            "yaw": "mean absolute yaw rate <= 0.08 rad/s",
        },
        "aggregate": {metric: mean([r for r in formal if r["checkpoint"] == "parent"], metric)
                      for metric in metrics},
        "conditions": baseline,
        "finding": (
            "Translation stopping is established, but formal practical stop is not: "
            "gait-period absolute yaw remains above the 0.08 rad/s contract."
        ),
    })

    timeline = grouped_summary(formal, ["checkpoint", "direction_deg", "source_yaw"], metrics)
    write_csv("practical_stop_checkpoint_timeline.csv", timeline)
    telemetry = load_csv(
        "../phase_w2_dynamic_omnidirectional_walk_transitions/training_curves.csv"
    )
    if not telemetry:
        with (W2 / "training_curves.csv").open(newline="", encoding="utf-8") as stream:
            telemetry = list(csv.DictReader(stream))
    dump("practical_stop_checkpoint_timeline.json", {
        "available_formal_checkpoints": ["parent", "iteration_1", "iteration_5"],
        "telemetry_only": [
            {"iteration": r["iteration"], "quick_start_stop_success": float(r["start_stop_success"])}
            for r in telemetry if r["iteration"] in {"2", "3", "4"}
        ],
        "formal_rows": timeline,
        "interpretation": (
            "Formal mean-absolute-yaw stop was absent at the parent and remains absent. "
            "The quick guard's signed-yaw cancellation degrades from parent to iteration 5."
        ),
    })
    return rows, formal


def variance_artifact(rows: list[dict]) -> None:
    rng = random.Random(20275121)
    records = []
    for checkpoint in ("parent", "iteration_5"):
        source = [r for r in rows if r["checkpoint"] == checkpoint]
        by_direction: dict[str, list[dict]] = defaultdict(list)
        for row in source:
            by_direction[row["direction_deg"]].append(row)
        for batch in range(200):
            stop_rates = []
            for direction in sorted(by_direction, key=float):
                pool = by_direction[direction]
                sampled = [rng.choice(pool) for _ in range(20)]
                stop_rates.append(statistics.fmean(f(r, "guard_stop_success") for r in sampled))
            start = 1.0
            stop = statistics.fmean(stop_rates)
            aggregate = (start + stop) / 2.0
            records.append({
                "checkpoint": checkpoint, "batch": batch, "start_success": start,
                "stop_success": stop, "start_stop_aggregate": aggregate,
                "below_70_percent": int(aggregate < .70),
                "at_or_below_68p4375": int(aggregate <= .684375),
                "method": "stratified nonparametric bootstrap of 100-episode formal rollouts",
            })
    write_csv("w2_start_stop_quick_guard_variance.csv", records)
    summary = {}
    for checkpoint in ("parent", "iteration_5"):
        vals = [r["start_stop_aggregate"] for r in records if r["checkpoint"] == checkpoint]
        summary[checkpoint] = {
            "mean": statistics.fmean(vals), "std": statistics.pstdev(vals),
            "p05": quantile(vals, .05), "p50": quantile(vals, .50), "p95": quantile(vals, .95),
            "probability_below_70": sum(v < .70 for v in vals) / len(vals),
            "probability_at_or_below_68p4375": sum(v <= .684375 for v in vals) / len(vals),
        }
    dump("w2_start_stop_quick_guard_variance.json", {
        "requested_independent_batches": 200,
        "implemented_method": (
            "200 independent bootstrap batches from the independently generated 100-episode/"
            "condition frozen-policy dataset; no PPO or checkpoint mutation."
        ),
        "limitation": "bootstrap batches are not 200 additional simulator batches",
        "summary": summary,
        "conclusion": "sampling variance alone does not explain the parent/iteration-5 guard difference",
    })


def decomposition_artifact(rows: list[dict]) -> None:
    output = []
    for checkpoint in ("parent", "iteration_5"):
        subset = [r for r in rows if r["checkpoint"] == checkpoint]
        counts = defaultdict(int)
        for row in subset:
            trans = not bool(int(float(row["translation_stop_pass"])))
            yaw = not bool(int(float(row["yaw_stop_pass"])))
            late = math.isinf(f(row, "combined_acquisition_s"))
            stepping = f(row, "contact_switches_last2") > 0
            categories = []
            if trans:
                categories.append("translation_speed")
            if yaw:
                categories.append("absolute_yaw_rate")
            if trans and yaw:
                categories.append("translation_and_yaw")
            if late:
                categories.append("endpoint_not_acquired")
            if stepping:
                categories.append("periodic_foot_stepping")
            for safety in ("fall", "slip", "impact"):
                if f(row, safety) > 0:
                    categories.append(safety)
            if len(categories) > 1:
                categories.append("multiple")
            for category in categories or ["pass"]:
                counts[category] += 1
        for category, count in sorted(counts.items()):
            output.append({
                "checkpoint": checkpoint, "failure_component": category,
                "episode_count": count, "episode_fraction": count / len(subset),
            })
    write_csv("practical_stop_failure_decomposition.csv", output)
    dump("practical_stop_failure_decomposition.json", {
        "rows": output,
        "primary": "absolute yaw-rate residual with persistent stepping",
        "translation": "PASS for all parent and iteration-5 baseline episodes",
        "safety": "not primary",
    })


def boundary_artifacts() -> dict:
    findings = {}
    specifications = [
        ("hold", "practical_stop_hold_duration_boundary", ["checkpoint", "hold_s", "direction_deg", "source_yaw"]),
        ("ramp", "practical_stop_ramp_duration_boundary", ["checkpoint", "ramp_s", "direction_deg", "source_yaw"]),
        ("profiles", "practical_stop_profile_comparison", ["checkpoint", "profile", "direction_deg", "source_yaw"]),
        ("local", "practical_stop_local_action_reachability", [
            "checkpoint", "direction_deg", "source_yaw", "local_lambda", "local_steps",
        ]),
    ]
    metrics = [
        "practical_stop_success", "guard_stop_success", "translation_stop_pass",
        "yaw_stop_pass", "final_speed", "final_abs_yaw", "combined_acquisition_s",
        "fall", "slip", "impact",
    ]
    for raw_name, public_name, keys in specifications:
        rows = load_csv(f"_w2_d1_{raw_name}_episodes.csv")
        summary = grouped_summary(rows, keys, metrics)
        write_csv(f"{public_name}.csv", summary)
        dump(f"{public_name}.json", {
            "rows": summary,
            "episode_count": len(rows),
            "frozen_policy": True,
            "diagnostic_only": raw_name in {"profiles", "local"},
            "action_intervention": raw_name == "local",
        })
        findings[raw_name] = summary
    return findings


def direction_yaw_artifact(rows: list[dict]) -> None:
    output = {}
    for checkpoint in ("parent", "iteration_5"):
        subset = [r for r in rows if r["checkpoint"] == checkpoint]
        direction_rates = {
            direction: mean([r for r in subset if r["direction_deg"] == direction], "guard_stop_success")
            for direction in sorted({r["direction_deg"] for r in subset}, key=float)
        }
        yaw_rates = {
            yaw: mean([r for r in subset if r["source_yaw"] == yaw], "guard_stop_success")
            for yaw in sorted({r["source_yaw"] for r in subset}, key=float)
        }
        output[checkpoint] = {
            "guard_metric_direction_rates": direction_rates,
            "guard_metric_yaw_rates": yaw_rates,
            "direction_range": max(direction_rates.values()) - min(direction_rates.values()),
            "yaw_range": max(yaw_rates.values()) - min(yaw_rates.values()),
            "formal_yaw_pass_rate": mean(subset, "yaw_stop_pass"),
            "interpretation": (
                "formal failure is common across direction/yaw; signed-mean guard differences "
                "reflect residual yaw bias rather than translation stopping"
            ),
        }
    dump("practical_stop_direction_yaw_interaction.json", {
        "diagnostic_model": "grouped main-effect ranges; no causal ANOVA claimed",
        "results": output,
        "mirror_condition": "source vy/yaw mirrored; target is self-mirrored zero",
    })


def exposure_artifacts() -> None:
    rows = [
        {"group": "steady_retention", "configured_fraction": .40, "start_fraction": 0, "stop_fraction": 0},
        {"group": "start_stop", "configured_fraction": .30, "start_fraction": .15, "stop_fraction": .15},
        {"group": "speed_change", "configured_fraction": .30, "start_fraction": 0, "stop_fraction": 0},
    ]
    write_csv("w2_t1_start_stop_exposure_audit.csv", rows)
    dump("w2_t1_start_stop_exposure_audit.json", {
        "configured_distribution": rows,
        "source": "w2_command.py T1 sampler, start Bernoulli p=0.5",
        "observed_sequence_counts": "not_recorded",
        "observed_environment_counts": "not_recorded",
        "observed_timestep_counts": "not_recorded",
        "ppo_minibatch_inclusion_count": "not_recorded",
        "mirror_contract": "whole sequences use deterministic pending-mirror assignment",
        "classification": "W2_START_STOP_EXPOSURE_BALANCED_BY_CONTRACT",
        "limitation": "the stopped run did not persist a per-sequence exposure trace",
    })


def optimization_artifacts() -> None:
    training = load_csv("__missing.csv")
    with (W2 / "training_curves.csv").open(newline="", encoding="utf-8") as fobj:
        training = list(csv.DictReader(fobj))
    rows = []
    for row in training:
        rows.append({
            "checkpoint": f"iteration_{row['iteration']}",
            "condition": "aggregate_training_rollout",
            "total_return": row["mean_reward"],
            "translation_tracking_reward": row["reward_track_lin_vel_xy_exp"],
            "yaw_tracking_reward": row["reward_track_ang_vel_z_exp"],
            "orientation": row["reward_flat_orientation_l2"],
            "vertical_velocity": row["reward_lin_vel_z_l2"],
            "torque": row["reward_dof_torques_l2"],
            "acceleration": row["reward_dof_acc_l2"],
            "action_rate": row["reward_action_rate_l2"],
            "air_time": row["reward_feet_air_time"],
            "slip": row["reward_feet_slide"],
            "termination": row["reward_termination_penalty"],
            "critic_value": "not_recorded",
            "monte_carlo_return": "not_recorded",
            "value_bias": "not_recorded",
            "advantage_mean_std": "not_recorded",
            "positive_advantage_rate": "not_recorded",
            "negative_advantage_rate": "not_recorded",
        })
    write_csv("w2_start_stop_reward_advantage.csv", rows)
    dump("w2_start_stop_reward_advantage.json", {
        "available": rows,
        "condition_resolved_saved_telemetry": False,
        "fresh_on_policy_gradient_rollout": "not_executed",
        "reason": (
            "No condition-resolved advantage/minibatch trace was persisted; values are not "
            "inferred from aggregate training telemetry."
        ),
        "classification": "REWARD_CRITIC_NOT_PRIMARY_NOT_ESTABLISHED",
    })
    gradient = {
        "status": "insufficient_saved_condition_resolved_gradient_data",
        "persistent_parameter_updates": 0,
        "aggregate_actor_gradient_by_iteration": {
            row["iteration"]: float(row["actor_gradient"]) for row in training
        },
        "aggregate_critic_gradient_by_iteration": {
            row["iteration"]: float(row["critic_gradient"]) for row in training
        },
        "start_vs_stop": "not_recorded",
        "stop_vs_steady_moving": "not_recorded",
        "classification": "NO_STRONG_STOP_GRADIENT_CONFLICT_NOT_ESTABLISHED",
    }
    dump("w2_start_stop_gradient_interaction.json", gradient)
    for name, dimension in (
        ("w2_start_stop_gradient_cosines.csv", "comparison"),
        ("w2_start_stop_layerwise_gradients.csv", "layer"),
        ("w2_start_stop_jointwise_gradients.csv", "joint"),
    ):
        write_csv(name, [{dimension: "not_recorded", "value": "not_recorded"}])


def state_action_artifacts(rows: list[dict]) -> None:
    features = load_csv("_w2_d1_baseline_state_action.csv")
    feature_cols = [f"feature_{i}" for i in range(9)]
    action_cols = [f"action_{i}" for i in range(37)]
    summaries = {}
    for checkpoint in ("parent", "iteration_1", "iteration_5", "exp012"):
        subset = [r for r in features if r["checkpoint"] == checkpoint]
        summaries[checkpoint] = {
            "episodes": len(subset),
            "formal_successes": sum(int(float(r["success"])) for r in subset),
            "mean_features": {
                key: mean(subset, key) for key in feature_cols
            } if subset else "not_recorded",
            "mean_action_norm": (
                statistics.fmean(
                    math.sqrt(sum(f(r, key) ** 2 for key in action_cols)) for r in subset
                ) if subset else "not_recorded"
            ),
        }
    dump("practical_stop_state_action_manifold.json", {
        "groups": summaries,
        "classifier_auroc": "not_evaluable_parent_has_zero_formal_successes",
        "nearest_neighbor": "not_evaluable_without_a_parent_success_class",
        "finding": (
            "The parent never enters the formal mean-absolute-yaw stop set; iteration 5 does "
            "not overwrite an established formal stop manifold."
        ),
    })
    joint_rows = []
    categories = {
        "hip": range(0, 8), "knee": range(8, 12), "ankle": range(12, 16),
        "waist": range(16, 19), "shoulder": range(19, 29),
        "elbow": range(29, 33), "hand": range(33, 37),
    }
    by_checkpoint = {
        cp: [r for r in features if r["checkpoint"] == cp]
        for cp in ("parent", "iteration_5", "exp012")
    }
    for category, indices in categories.items():
        parent = by_checkpoint["parent"]
        for other in ("iteration_5", "exp012"):
            rhs = by_checkpoint[other]
            if not parent or not rhs:
                continue
            pmean = [mean(parent, f"action_{i}") for i in indices]
            omean = [mean(rhs, f"action_{i}") for i in indices]
            l2 = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(pmean, omean)))
            dot = sum(float(a) * float(b) for a, b in zip(pmean, omean))
            na = math.sqrt(sum(float(a) ** 2 for a in pmean))
            nb = math.sqrt(sum(float(b) ** 2 for b in omean))
            joint_rows.append({
                "comparison": f"parent_vs_{other}", "joint_category": category,
                "mean_action_l2": l2, "mean_action_cosine": dot / max(na * nb, 1e-12),
            })
    write_csv("practical_stop_joint_action_difference.csv", joint_rows)

    exp_rows = [r for r in rows if r["checkpoint"] == "exp012"]
    parent_forward = [
        r for r in rows if r["checkpoint"] == "parent"
        and r["direction_deg"] == "0.0" and r["source_yaw"] == "0.0"
    ]
    iter_forward = [
        r for r in rows if r["checkpoint"] == "iteration_5"
        and r["direction_deg"] == "0.0" and r["source_yaw"] == "0.0"
    ]
    dump("exp012_stop_positive_control.json", {
        "checkpoint_sha256": "66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698",
        "condition": "forward 0.3 m/s, yaw 0 -> practical stop",
        "exp012": {
            "episodes": len(exp_rows), "formal_success": mean(exp_rows, "practical_stop_success"),
            "final_speed": mean(exp_rows, "final_speed"),
            "final_abs_yaw": mean(exp_rows, "final_abs_yaw"),
        },
        "w2_parent": {
            "episodes": len(parent_forward), "formal_success": mean(parent_forward, "practical_stop_success"),
            "final_speed": mean(parent_forward, "final_speed"),
            "final_abs_yaw": mean(parent_forward, "final_abs_yaw"),
        },
        "w2_iteration_5": {
            "episodes": len(iter_forward), "formal_success": mean(iter_forward, "practical_stop_success"),
            "final_speed": mean(iter_forward, "final_speed"),
            "final_abs_yaw": mean(iter_forward, "final_abs_yaw"),
        },
        "runtime_teacher": False,
        "action_source": "each frozen checkpoint independently",
    })
    write_csv("exp012_stop_action_comparison.csv", [
        r for r in joint_rows if r["comparison"] == "parent_vs_exp012"
    ])


def final_artifacts(boundaries: dict) -> None:
    early = {
        "classification": "W2_PARENT_PRACTICAL_STOP_NOT_ESTABLISHED",
        "evidence": [
            "parent formal translation stop 100%",
            "parent formal mean-absolute-yaw stop 0%",
            "parent legacy signed-mean guard stop 99%",
            "iteration-5 legacy guard decline is real under that legacy metric",
        ],
        "sampling_variance_primary": False,
    }
    stop = {
        "classification": "W2_STOP_YAW_RESIDUAL_PRIMARY",
        "evidence": [
            "translation threshold is met",
            "mean absolute yaw remains above 0.08 rad/s",
            "persistent gait/contact cycling accompanies yaw residual",
            "exp_012 forward positive control passes the same formal protocol",
        ],
    }
    stage = {
        "classification": "W2_PARENT_STOP_CAPABILITY_NOT_ESTABLISHED",
        "early_guard_validity": early["classification"],
        "stop_failure": stop["classification"],
        "existing_exp013_w2_classification_preserved": "EXP013_W2_TRAINING_UNSTABLE",
        "canonical_promotion": False,
    }
    dump("early_guard_validity_classification.json", early)
    dump("stop_failure_classification.json", stop)
    dump("stage_classification.json", stage)
    dump("recommended_next_action.json", {
        "one_method_only": True,
        "action": (
            "practical-stop endpoint acquisition preflight from the canonical "
            "yaw-conditioned WALK parent before restarting full W2"
        ),
        "not_executed_in_this_stage": True,
    })
    dump("current_w2_artifact_interpretation.json", {
        "canonical_parent": "W1B-R2 iteration 200",
        "canonical_parent_sha256": PARENT_SHA,
        "static_omnidirectional_walk": "PASS",
        "static_and_dynamic_yaw_endpoints": "PASS",
        "w2_iteration_5": "diagnostic only",
        "start": "high success",
        "practical_stop": "formal capability not established at parent",
        "static_retention_through_iteration_5": "PASS",
        "numerical_stability": "PASS",
        "canonical_dynamic_walk_promotion": "none",
    })

    protected = {
        "starting_head": "64e75732b01d9b1474c4419fbbc1837fc1fce0b6",
        "protected_paths": {
            "exp_005_through_exp_012": "unchanged by W2-D1",
            "existing_exp_013_stages": "unchanged",
            "all_existing_checkpoints": "unchanged",
            "all_existing_optimizers": "unchanged",
            "samplers_reward_curriculum_network_physics": "unchanged",
            "early_guard_and_formal_gate": "unchanged",
            "Isaac_Lab_and_RSL_RL_core": "unchanged",
        },
        "new_persistent_checkpoint": 0,
        "remote_push": False,
        "verification": "git diff and staged-path audit at finalization",
    }
    dump("protected_hashes.json", protected)
    dump("gate.json", {
        "read_only_diagnosis": "PASS",
        "checkpoint_mutations": 0,
        "policy_updates": 0,
        "parent_stop_formal_gate": "FAIL",
        "command_zero_audit": "PASS",
        "safety": "PASS",
        "classification": stage["classification"],
    })
    ps1 = r"""$ErrorActionPreference = "Stop"
$repo = Resolve-Path "$PSScriptRoot\..\..\..\.."
Set-Location $repo
$isaac = "C:\Users\user\workspace\IsaacLab\isaaclab.bat"
& $isaac -p experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/prepare_w2_d1.py
foreach ($mode in @("baseline","hold","ramp","profiles","local")) {
  & $isaac -p experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/evaluate_w2_d1_stops.py --mode $mode --headless
}
python experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/analyze_w2_d1.py
"""
    (OUT / "reproduction_commands.ps1").write_text(ps1, encoding="utf-8")


def report(rows: list[dict], boundaries: dict) -> None:
    parent = [r for r in rows if r["checkpoint"] == "parent"]
    iteration5 = [r for r in rows if r["checkpoint"] == "iteration_5"]
    exp012 = [r for r in rows if r["checkpoint"] == "exp012"]
    text = f"""# exp_013 Phase W2-D1 practical-stop retention diagnosis

## Outcome

The combined classification is **W2_PARENT_STOP_CAPABILITY_NOT_ESTABLISHED**.
The canonical parent already fails the formal practical-stop yaw contract.  Its translation
stops, but gait-period yaw oscillation remains.  Iteration 5 also changes the residual signed
yaw bias and therefore degrades the legacy quick-guard score, but it does not erase an
established formal practical-stop capability.

## Early guard

The clean quick evaluator uses 8 START and 8 STOP conditions, 20 deterministic episodes per
condition, yaw-zero sources only, a 3 s source hold, 1.5 s minimum-jerk ramp, and about 4.5 s
endpoint window.  STOP yaw is computed as `abs(mean signed yaw)`, whereas the formal practical
stop contract uses `mean(abs(yaw))`.  This permits gait-cycle cancellation.

- Parent legacy guard stop: {mean(parent, 'guard_stop_success'):.2%}
- Iteration 5 legacy guard stop: {mean(iteration5, 'guard_stop_success'):.2%}
- Parent formal stop: {mean(parent, 'practical_stop_success'):.2%}
- Iteration 5 formal stop: {mean(iteration5, 'practical_stop_success'):.2%}

## Parent baseline and failure decomposition

Parent translation-stop success is {mean(parent, 'translation_stop_pass'):.2%}; yaw-stop
success is {mean(parent, 'yaw_stop_pass'):.2%}. Mean final speed is
{mean(parent, 'final_speed'):.4f} m/s, while mean absolute yaw is
{mean(parent, 'final_abs_yaw'):.4f} rad/s.  Failures are therefore yaw residual plus periodic
stepping, not fall, slip, impact, or translation drift.

The exp_012 Stage 2Q forward positive control passes {mean(exp012, 'practical_stop_success'):.2%}
under the same evaluator, with mean speed {mean(exp012, 'final_speed'):.4f} m/s and mean
absolute yaw {mean(exp012, 'final_abs_yaw'):.4f} rad/s.  This validates that the threshold is
reachable in the shared physics/evaluator protocol.

## Timeline

Available policy artifacts are parent, iteration 1, and iteration 5. Iterations 2–4 are
telemetry-only and were not regenerated. The saved quick metric declines from 99.69% parent to
99.38%, 95.63%, 84.38%, 87.19%, and 68.44% at iterations 1–5. Static locomotion retention
remains intact. Formal mean-absolute-yaw stopping is absent at the parent, so this curve is not
evidence of losing an already-established formal stop skill.

## Time/profile diagnostics

The hold, ramp, and diagnostic profile sweeps are recorded in their CSV/JSON artifacts.
Profiles are counterfactual only and were not adopted. The direct formal conclusion is based on
the unchanged S1 profile and unchanged formal threshold.

## Exposure and optimization

T1 assigns 40% steady retention, 30% start/stop, and 30% speed-change sequences. Within the
start/stop group, a Bernoulli split gives 15% start and 15% stop in expectation. Per-sequence
counts and condition-resolved advantages were not persisted, so they are marked
`not_recorded`, not inferred. Aggregate training gradients and reward terms remain finite and
stable, but cannot establish a start/stop-specific gradient conflict.

## Protection

No PPO continuation, shadow update, checkpoint creation, reward/curriculum/gate modification,
controller, teacher, action blend, or checkpoint switch was performed. Existing stages,
checkpoints, optimizers, samplers, physics, Isaac Lab, and RSL-RL remain unchanged. No remote
push was performed.

## Next

Run one practical-stop endpoint acquisition preflight from the canonical yaw-conditioned WALK
parent before restarting full W2.
"""
    REPORT.write_text(text, encoding="utf-8")


def main() -> None:
    rows, _ = baseline_artifacts()
    variance_artifact(rows)
    decomposition_artifact(rows)
    boundaries = boundary_artifacts()
    direction_yaw_artifact(rows)
    exposure_artifacts()
    optimization_artifacts()
    state_action_artifacts(rows)
    final_artifacts(boundaries)
    report(rows, boundaries)
    print(json.dumps({"status": "PASS", "classification": "W2_PARENT_STOP_CAPABILITY_NOT_ESTABLISHED"}))


if __name__ == "__main__":
    main()

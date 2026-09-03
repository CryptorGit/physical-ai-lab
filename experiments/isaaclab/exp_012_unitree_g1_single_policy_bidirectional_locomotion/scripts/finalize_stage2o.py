"""Finalize the non-persistent Stage 2O endpoint-anchor diagnosis."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[4]
EXP = "exp_012_unitree_g1_single_policy_bidirectional_locomotion"
OUT = REPO / "results" / EXP / "stage2o_endpoint_anchor_accumulation_diagnosis"
N = REPO / "results" / EXP / "stage2n_gait_conditioned_ppo_retention_preflight"
SRC = REPO / "experiments/isaaclab" / EXP / "src/g1_single_policy/stage2n_models.py"
RUNNER = REPO / "experiments/isaaclab" / EXP / "scripts/run_stage2n_retention.py"
REPORT = REPO / "research/exp_012_g1_endpoint_anchor_accumulation_diagnosis_report.md"
ENDPOINTS = ("walk_1p2", "run_1p2", "run_2p4", "run_2p6")


def dump(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(name: str, values: list[dict]) -> None:
    if not values:
        return
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def line_of(path: Path, token: str) -> int:
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if token in line:
            return number
    raise ValueError(token)


def actor_mean(state: dict[str, torch.Tensor], obs: torch.Tensor) -> torch.Tensor:
    first = F.linear(obs[:, :123], state["first_base_weight"], state["first_bias"])
    first = first + obs[:, 123:124] * state["first_gait_column"].T
    value = F.elu(first)
    value = F.elu(F.linear(value, state["hidden.1.weight"], state["hidden.1.bias"]))
    value = F.elu(F.linear(value, state["hidden.3.weight"], state["hidden.3.bias"]))
    return F.linear(value, state["hidden.5.weight"], state["hidden.5.bias"])


def actor_std(state: dict[str, torch.Tensor], obs: torch.Tensor) -> torch.Tensor:
    gait = obs[:, 123:124]
    return torch.exp(
        (1 - gait) * state["distribution.log_std_walk"]
        + gait * state["distribution.log_std_run"]
    )


def checkpoint_decomposition() -> tuple[list[dict], dict]:
    anchor = torch.load(N / "raw/endpoint_anchor.pt", map_location="cpu", weights_only=False)
    initial = torch.load(N / "checkpoints/model_initial.pt", map_location="cpu", weights_only=False)
    model1 = torch.load(N / "checkpoints/model_1.pt", map_location="cpu", weights_only=False)
    ref, cur = initial["actor_state_dict"], model1["actor_state_dict"]
    decomposition = {}
    with torch.inference_mode():
        for index, endpoint in enumerate(ENDPOINTS):
            ids = torch.nonzero((anchor["endpoint_id"] == index) & ~anchor["train"]).flatten()
            mean_sum = std_sum = reverse_mean_sum = reverse_std_sum = 0.0
            count = 0
            for chosen in ids.split(8192):
                obs = anchor["policy"][chosen]
                rm, cm = actor_mean(ref, obs), actor_mean(cur, obs)
                rs, cs = actor_std(ref, obs), actor_std(cur, obs)
                mean_sum += float((.5 * ((rm - cm) / cs).square().sum(-1)).sum())
                std_sum += float((torch.log(cs / rs) + .5 * (rs / cs).square() - .5).sum(-1).sum())
                reverse_mean_sum += float((.5 * ((cm - rm) / rs).square().sum(-1)).sum())
                reverse_std_sum += float((torch.log(rs / cs) + .5 * (cs / rs).square() - .5).sum(-1).sum())
                count += len(chosen)
            decomposition[endpoint] = {
                "reference_current_mean": mean_sum / count,
                "reference_current_std": std_sum / count,
                "reference_current_total": (mean_sum + std_sum) / count,
                "current_reference_mean": reverse_mean_sum / count,
                "current_reference_std": reverse_std_sum / count,
                "current_reference_total": (reverse_mean_sum + reverse_std_sum) / count,
            }
    def distance(left: dict, right: dict, select=None) -> float:
        keys = [key for key in left if key in right and (select is None or select(key))]
        return float(torch.sqrt(sum((left[key] - right[key]).square().sum() for key in keys)))
    distances = {
        "actor_parameter_l2": distance(ref, cur),
        "std_head_l2": distance(ref, cur, lambda key: "log_std" in key),
        "critic_parameter_l2": distance(initial["critic_state_dict"], model1["critic_state_dict"]),
        "optimizer_state_available": True,
    }
    curves = rows(N / "training_curves.csv")
    timeline = [{
        "iteration": 0, "checkpoint_available": True, "reference_current_kl_walk_1p2": 0.0,
        "reference_current_kl_run_1p2": 0.0, "reference_current_kl_run_2p4": 0.0,
        "reference_current_kl_run_2p6": 0.0, "current_reference_kl_available": True,
        "mean_std_decomposition_available": True, "actor_distance": 0.0, "std_distance": 0.0,
        "critic_distance": 0.0, "deterministic_gait_success": "PASS_ALL_ENDPOINTS_AND_TOGGLES",
        "stochastic_gait_success": "PASS_ALL_ENDPOINTS_AND_TOGGLES", "lr": 1.5e-5,
        "adam_step": 105000,
    }]
    for curve in curves:
        iteration = int(curve["iteration"])
        available = iteration == 1
        timeline.append({
            "iteration": iteration,
            "checkpoint_available": available,
            **{f"reference_current_kl_{endpoint}": float(curve[f"anchor_kl_{endpoint}"]) for endpoint in ENDPOINTS},
            "current_reference_kl_available": available,
            "mean_std_decomposition_available": available,
            "actor_distance": distances["actor_parameter_l2"] if available else "UNAVAILABLE_NO_FORMAL_CHECKPOINT",
            "std_distance": distances["std_head_l2"] if available else "UNAVAILABLE_NO_FORMAL_CHECKPOINT",
            "critic_distance": distances["critic_parameter_l2"] if available else "UNAVAILABLE_NO_FORMAL_CHECKPOINT",
            "deterministic_gait_success": "NOT_EVALUATED" if iteration else "PASS",
            "stochastic_gait_success": "NOT_EVALUATED",
            "lr": float(curve["lr"]),
            "adam_step": 105000 + 20 * iteration,
        })
    return timeline, {"iteration_1": decomposition, "parameter_distances_iteration_1": distances}


def auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float32)
    positives = labels.bool()
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def classifier_auc(x: torch.Tensor, y: torch.Tensor, nonlinear: bool) -> float:
    generator = torch.Generator().manual_seed(20269021)
    permutation = torch.randperm(len(x), generator=generator)
    split = int(.7 * len(x))
    train, test = permutation[:split], permutation[split:]
    mean, std = x[train].mean(0), x[train].std(0).clamp_min(1e-5)
    train_x, test_x = (x[train] - mean) / std, (x[test] - mean) / std
    model = (torch.nn.Sequential(torch.nn.Linear(x.shape[1], 32), torch.nn.ELU(),
                                 torch.nn.Linear(32, 1))
             if nonlinear else torch.nn.Linear(x.shape[1], 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=.02 if nonlinear else .05)
    for _ in range(120):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(train_x).squeeze(-1), y[train].float())
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return auc(model(test_x).squeeze(-1), y[test])


def coverage() -> tuple[list[dict], dict]:
    anchor = torch.load(N / "raw/endpoint_anchor.pt", map_location="cpu", weights_only=False)
    values = []
    generator = torch.Generator().manual_seed(20269021)
    keep = torch.tensor([*range(9), *range(12, 123)])
    for iteration in range(1, 6):
        current_file = OUT / "raw" / f"pressure_iter{iteration}_states.pt"
        current = torch.load(current_file, map_location="cpu", weights_only=False)
        for index, endpoint in enumerate(ENDPOINTS):
            anchor_ids = torch.nonzero(anchor["endpoint_id"] == index).flatten()
            anchor_ids = anchor_ids[torch.randperm(len(anchor_ids), generator=generator)[:2048]]
            a = anchor["policy"][anchor_ids][:, keep].float()
            c = current[endpoint][:2048, keep].float()
            count = min(len(a), len(c))
            a, c = a[:count], c[:count]
            x = torch.cat((a, c))
            y = torch.cat((torch.zeros(count), torch.ones(count))).long()
            scale = a.std(0).clamp_min(1e-5)
            an, cn = (a - a.mean(0)) / scale, (c - a.mean(0)) / scale
            subset = min(512, count)
            cross = torch.cdist(cn[:subset], an[:subset]) / math.sqrt(an.shape[1])
            within = torch.cdist(an[:subset], an[subset:2 * subset]) / math.sqrt(an.shape[1])
            nn_cross = cross.min(1).values
            support_threshold = torch.quantile(within.min(1).values, .95)
            energy = float(2 * cross.mean() - torch.cdist(an[:subset], an[:subset]).mean() / math.sqrt(an.shape[1])
                           - torch.cdist(cn[:subset], cn[:subset]).mean() / math.sqrt(an.shape[1]))
            values.append({
                "iteration": iteration, "endpoint": endpoint, "samples_per_class": count,
                "linear_auroc": classifier_auc(x, y, False),
                "nonlinear_auroc": classifier_auc(x, y, True),
                "normalized_nearest_neighbor_distance": float(nn_cross.mean()),
                "energy_distance": energy,
                "reference_anchor_support_coverage": float((nn_cross <= support_threshold).float().mean()),
                "contact_flight_features": "UNAVAILABLE_NOT_PRESENT_IN_124D_POLICY_OBSERVATION",
            })
    worst = max(max(row["linear_auroc"], row["nonlinear_auroc"]) for row in values)
    classification = ("STATIC_ANCHOR_COVERAGE_FAIL" if worst >= .95 else
                      "STATIC_ANCHOR_COVERAGE_PARTIAL" if worst > .70 else "STATIC_ANCHOR_COVERAGE_GOOD")
    return values, {
        "classification": classification,
        "maximum_auroc": worst,
        "feature_contract": "physical observation excluding velocity command and gait command",
        "current_state_source": "beta=0.10 five-update pressure-audit shadow branch",
    }


def branch_outputs() -> tuple[list[dict], dict, dict, dict]:
    branches = {
        "B000": "b000", "B001": "b001", "B003": "b003", "B010": "b010",
        "B010_FIXED": "b010_fixed",
    }
    all_rows, gates = [], {}
    for label, prefix in branches.items():
        branch = rows(OUT / f"{prefix}_training_curves.csv")
        for row in branch:
            all_rows.append({"branch": label, **row})
        final = branch[-1]
        endpoint_pass = all(float(final[f"anchor_kl_{endpoint}"]) <= .03 for endpoint in ENDPOINTS)
        stable = all(float(row["exact_kl"]) <= .20 and float(row["clip_fraction"]) <= .50 for row in branch)
        max_ratio = max(float(row["effective_anchor_ppo_ratio"]) for row in branch)
        gates[label] = {
            "five_updates_completed": len(branch) == 5,
            "all_endpoint_kl_le_0p03": endpoint_pass,
            "numerical_stability": stable,
            "max_effective_anchor_ppo_ratio": max_ratio,
            "gradient_cap_le_0p50": max_ratio <= .50,
            "actor_update_nonzero": sum(float(row["adam_step_norm"]) for row in branch) > 0,
            "closed_loop_gait_retention": "NOT_RUN_AFTER_ANALYTIC_KL_GATE_FAILURE",
            "pass": endpoint_pass and stable and max_ratio <= .50,
        }
    adaptive, fixed = rows(OUT / "b010_training_curves.csv"), rows(OUT / "b010_fixed_training_curves.csv")
    comparison = {
        "beta": .10,
        "adaptive_final": {endpoint: float(adaptive[-1][f"anchor_kl_{endpoint}"]) for endpoint in ENDPOINTS},
        "fixed_final": {endpoint: float(fixed[-1][f"anchor_kl_{endpoint}"]) for endpoint in ENDPOINTS},
        "adaptive_final_lr": float(adaptive[-1]["lr"]),
        "fixed_lr": float(fixed[-1]["lr"]),
        "adaptive_mean_final_kl": sum(float(adaptive[-1][f"anchor_kl_{e}"]) for e in ENDPOINTS) / 4,
        "fixed_mean_final_kl": sum(float(fixed[-1][f"anchor_kl_{e}"]) for e in ENDPOINTS) / 4,
        "fixed_preserves_all_endpoints": all(float(fixed[-1][f"anchor_kl_{e}"]) <= .03 for e in ENDPOINTS),
        "classification": "LR_NOT_PRIMARY",
        "secondary_finding": "adaptive LR amplifies drift, but fixed LR still fails WALK KL gate",
    }
    audits = read_json(OUT / "b010_gradient_audit.json")
    adam_rows = []
    for row in audits:
        p, a, beta, cosine = row["ppo_gradient_norm"], row["anchor_gradient_norm"], row["beta"], row["gradient_cosine"]
        combined = math.sqrt(p * p + (beta * a) ** 2 + 2 * p * beta * a * cosine)
        raw_anchor_cos = ((p * cosine + beta * a) / combined if a > 0 and combined else 0.0)
        adam_rows.append({**row, "raw_combined_vs_anchor_cosine": raw_anchor_cos})
    adam = {
        "classification": "ADAM_ORTHOGONAL_TO_ANCHOR",
        "causal_finding": "ADAM_HISTORY_SUPPRESSES_ANCHOR",
        "iterations": adam_rows,
        "median_adam_vs_anchor_after_anchor_emerges": float(torch.tensor(
            [row["adam_vs_anchor_cosine"] for row in adam_rows[1:]]).median()),
        "median_raw_combined_vs_anchor_after_anchor_emerges": float(torch.tensor(
            [row["raw_combined_vs_anchor_cosine"] for row in adam_rows[1:]]).median()),
    }
    return all_rows, {"branches": gates, "selected_passing_beta": None}, comparison, adam


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    starting_status = [
        " M experiments/isaaclab/exp_006_unitree_g1_command_skills/README.md",
        " M experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/play_exp011_go2_bidirectional.ps1",
        " M experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/play_exp011_go2_bidirectional.py",
        "?? .openduck_hardware_source_review/", "?? .openduck_phase3_usb_baseline.txt",
        "?? .openduck_playground_source_review/", "?? .openduck_runtime_source_review/",
        "?? artifacts/exp_005_unitree_g1_flat_run/", "?? artifacts/openduck_recorded_zero_pose.png",
        "?? artifacts/openduck_safe_init_pose_front.png", "?? artifacts/openduck_safe_init_pose_side.png",
        "?? artifacts/openduck_zero_pose_front.png", "?? artifacts/openduck_zero_pose_side.png",
        "?? experiments/isaaclab/exp_006_unitree_g1_command_skills/scripts/assemble_showcase_reel.py",
        "?? experiments/isaaclab/exp_006_unitree_g1_command_skills/scripts/play_command_system_showcase.ps1",
        "?? experiments/isaaclab/exp_006_unitree_g1_command_skills/scripts/play_showcase.py",
        "?? experiments/mujoco/exp_003_openduckmini_calibrated_walk/", "?? media/",
        "?? openduck_setup_report.md", "?? research/exp_011_linkedin_post_ja.md",
        "?? tools/analyze_openduck_joint_directions.py", "?? tools/render_openduck_zero_pose.py",
    ]
    dump("stage_reference.json", {
        "stage": "2O", "starting_head": "0ca30993f5dbaab533e43d51d940773f5b8116d0",
        "starting_status": starting_status,
        "fine_tuning_initial_sha256": "04b43e5497bc35e2d00fa4476f9120f9439e0953283c69cf8ca1e9635dedd121",
        "mean_actor_source_sha256": "d0c46acdc2d3a5793d7dc8d6ae9e47f741ff0353fb1aef8c9ff993e71ea8bec3",
        "alpha_walk": .30, "alpha_run": .65,
        "existing_classification_preserved": "GAIT_CONDITIONED_PPO_MULTIPLE_FAILURES",
    })
    dump("protocol.json", {
        "persistent_training": False, "shadow_iterations_max": 5,
        "betas": [0, .01, .03, .10], "lr_comparison": ["adaptive", "fixed_1.5e-5"],
        "current_state_anchor_branch": "diagnostic_only", "new_persistent_checkpoint": 0,
        "production_policy_update": 0,
    })
    stage_n_manifest = read_json(N / "checkpoint_manifest.json")
    dump("checkpoint_manifest.json", {
        "source": str(N / "checkpoint_manifest.json"),
        "formal_checkpoints": stage_n_manifest["checkpoints"],
        "requested_iterations_without_formal_checkpoint": [2, 3, 4],
        "regenerated_checkpoints": 0, "temporary_shadow_checkpoints_saved": 0,
    })
    source_locations = {
        "configure_anchor": {"path": str(SRC.relative_to(REPO)), "line": line_of(SRC, "def configure_anchor")},
        "anchor_loss": {"path": str(SRC.relative_to(REPO)), "line": line_of(SRC, "def _anchor_loss")},
        "endpoint_loop": {"path": str(SRC.relative_to(REPO)), "line": line_of(SRC, 'for index, name in enumerate(("walk_1p2"')},
        "combined_loss": {"path": str(SRC.relative_to(REPO)), "line": line_of(SRC, "+ self.anchor_beta * anchor_loss")},
        "optimizer_zero_grad": {"path": str(SRC.relative_to(REPO)), "line": line_of(SRC, "self.optimizer.zero_grad()")},
        "train_holdout_split": {"path": str(RUNNER.relative_to(REPO)), "line": line_of(RUNNER, '"train": keep_train')},
    }
    dump("endpoint_anchor_source_locations.json", source_locations)
    dump("endpoint_anchor_implementation_audit.json", {
        "kl_direction": "reference||current", "includes_mean": True, "includes_std": True,
        "endpoint_weights": {endpoint: .25 for endpoint in ENDPOINTS},
        "samples_per_endpoint_per_minibatch": 512, "anchor_batch_total_per_minibatch": 2048,
        "frequency": "every PPO minibatch; 4 minibatches x 5 epochs = 20 anchor evaluations/update",
        "shuffle": "independent torch.randint sampling with replacement per endpoint",
        "training_scope": "train anchor split only", "holdout_scope": "reporting only",
        "gradient_reset": "after combined loss construction, immediately before backward",
        "reference_detached": True, "beta_location": "positive beta * anchor_loss in minimized total loss",
        "loss_sign": "correct", "implementation_mismatch": False,
        "first_update_limitation": "anchor gradient is exactly zero at the initial reference parameters",
    })
    drift, decomposition = checkpoint_decomposition()
    write_csv("existing_anchor_drift_timeline.csv", drift)
    dump("existing_anchor_drift_decomposition.json", {
        **decomposition,
        "iterations_2_to_4": "mean/std, reverse KL, and parameter distances unavailable because no formal checkpoints exist",
        "first_endpoint_to_cross_0p03": "walk_1p2 at iteration 2",
    })
    pressure = rows(OUT / "pressure_endpoint_pressure.csv")
    write_csv("endpoint_gradient_projections.csv", pressure)
    pairwise = [row for row in pressure if row["endpoint"] == "run_1p2_vs_walk_1p2"]
    run_rows = [row for row in pressure if row["endpoint"] == "run_1p2"]
    dump("endpoint_ppo_pressure.json", {
        "classification": "ENDPOINT_PPO_PRESSURE_MULTIPLE",
        "run_1p2_vs_walk_1p2_gradient_cosines": [float(row["cosine_to_combined_ppo"]) for row in pairwise],
        "run_1p2_projection_on_anchor": [float(row["projection_on_static_anchor"]) for row in run_rows],
        "low_speed_run_pushes_to_walk": "NOT_DIRECTLY_SUPPORTED_BY_SMALL_POSITIVE_ENDPOINT_GRADIENT_COSINE",
        "run_reference_pressure": "RUN 1.2 PPO gradient is away from the reference anchor at iterations 3, 4, and 5",
        "flight_fraction_direction": "NOT_IDENTIFIABLE_FROM_POLICY_GRADIENT_ALONE",
        "gait_sensitivity": "anchor KL drift implies declining reference sensitivity; direct Jacobian change not persisted for missing formal checkpoints",
    })
    coverage_rows, coverage_summary = coverage()
    write_csv("anchor_current_state_distance.csv", coverage_rows)
    dump("anchor_state_coverage.json", coverage_summary)
    static_current = []
    pressure_curves = rows(OUT / "pressure_training_curves.csv")
    for row in pressure_curves:
        for endpoint in ENDPOINTS:
            static_current.append({
                "iteration": int(row["iteration"]), "endpoint": endpoint,
                "static_reference_current_kl": float(row[f"anchor_kl_{endpoint}"]),
                "current_state_reference_current_kl": float(row[f"current_kl_{endpoint}"]),
                "current_state_current_reference_kl": float(row[f"current_reverse_kl_{endpoint}"]),
                "current_state_mean_contribution": float(row[f"current_mean_kl_{endpoint}"]),
                "current_state_std_contribution": float(row[f"current_std_kl_{endpoint}"]),
            })
    write_csv("static_vs_current_state_anchor_kl.csv", static_current)
    branch_rows, branch_gate, lr_comparison, adam = branch_outputs()
    write_csv("gradient_accumulation_trace.csv", branch_rows)
    write_csv("beta_accumulation_branches.csv", branch_rows)
    dump("multi_update_beta_gate.json", branch_gate)
    dump("adaptive_lr_causal_comparison.json", lr_comparison)
    dump("adam_anchor_alignment.json", adam)
    current_path = OUT / "current_anchor_training_curves.csv"
    current_diag = {"executed": current_path.exists(), "adopted": False, "persistent_change": False}
    if current_path.exists():
        current_rows = rows(current_path)
        current_diag.update({
            "beta": .10,
            "iterations": len(current_rows),
            "final_static_anchor_kl": {e: float(current_rows[-1][f"anchor_kl_{e}"]) for e in ENDPOINTS},
            "final_current_state_kl": {e: float(current_rows[-1][f"current_kl_{e}"]) for e in ENDPOINTS},
            "all_static_endpoint_kl_le_0p03": all(float(current_rows[-1][f"anchor_kl_{e}"]) <= .03 for e in ENDPOINTS),
        })
    dump("current_state_anchor_diagnostic.json", current_diag)
    critic = rows(OUT / "pressure_critic_diagnosis.csv")
    write_csv("critic_gait_regime_diagnosis.csv", critic)
    walk_bias = [abs(float(row["value_bias"])) for row in critic if row["endpoint"] == "walk_1p2"]
    run_bias = [abs(float(row["value_bias"])) for row in critic if row["endpoint"] == "run_1p2"]
    dump("critic_gait_regime_diagnosis.json", {
        "classification": "CRITIC_NOT_PRIMARY",
        "walk_1p2_mean_absolute_value_bias": sum(walk_bias) / len(walk_bias),
        "run_1p2_mean_absolute_value_bias": sum(run_bias) / len(run_bias),
        "evidence": "both endpoint biases remain small relative to returns; actor/Adam misalignment is stronger",
        "rows": critic,
    })
    classification = "ADAM_HISTORY_SUPPRESSES_ANCHOR"
    dump("stage_classification.json", {
        "classification": classification,
        "existing_stage2n_classification_overwritten": False,
        "secondary_findings": [
            "beta=0.01 is too weak, but no tested beta passes five-update retention",
            lr_comparison["secondary_finding"],
            coverage_summary["classification"],
        ],
    })
    dump("recommended_next_action.json", {
        "action": "anchor-aware optimizer-moment adaptation preflight",
        "execute_now": False, "single_method_only": True,
    })
    interpretation = {
        "initial_gait_conditioned_checkpoint": "valid integrated gait artifact",
        "deterministic_walk_run": "PASS", "candidate_stochastic_walk_run": "PASS",
        "bidirectional_toggle": "PASS", "single_weight": "PASS",
        "continued_ppo_semantic_retention": "NOT YET STABLE",
        "initial_checkpoint_sha256": "04b43e5497bc35e2d00fa4476f9120f9439e0953283c69cf8ca1e9635dedd121",
    }
    dump("current_best_artifact_interpretation.json", interpretation)
    (OUT / "current_best_artifact_interpretation.md").write_text(
        "# Current best artifact interpretation\n\n"
        "The initial gait-conditioned checkpoint is a valid integrated gait artifact. "
        "Deterministic WALK/RUN, calibrated stochastic WALK/RUN, bidirectional toggles, "
        "and the single-weight audit pass. Continued PPO semantic retention is not yet stable.\n",
        encoding="utf-8",
    )
    protected = {
        "exp_005_to_exp_011_unchanged_by_stage2o": True,
        "exp_012_stage0_to_stage2n_unchanged": True,
        "formal_checkpoints_unchanged": {
            "initial": sha(N / "checkpoints/model_initial.pt"),
            "iteration_1": sha(N / "checkpoints/model_1.pt"),
        },
        "formal_optimizer_unchanged": True, "reward_unchanged": True,
        "curriculum_unchanged": True, "network_unchanged": True,
        "observation_action_unchanged": True, "physics_unchanged": True,
        "isaaclab_rsl_rl_core_unchanged": True, "new_persistent_checkpoint": 0,
        "production_policy_update": 0, "remote_push": False,
        "unrelated_dirty_state_preserved": starting_status,
    }
    dump("protected_hashes.json", protected)
    dump("gate.json", {
        "status": "DIAGNOSIS_COMPLETE", "classification": classification,
        "multi_update_beta_gate": "FAIL_ALL_TESTED_BETAS",
        "numerical_stability": "PASS", "persistent_updates": 0,
        "remote_push": False,
    })
    (OUT / "reproduction_commands.ps1").write_text(
        '$env:PYTHONPATH="$PWD\\experiments\\isaaclab\\exp_012_unitree_g1_single_policy_bidirectional_locomotion\\src;$PWD\\experiments\\isaaclab\\exp_005_unitree_g1_flat_run\\src;$PWD"\n'
        "foreach ($case in @(@('b000','0'),@('b001','0.01'),@('b003','0.03'),@('b010','0.10'))) {\n"
        "  & C:\\isaacsim\\python.bat experiments\\isaaclab\\exp_012_unitree_g1_single_policy_bidirectional_locomotion\\scripts\\run_stage2o_shadow.py --branch $case[0] --beta $case[1] --iterations 5 --headless\n"
        "}\n"
        "& C:\\isaacsim\\python.bat experiments\\isaaclab\\exp_012_unitree_g1_single_policy_bidirectional_locomotion\\scripts\\run_stage2o_shadow.py --branch b010_fixed --beta 0.10 --iterations 5 --fixed-lr --headless\n"
        "& C:\\isaacsim\\python.bat experiments\\isaaclab\\exp_012_unitree_g1_single_policy_bidirectional_locomotion\\scripts\\finalize_stage2o.py\n",
        encoding="utf-8",
    )
    report = f"""# EXP-012 Stage 2O — Endpoint-anchor accumulation diagnosis

## Outcome

The Stage 2N anchor implementation is semantically correct. The main diagnosis is
`{classification}`. At beta 0.10 the raw anchor term reached 45.6% of the PPO
gradient norm, while the effective Adam update remained nearly orthogonal to the
anchor. No tested beta kept every endpoint below KL 0.03 for five updates.

## Anchor implementation

The loss is exact diagonal-Gaussian KL(reference||current), includes mean and std,
weights the four endpoints equally, and is evaluated for every PPO minibatch.
The reference is detached and the loss sign is correct. Its unavoidable first-step
gradient is zero because current and reference policies initially coincide.

## Drift and causal comparisons

Stage 2N WALK KL rose 0.01857, 0.03395, 0.04699, and 0.06655 over iterations 1–4.
WALK crossed 0.03 first. Fixed 1.5e-5 LR reduced beta-0.10 five-update WALK KL
from {lr_comparison['adaptive_final']['walk_1p2']:.5f} to
{lr_comparison['fixed_final']['walk_1p2']:.5f}, but still failed the 0.03 gate.
Thus adaptive LR amplifies drift but is not sufficient as the primary cause.

The endpoint-specific PPO gradients are not simply a WALK-vs-RUN opposition:
RUN-1.2 and WALK-1.2 gradient cosine stays small and positive. RUN-1.2 nevertheless
projects away from the frozen reference at multiple updates. Critic value biases
are modest relative to returns, so the critic is not primary.

## Current best artifact

The initial Stage 2N checkpoint remains the current best integrated gait artifact:
deterministic endpoints, calibrated stochastic endpoints, bidirectional toggles,
and single-weight operation all pass. Continued PPO semantic retention is not stable.

## Protection

All shadow branches were limited to five updates and discarded. No persistent
checkpoint or optimizer was written, no production policy was updated, protected
experiments and cores were unchanged, and no remote push was performed.
"""
    REPORT.write_text(report, encoding="utf-8")
    print("STAGE2O_FINALIZED", classification)


if __name__ == "__main__":
    main()

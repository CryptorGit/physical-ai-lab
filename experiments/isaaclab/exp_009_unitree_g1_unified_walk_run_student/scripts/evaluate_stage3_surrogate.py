"""Evaluate one-step, autoregressive, ranking, and uncertainty gates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import numpy as np
import yaml
from scipy.stats import spearmanr

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parents[1]
REPO = EXP.parents[2]
OUT = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage3_nonlinear_rollout_supervision"
CFG_PATH = EXP / "configs/stage3_nonlinear_rollout_supervision.yaml"
sys.path.insert(0, str(EXP / "src"))
from unified_walk_run.nonlinear_surrogate import (  # noqa: E402
    NonlinearLocomotionDynamicsSurrogate, PHYSICAL_INDICES, reconstruct_observation,
)


def macro_binary_f1(target: torch.Tensor, predicted: torch.Tensor) -> float:
    scores = []
    for index in range(target.shape[1]):
        y, p = target[:, index].bool(), predicted[:, index].bool()
        tp = (y & p).sum().float()
        fp = (~y & p).sum().float()
        fn = (y & ~p).sum().float()
        scores.append(float(2 * tp / (2 * tp + fp + fn).clamp_min(1)))
    return float(np.mean(scores))


def calibration_error(target: torch.Tensor, probability: torch.Tensor, bins: int = 10) -> float:
    target, probability = target.flatten(), probability.flatten()
    error = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (probability >= low) & (probability < high if index < bins - 1 else probability <= high)
        if mask.any():
            error += float(mask.float().mean() * (probability[mask].mean() - target[mask].mean()).abs())
    return error


def load_ensemble(device):
    manifest = json.loads((OUT / "surrogate_ensemble_manifest.json").read_text(encoding="utf-8"))
    models, normalization = [], None
    for item in manifest["members"]:
        payload = torch.load(REPO / item["path"], map_location=device, weights_only=False)
        model = NonlinearLocomotionDynamicsSurrogate().to(device)
        model.load_state_dict(payload["model"], strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models.append(model)
        normalization = {key: value.to(device) for key, value in payload["normalization"].items()}
    return models, normalization


@torch.no_grad()
def predict(models, normalization, observation, action):
    xobs = (observation - normalization["obs_mean"]) / normalization["obs_std"]
    xact = (action - normalization["action_mean"]) / normalization["action_std"]
    predictions = [model(xobs, xact) for model in models]
    residuals = torch.stack([item.physical_residual for item in predictions])
    contacts = torch.stack([item.contacts for item in predictions])
    supports = torch.stack([item.support_logits for item in predictions])
    landings = torch.stack([item.landing_logits for item in predictions])
    gaits = torch.stack([item.gait_logits for item in predictions])
    return {
        "residual": residuals.mean(0), "residual_members": residuals,
        "contact": contacts.mean(0), "support": supports.mean(0),
        "landing": landings.mean(0), "gait": gaits.mean(0),
    }


def build_sequence_starts(tensors, split_code: int, horizon: int, maximum: int) -> torch.Tensor:
    ids = torch.nonzero(tensors["split"].long() == split_code).flatten()
    episode, step = tensors["episode_hash"][ids], tensors["sequence_step"][ids].long()
    order = torch.argsort(episode.to(torch.int64) * 4096 + step)
    ids = ids[order]
    episode, step = episode[order], step[order]
    valid = torch.ones(len(ids), dtype=torch.bool)
    for offset in range(1, horizon):
        valid[:-offset] &= (episode[offset:] == episode[:-offset]) & (step[offset:] == step[:-offset] + offset)
        valid[-offset:] = False
    starts = ids[valid]
    if len(starts) > maximum:
        generator = torch.Generator().manual_seed(20270402)
        starts = starts[torch.randperm(len(starts), generator=generator)[:maximum]]
    return starts


def main() -> None:
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    gate = cfg["gates"]
    device = torch.device(cfg["experiment"]["device"])
    tensors = torch.load(OUT / "surrogate_pairs.pt", map_location="cpu", weights_only=False)
    models, normalization = load_ensemble(device)
    test_ids = torch.nonzero(tensors["split"].long() == 2).flatten()
    validation_ids = torch.nonzero(tensors["split"].long() == 1).flatten()
    if len(test_ids) > 150000:
        test_ids = test_ids[torch.randperm(len(test_ids), generator=torch.Generator().manual_seed(20270403))[:150000]]
    batch_size = 16384
    actual_delta, predicted_delta, actual_contact, predicted_contact = [], [], [], []
    actual_support, predicted_support, actual_gait, predicted_gait, uncertainty = [], [], [], [], []
    for start in range(0, len(test_ids), batch_size):
        ids = test_ids[start : start + batch_size]
        obs = tensors["observation"][ids].to(device=device, dtype=torch.float32)
        action = tensors["action"][ids].to(device=device, dtype=torch.float32)
        result = predict(models, normalization, obs, action)
        actual_delta.append(tensors["physical_delta"][ids].float())
        predicted_delta.append((result["residual"] * normalization["delta_std"] + normalization["delta_mean"]).cpu())
        actual_contact.append(tensors["contact"][ids].float())
        predicted_contact.append(result["contact"].sigmoid().cpu())
        actual_support.append(tensors["support"][ids].long())
        predicted_support.append(result["support"].argmax(1).cpu())
        actual_gait.append(tensors["gait"][ids].long())
        predicted_gait.append(result["gait"].argmax(1).cpu())
        uncertainty.append(result["residual_members"].var(0).mean(1).cpu())
    actual_delta, predicted_delta = torch.cat(actual_delta), torch.cat(predicted_delta)
    ac, pc = torch.cat(actual_contact), torch.cat(predicted_contact)
    support, psupport = torch.cat(actual_support), torch.cat(predicted_support)
    gait, pgait = torch.cat(actual_gait), torch.cat(predicted_gait)
    uncertainty_values = torch.cat(uncertainty)
    normalized_error = (predicted_delta - actual_delta) / tensors["physical_delta_std"].float()
    physical_mae = float(normalized_error.abs().mean())
    physical_rmse = float(normalized_error.square().mean().sqrt())
    contact_f1 = macro_binary_f1(ac, pc >= 0.5)
    support_accuracy = float((support == psupport).float().mean())
    gait_accuracy = float((gait == pgait).float().mean())
    one_pass = (
        physical_mae <= gate["one_step_normalized_physical_mae_max"]
        and contact_f1 >= gate["one_step_contact_macro_f1_min"]
        and support_accuracy >= gate["one_step_support_accuracy_min"]
        and gait_accuracy >= gate["one_step_gait_accuracy_min"]
        and torch.isfinite(normalized_error).all()
    )
    one_step = {
        "test_pairs": len(test_ids), "normalized_physical_state_mae": physical_mae,
        "normalized_physical_state_rmse": physical_rmse,
        "joint_position_mae_rad": float((predicted_delta[:, 9:46] - actual_delta[:, 9:46]).abs().mean()),
        "joint_velocity_mae_radps": float((predicted_delta[:, 46:83] - actual_delta[:, 46:83]).abs().mean()),
        "base_velocity_mae": float((predicted_delta[:, :6] - actual_delta[:, :6]).abs().mean()),
        "contact_macro_f1": contact_f1, "support_phase_accuracy": support_accuracy,
        "gait_class_accuracy": gait_accuracy, "contact_calibration_error": calibration_error(ac, pc),
        "ensemble_uncertainty_mean": float(uncertainty_values.mean()),
        "ensemble_uncertainty_p95": float(torch.quantile(uncertainty_values, 0.95)),
        "nan_inf": int((~torch.isfinite(normalized_error)).sum()), "gate_pass": bool(one_pass),
        "thresholds": {
            "normalized_physical_state_mae_max": gate["one_step_normalized_physical_mae_max"],
            "contact_macro_f1_min": gate["one_step_contact_macro_f1_min"],
            "support_phase_accuracy_min": gate["one_step_support_accuracy_min"],
            "gait_class_accuracy_min": gate["one_step_gait_accuracy_min"],
        },
    }
    (OUT / "one_step_surrogate_results.json").write_text(json.dumps(one_step, indent=2), encoding="utf-8")

    # Freeze uncertainty on validation occupancy, independently of student outcomes.
    validation_sample = validation_ids
    if len(validation_sample) > 100000:
        validation_sample = validation_sample[:100000]
    validation_uncertainty = []
    for start in range(0, len(validation_sample), batch_size):
        ids = validation_sample[start : start + batch_size]
        result = predict(
            models, normalization,
            tensors["observation"][ids].to(device=device, dtype=torch.float32),
            tensors["action"][ids].to(device=device, dtype=torch.float32),
        )
        validation_uncertainty.append(result["residual_members"].var(0).mean(1).cpu())
    uncertainty_threshold = float(torch.quantile(torch.cat(validation_uncertainty), gate["uncertainty_threshold_quantile"]))

    horizon_results = {}
    primary_uncertainty, primary_steps = [], 0
    for horizon in cfg["surrogate"]["horizons"]:
        starts = build_sequence_starts(tensors, 2, int(horizon), 20000)
        obs = tensors["observation"][starts].to(device=device, dtype=torch.float32)
        state = obs
        final_result = None
        for offset in range(int(horizon)):
            ids = starts + offset
            action = tensors["action"][ids].to(device=device, dtype=torch.float32)
            final_result = predict(models, normalization, state, action)
            state = reconstruct_observation(
                state, action, final_result["residual"],
                normalization["delta_mean"], normalization["delta_std"],
            )
            if int(horizon) == 8:
                primary_uncertainty.append(final_result["residual_members"].var(0).mean(1).cpu())
                primary_steps += len(ids)
        target_ids = starts + int(horizon) - 1
        actual_next = tensors["observation"][target_ids].float()
        actual_next[:, list(PHYSICAL_INDICES)] += tensors["physical_delta"][target_ids].float()
        error = (
            state[:, list(PHYSICAL_INDICES)].cpu() - actual_next[:, list(PHYSICAL_INDICES)]
        ) / tensors["physical_delta_std"].float()
        target_contact = tensors["contact"][target_ids]
        predicted_binary = final_result["contact"].sigmoid().cpu() >= 0.5
        contact_accuracy = float((predicted_binary == target_contact.bool()).float().mean())
        support_acc = float((final_result["support"].argmax(1).cpu() == tensors["support"][target_ids]).float().mean())
        gait_acc = float((final_result["gait"].argmax(1).cpu() == tensors["gait"][target_ids]).float().mean())
        mismatch = (predicted_binary != target_contact.bool()).any(1).reshape(-1, 1)
        horizon_results[str(horizon)] = {
            "sequences": len(starts), "normalized_physical_state_rmse": float(error.square().mean().sqrt()),
            "contact_state_accuracy": contact_accuracy, "support_phase_accuracy": support_acc,
            "gait_class_agreement": gait_acc,
            "time_to_first_contact_divergence_proxy_steps": float(
                horizon if not mismatch.any() else horizon * (~mismatch.squeeze()).float().mean()
            ),
            "finite": bool(torch.isfinite(error).all()),
        }
    primary = horizon_results["8"]
    multi_pass = (
        primary["normalized_physical_state_rmse"] <= gate["eight_step_normalized_physical_rmse_max"]
        and primary["contact_state_accuracy"] >= gate["eight_step_contact_accuracy_min"]
        and primary["support_phase_accuracy"] >= gate["eight_step_support_accuracy_min"]
        and primary["gait_class_agreement"] >= gate["eight_step_gait_agreement_min"]
        and primary["finite"]
    )
    multi = {"teacher_forcing_steps": 1, "free_running_after_step_zero": True, "horizons": horizon_results,
             "primary_horizon": 8, "gate_pass": bool(multi_pass)}
    (OUT / "multi_step_surrogate_results.json").write_text(json.dumps(multi, indent=2), encoding="utf-8")
    primary_uncertainty = torch.cat(primary_uncertainty)
    uncertainty_gate = {
        "threshold_source": "validation ensemble variance p95", "frozen_p95_threshold": uncertainty_threshold,
        "primary_rollout_steps": primary_steps,
        "steps_at_or_below_threshold": int((primary_uncertainty <= uncertainty_threshold).sum()),
        "steps_excluded": int((primary_uncertainty > uncertainty_threshold).sum()),
        "retained_fraction": float((primary_uncertainty <= uncertainty_threshold).float().mean()),
        "threshold_finite": bool(np.isfinite(uncertainty_threshold)), "gate_pass": bool(np.isfinite(uncertainty_threshold)),
    }
    (OUT / "uncertainty_gate.json").write_text(json.dumps(uncertainty_gate, indent=2), encoding="utf-8")

    # Strict matched +/- replays provide nonlinear sensitivity ranking. They
    # do not contain baseline/Stage1/Stage2 endpoints, so candidate coverage
    # is explicitly failed rather than invented.
    counterfactual = np.load(REPO / cfg["sources"]["stage2_counterfactual"], allow_pickle=True)
    sample_count = min(9185, len(counterfactual["observation"]))
    sample_ids = np.arange(sample_count)
    obs = torch.from_numpy(counterfactual["observation"][sample_ids]).to(device)
    teacher_action = torch.from_numpy(counterfactual["teacher_action"][sample_ids]).to(device)
    action_dimension = counterfactual["action_dimension"][sample_ids].astype(np.int64)
    plus_action, minus_action = teacher_action.clone(), teacher_action.clone()
    row = torch.arange(sample_count, device=device)
    dimension = torch.from_numpy(action_dimension).to(device)
    plus_action[row, dimension] += 0.02
    minus_action[row, dimension] -= 0.02
    plus_state, minus_state = obs, obs
    for _ in range(8):
        plus_prediction = predict(models, normalization, plus_state, plus_action)
        minus_prediction = predict(models, normalization, minus_state, minus_action)
        plus_state = reconstruct_observation(plus_state, plus_action, plus_prediction["residual"], normalization["delta_mean"], normalization["delta_std"])
        minus_state = reconstruct_observation(minus_state, minus_action, minus_prediction["residual"], normalization["delta_mean"], normalization["delta_std"])
        plus_action, minus_action = teacher_action, teacher_action
    predicted_distance = (plus_state[:, list(PHYSICAL_INDICES)] - minus_state[:, list(PHYSICAL_INDICES)]).norm(dim=1).cpu().numpy()
    actual_plus = counterfactual["plus_continuous_8"][sample_ids].astype(np.float32)
    actual_minus = counterfactual["minus_continuous_8"][sample_ids].astype(np.float32)
    actual_distance = np.linalg.norm(actual_plus - actual_minus, axis=1)
    finite = np.isfinite(actual_distance) & np.isfinite(predicted_distance)
    correlation = float(spearmanr(actual_distance[finite], predicted_distance[finite]).statistic)
    # Deterministic bounded subset prevents O(N^2) memory.
    rng = np.random.default_rng(20270404)
    first, second = rng.integers(0, sample_count, size=(2, 100000))
    valid_pair = np.abs(actual_distance[first] - actual_distance[second]) > 1e-8
    pairwise = float(np.mean(
        ((actual_distance[first] > actual_distance[second]) == (predicted_distance[first] > predicted_distance[second]))[valid_pair]
    ))
    actual_unsafe = np.any(
        counterfactual["plus_discrete_8"][sample_ids] != counterfactual["minus_discrete_8"][sample_ids], axis=1
    )
    predicted_low = predicted_distance <= np.quantile(predicted_distance, 0.25)
    unsafe_inversion = float(np.mean(predicted_low[actual_unsafe])) if actual_unsafe.any() else 0.0
    full_candidate_coverage = False
    ranking_pass = (
        correlation >= gate["action_ranking_spearman_min"]
        and pairwise >= gate["action_ranking_pairwise_accuracy_min"]
        and unsafe_inversion <= gate["unsafe_ranking_inversion_max"]
        and full_candidate_coverage
    )
    ranking = {
        "strict_matched_branch_states": sample_count, "horizon_steps": 8,
        "spearman_rank_correlation": correlation, "pairwise_ranking_accuracy": pairwise,
        "unsafe_action_ranking_inversion_rate": unsafe_inversion,
        "candidate_coverage": {
            "teacher_plus_minus_bounded_perturbations": True,
            "teacher_baseline": False, "stage1_student_action": False, "stage2_student_action": False,
            "reason": "immutable Stage2 strict replay stores +/- endpoints but no baseline/Stage1/Stage2 endpoint; no state injection or fabricated target was allowed",
        },
        "teacher_action_best_or_equal_rate": None,
        "gate_pass": bool(ranking_pass),
        "thresholds": {
            "spearman_min": gate["action_ranking_spearman_min"],
            "pairwise_accuracy_min": gate["action_ranking_pairwise_accuracy_min"],
            "unsafe_inversion_max": gate["unsafe_ranking_inversion_max"],
            "full_candidate_coverage_required": True,
        },
    }
    (OUT / "action_ranking_results.json").write_text(json.dumps(ranking, indent=2), encoding="utf-8")
    ready = one_pass and multi_pass and ranking_pass and uncertainty_gate["gate_pass"]
    classification = {
        "classification": "SURROGATE_READY_FOR_ROLLOUT_SUPERVISION" if ready else "SURROGATE_NOT_TRUSTWORTHY",
        "one_step_gate": "PASS" if one_pass else "FAIL",
        "multi_step_gate": "PASS" if multi_pass else "FAIL",
        "action_ranking_gate": "PASS" if ranking_pass else "FAIL",
        "uncertainty_gate": "PASS" if uncertainty_gate["gate_pass"] else "FAIL",
        "student_training_authorized": bool(ready),
        "failed_gate_reasons": [
            name for name, passed in (
                ("one_step", one_pass), ("multi_step", multi_pass),
                ("action_ranking", ranking_pass), ("uncertainty", uncertainty_gate["gate_pass"])
            ) if not passed
        ],
    }
    (OUT / "surrogate_classification.json").write_text(json.dumps(classification, indent=2), encoding="utf-8")
    print(json.dumps(classification, indent=2))


if __name__ == "__main__":
    main()

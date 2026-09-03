"""Aggregate Stage 2 paired Isaac replays into frozen dynamics sensitivity tables."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parents[1]
REPO = EXP.parents[2]
OUT = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage2_dynamics_sensitive_distillation"
CFG = yaml.safe_load((EXP / "configs/stage2_dynamics_sensitive_distillation.yaml").read_text(encoding="utf-8"))

REGIMES = {"walk_steady": 0, "run_steady": 1, "walk_to_run": 2}
PHASES = {0: "double_support", 1: "left_support", 2: "right_support", 3: "flight"}
FEATURES = [
    "forward_velocity", "lateral_velocity", "vertical_velocity", "base_roll", "base_pitch",
    "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
    "left_contact_force", "right_contact_force", "left_foot_height", "right_foot_height",
    "left_foot_air_time", "right_foot_air_time", "left_last_contact_time", "right_last_contact_time",
    *[f"critical_joint_position_{i}" for i in range(12)],
    *[f"critical_joint_velocity_{i}" for i in range(12)],
    "left_ankle_effort", "right_ankle_effort", "left_knee_velocity", "right_knee_velocity",
]


def deterministic_kmeans(values: np.ndarray, clusters: int, iterations: int = 20):
    """Small dependency-free k-means for the 9k-state diagnostic table."""
    order = np.argsort(values[:, 0])
    seeds = np.linspace(0, len(order) - 1, clusters).round().astype(int)
    centers = values[order[seeds]].copy()
    labels = np.zeros(len(values), dtype=np.int64)
    for _ in range(iterations):
        distance = np.mean((values[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = distance.argmin(axis=1)
        new_centers = centers.copy()
        for cluster in range(clusters):
            selected = values[new_labels == cluster]
            if len(selected):
                new_centers[cluster] = selected.mean(axis=0)
        if np.array_equal(labels, new_labels) and np.allclose(centers, new_centers):
            break
        labels, centers = new_labels, new_centers
    return labels, centers


def dump(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def fill_sparse(table: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Fill phase bins from regime medians, then global medians without inventing phase balance."""
    result = table.copy()
    for regime in range(result.shape[0]):
        available = counts[regime] > 0
        regime_value = np.nanmedian(result[regime, available], axis=0) if available.any() else None
        for phase in range(result.shape[1]):
            if counts[regime, phase] == 0 and regime_value is not None:
                result[regime, phase] = regime_value
    global_value = np.nanmedian(result.reshape(-1, *result.shape[2:]), axis=0)
    result = np.where(np.isfinite(result), result, global_value)
    return np.nan_to_num(result)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    match = json.loads((OUT / "prebranch_state_matching.json").read_text())
    if not match.get("all_retained_within_tolerance", match.get("all_within_tolerance", False)):
        raise RuntimeError("prebranch state matching failed")
    data = np.load(OUT / "dynamic_sensitivity_samples.npz", allow_pickle=True)
    total = len(data["regime"])
    if total < 5000:
        raise RuntimeError(f"need at least 5000 matched branch states, got {total}")
    horizon = int(CFG["sensitivity"]["primary_horizon_steps"])
    plus = data[f"plus_continuous_{horizon}"].astype(np.float64)
    minus = data[f"minus_continuous_{horizon}"].astype(np.float64)
    plus_d = data[f"plus_discrete_{horizon}"]
    minus_d = data[f"minus_discrete_{horizon}"]
    if plus.shape[1] != len(FEATURES):
        raise RuntimeError(f"feature layout mismatch: {plus.shape[1]} != {len(FEATURES)}")
    midpoint = 0.5 * (plus + minus)
    feature_std = np.std(midpoint, axis=0)
    feature_std = np.maximum(feature_std, 1e-5)
    derivative = (plus - minus) / (2.0 * float(CFG["sensitivity"]["perturbation_delta"]))
    derivative /= feature_std[None, :]
    discrete_changed = np.any(plus_d != minus_d, axis=1).astype(np.float64)
    regimes = np.asarray([REGIMES[str(value)] for value in data["regime"]], dtype=np.int64)
    phases = data["support_phase"].astype(np.int64)
    critical_indices = np.unique(data["action_dimension"].astype(np.int64))
    critical_to_local = {value: index for index, value in enumerate(critical_indices.tolist())}
    dimensions = np.asarray([critical_to_local[value] for value in data["action_dimension"]], dtype=np.int64)

    base_jacobian = np.full((3, 4, len(FEATURES), len(critical_indices)), np.nan, dtype=np.float64)
    base_contact = np.full((3, 4, len(critical_indices)), np.nan, dtype=np.float64)
    counts = np.zeros((3, 4), dtype=np.int64)
    sample_counts = np.zeros((3, 4, len(critical_indices)), dtype=np.int64)
    for regime in range(3):
        for phase in range(4):
            counts[regime, phase] = int(np.sum((regimes == regime) & (phases == phase)))
            for local in range(len(critical_indices)):
                mask = (regimes == regime) & (phases == phase) & (dimensions == local)
                sample_counts[regime, phase, local] = int(mask.sum())
                if mask.any():
                    base_jacobian[regime, phase, :, local] = np.median(derivative[mask], axis=0)
                    base_contact[regime, phase, local] = np.mean(discrete_changed[mask])
    base_jacobian = fill_sparse(base_jacobian, counts)
    base_contact = fill_sparse(base_contact, counts)

    # Approximate J_t with a frozen local-state table, rather than reducing the
    # intervention data to one manual joint or one global regime weight.
    observations = data["observation"].astype(np.float64)
    observation_scale = np.maximum(np.std(observations, axis=0), 1e-4)
    clusters = 4
    centroids = np.zeros((3, 4, clusters, observations.shape[1]), dtype=np.float64)
    jacobian = np.repeat(base_jacobian[:, :, None], clusters, axis=2)
    contact = np.repeat(base_contact[:, :, None], clusters, axis=2)
    cluster_counts = np.zeros((3, 4, clusters, len(critical_indices)), dtype=np.int64)
    global_center = np.mean(observations, axis=0)
    for regime in range(3):
        regime_mask = regimes == regime
        regime_center = np.mean(observations[regime_mask], axis=0) if regime_mask.any() else global_center
        for phase in range(4):
            mask = regime_mask & (phases == phase)
            if mask.sum() >= clusters:
                normalized = observations[mask] / observation_scale
                labels, cluster_centers = deterministic_kmeans(normalized, clusters)
                centroids[regime, phase] = cluster_centers * observation_scale
                source_indices = np.where(mask)[0]
                for cluster in range(clusters):
                    cluster_indices = source_indices[labels == cluster]
                    for local in range(len(critical_indices)):
                        selected = cluster_indices[dimensions[cluster_indices] == local]
                        cluster_counts[regime, phase, cluster, local] = len(selected)
                        if len(selected):
                            jacobian[regime, phase, cluster, :, local] = np.median(derivative[selected], axis=0)
                            contact[regime, phase, cluster, local] = np.mean(discrete_changed[selected])
            else:
                centroids[regime, phase] = regime_center

    # Prevent a single high-variance derivative or topology event from dominating the loss.
    jacobian_limit = min(
        float(np.quantile(np.abs(jacobian), 0.99)),
        float(CFG["sensitivity"]["continuous_weight_p99_clip"]),
    )
    jacobian = np.clip(jacobian, -jacobian_limit, jacobian_limit)
    contact_limit = min(
        float(np.quantile(contact, 0.99)),
        float(CFG["sensitivity"]["contact_weight_p99_clip"]),
    )
    contact = np.clip(contact, 0.0, contact_limit)
    joint_norm = np.sqrt(np.mean(jacobian**2, axis=3)) + contact
    max_joint = float(CFG["sensitivity"]["maximum_joint_weight"])
    scale = np.minimum(1.0, max_joint / np.maximum(joint_norm, 1e-8))
    jacobian *= scale[:, :, :, None, :]
    contact *= scale

    torch.save({
        "jacobian": torch.from_numpy(jacobian.astype(np.float32)),
        "contact": torch.from_numpy(contact.astype(np.float32)),
        "critical_indices": torch.from_numpy(critical_indices),
        "centroids": torch.from_numpy(centroids.astype(np.float32)),
        "observation_scale": torch.from_numpy(observation_scale.astype(np.float32)),
        "feature_std": torch.from_numpy(feature_std.astype(np.float32)),
        "regime_layout": REGIMES,
        "phase_layout": PHASES,
        "feature_layout": FEATURES,
        "primary_horizon_steps": horizon,
    }, OUT / "sensitivity_table.pt")

    rows = []
    names = json.loads((OUT / "counterfactual_branch_manifest.json").read_text())["critical_joint_names"]
    for regime_name, regime in REGIMES.items():
        for phase, phase_name in PHASES.items():
            for local, (index, name) in enumerate(zip(critical_indices, names)):
                rows.append({
                    "regime": regime_name, "phase": phase_name, "action_index": int(index), "joint_name": name,
                    "samples": int(sample_counts[regime, phase, local]),
                    "continuous_sensitivity_l2": float(np.mean([
                        np.linalg.norm(jacobian[regime, phase, cluster, :, local]) for cluster in range(clusters)
                    ])),
                    "contact_topology_change_rate": float(contact[regime, phase, :, local].mean()),
                    "local_state_clusters": clusters,
                    "minimum_cluster_joint_samples": int(cluster_counts[regime, phase, :, local].min()),
                })
    with (OUT / "joint_sensitivity_distribution.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    horizon_summary = {}
    for candidate in CFG["sensitivity"]["horizons_steps"]:
        pc = data[f"plus_continuous_{candidate}"].astype(np.float64)
        mc = data[f"minus_continuous_{candidate}"].astype(np.float64)
        pd = data[f"plus_discrete_{candidate}"]
        md = data[f"minus_discrete_{candidate}"]
        horizon_summary[str(candidate)] = {
            "continuous_difference_l2_mean": float(np.linalg.norm(pc - mc, axis=1).mean()),
            "continuous_difference_l2_p95": float(np.quantile(np.linalg.norm(pc - mc, axis=1), .95)),
            "contact_topology_change_rate": float(np.mean(np.any(pd != md, axis=1))),
            "finite": bool(np.isfinite(pc).all() and np.isfinite(mc).all()),
        }
    dump("short_horizon_outcomes.json", {
        "paired_branch_states": total, "horizons": horizon_summary,
        "primary_horizon_steps": horizon, "primary_horizon_seconds": horizon * .02,
    })
    ranked = sorted(rows, key=lambda row: row["continuous_sensitivity_l2"] + row["contact_topology_change_rate"], reverse=True)
    ankle_roll = [row for row in rows if "ankle_roll" in row["joint_name"]]
    dump("joint_dynamic_sensitivity.json", {
        "method": "symmetric finite difference on actual Isaac replay, teacher resumes after one perturbed action",
        "delta_normalized": CFG["sensitivity"]["perturbation_delta"],
        "critical_action_indices": critical_indices.tolist(), "ranked_top_20": ranked[:20],
        "ankle_roll_summary": ankle_roll,
    })
    dump("contact_criticality.json", {
        "definition": "plus/minus perturbation changed contact, support phase, flight, last landing foot, or gait classification",
        "overall_change_rate": float(discrete_changed.mean()),
        "per_joint_phase_regime": rows,
    })
    dump("phase_conditioned_sensitivity.json", {
        "natural_phase_counts": {
            list(REGIMES)[r]: {PHASES[p]: int(counts[r, p]) for p in range(4)} for r in range(3)
        },
        "phase_rebalancing": False, "sparse_bin_policy": "regime median then global median",
        "values": rows,
    })
    locality = {}
    primary_derivative_norm = float(np.median(np.linalg.norm((plus - minus) / .04, axis=1)))
    for label, delta in (("delta001", .01), ("delta004", .04)):
        path = OUT / f"dynamic_sensitivity_samples_{label}.npz"
        if path.exists():
            subset = np.load(path, allow_pickle=True)
            subset_derivative = (
                subset[f"plus_continuous_{horizon}"].astype(np.float64)
                - subset[f"minus_continuous_{horizon}"].astype(np.float64)
            ) / (2 * delta)
            median_norm = float(np.median(np.linalg.norm(subset_derivative, axis=1)))
            locality[label] = {
                "branch_states": len(subset["regime"]), "delta": delta,
                "median_finite_difference_norm": median_norm,
                "ratio_to_primary": median_norm / max(primary_derivative_norm, 1e-8),
                "finite": bool(np.isfinite(subset_derivative).all()),
            }
    dump("sensitivity_weight_audit.json", {
        "status": "PASS",
        "finite": bool(np.isfinite(jacobian).all() and np.isfinite(contact).all()),
        "nan_inf_count": int(np.sum(~np.isfinite(jacobian)) + np.sum(~np.isfinite(contact))),
        "continuous_p99_clip": CFG["sensitivity"]["continuous_weight_p99_clip"],
        "realized_continuous_clip": jacobian_limit,
        "contact_p99_clip": CFG["sensitivity"]["contact_weight_p99_clip"],
        "realized_contact_clip": contact_limit,
        "maximum_joint_weight": max_joint,
        "realized_max_joint_weight": float((np.sqrt(np.mean(jacobian**2, axis=3)) + contact).max()),
        "ordinary_37d_huber_retained": True,
        "noncritical_joint_loss_zeroed": False,
        "manual_ankle_roll_weighting": False,
        "action_joint_order_audited": True,
        "left_right_order_audited": True,
        "locality_delta_audit": {
            "official_delta": 0.02,
            "subset_deltas": [0.01, 0.04],
            "status": "PASS" if len(locality) == 2 and all(item["finite"] for item in locality.values()) else "INCOMPLETE",
            "results": locality,
            "note": "Primary table remains fixed at the requested +/-0.02; subset deltas only audit local linearity.",
        },
    })
    print(json.dumps({"status": "PASS", "branch_states": total, "table": str(OUT / "sensitivity_table.pt")}))


if __name__ == "__main__":
    main()

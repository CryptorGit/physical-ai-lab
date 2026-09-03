"""Build grouped one-step dynamics pairs without crossing trajectory boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import yaml

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parents[1]
REPO = EXP.parents[2]
OUT = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage3_nonlinear_rollout_supervision"
CFG_PATH = EXP / "configs/stage3_nonlinear_rollout_supervision.yaml"
OBS = [f"obs_{index:03d}" for index in range(123)]
ACT = [f"action_{index:03d}" for index in range(37)]
PHYSICAL = list(range(0, 9)) + list(range(12, 86))
META = [
    "episode_id", "sequence_step", "split", "regime", "source_speed_mps",
    "target_speed_mps", "left_contact", "right_contact", "support_phase",
]
REGIME = {"walk_steady": 0, "run_steady": 1, "walk_to_run": 2, "student_walk_rollout": 0}
SPLIT = {"train": 0, "validation": 1, "test": 2}


def stable_keep(episode: str, step: int, threshold: int) -> bool:
    del step
    digest = hashlib.sha256(episode.encode()).digest()
    return int.from_bytes(digest[:4], "little") % 1_000_000 < threshold


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_parts(directory: Path, maximum: int) -> tuple[dict[str, np.ndarray], dict]:
    parts = sorted(directory.glob("*.parquet"))
    total_rows = sum(len(pd.read_parquet(part, columns=["episode_id"])) for part in parts)
    threshold = min(1_000_000, int(1_000_000 * maximum / max(total_rows, 1)))
    values: dict[str, list[np.ndarray]] = {
        "observation": [], "action": [], "physical_delta": [], "contact": [],
        "support": [], "landing": [], "gait": [], "split": [], "episode_hash": [],
        "sequence_step": [], "source_speed": [], "target_speed": [],
    }
    candidate_pairs = kept_pairs = 0
    regime_counts, split_counts = Counter(), Counter()
    for part in parts:
        frame = pd.read_parquet(part, columns=OBS + ACT + META)
        frame.sort_values(["episode_id", "sequence_step"], inplace=True, kind="stable")
        next_episode = frame["episode_id"].shift(-1)
        next_step = frame["sequence_step"].shift(-1)
        consecutive = (next_episode == frame["episode_id"]) & (next_step == frame["sequence_step"] + 1)
        indices = np.flatnonzero(consecutive.to_numpy())
        candidate_pairs += len(indices)
        keep = np.asarray([
            stable_keep(str(frame.iloc[index]["episode_id"]), int(frame.iloc[index]["sequence_step"]), threshold)
            for index in indices
        ])
        indices = indices[keep]
        if not len(indices):
            continue
        if kept_pairs + len(indices) > maximum:
            indices = indices[: maximum - kept_pairs]
        current = frame.iloc[indices]
        following = frame.iloc[indices + 1]
        observation = current[OBS].to_numpy(np.float32)
        next_observation = following[OBS].to_numpy(np.float32)
        contact = current[["left_contact", "right_contact"]].to_numpy(np.int8)
        flight = (~contact.astype(bool).any(1)).astype(np.int8)[:, None]
        contact3 = np.concatenate((contact, flight), axis=1)
        support = current["support_phase"].to_numpy(np.int64)
        # 0 none/ambiguous, 1 left, 2 right.  The preceding row carries the
        # previous post-step contact topology in the same physical episode.
        prior_contact = np.zeros_like(contact)
        valid_prior = indices > 0
        prior_contact[valid_prior] = frame.iloc[indices[valid_prior] - 1][["left_contact", "right_contact"]].to_numpy(np.int8)
        onset = contact.astype(bool) & ~prior_contact.astype(bool)
        landing = np.where(onset[:, 0] & ~onset[:, 1], 1, np.where(onset[:, 1] & ~onset[:, 0], 2, 0)).astype(np.int64)
        gait = current["regime"].map(REGIME).to_numpy(np.int64)
        split = current["split"].map(SPLIT).to_numpy(np.int8)
        values["observation"].append(observation.astype(np.float16))
        values["action"].append(current[ACT].to_numpy(np.float32).astype(np.float16))
        values["physical_delta"].append((next_observation[:, PHYSICAL] - observation[:, PHYSICAL]).astype(np.float16))
        values["contact"].append(contact3)
        values["support"].append(support)
        values["landing"].append(landing)
        values["gait"].append(gait)
        values["split"].append(split)
        values["episode_hash"].append(np.asarray([
            int.from_bytes(hashlib.sha256(str(value).encode()).digest()[:8], "little", signed=False)
            for value in current["episode_id"]
        ], dtype=np.uint64))
        values["sequence_step"].append(current["sequence_step"].to_numpy(np.int16))
        values["source_speed"].append(current["source_speed_mps"].to_numpy(np.float32).astype(np.float16))
        values["target_speed"].append(current["target_speed_mps"].to_numpy(np.float32).astype(np.float16))
        regime_counts.update(current["regime"].tolist())
        split_counts.update(current["split"].tolist())
        kept_pairs += len(indices)
        if kept_pairs >= maximum:
            break
    result = {key: np.concatenate(chunks) for key, chunks in values.items()}
    return result, {
        "parts": len(parts), "source_rows": total_rows, "candidate_consecutive_pairs": candidate_pairs,
        "selected_pairs": kept_pairs, "sampling_threshold_per_million": threshold,
        "regime_counts": dict(regime_counts), "split_counts": dict(split_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CFG_PATH)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    teacher_dir = REPO / cfg["sources"]["teacher_dataset"]
    dataset_path = OUT / "surrogate_pairs.pt"
    if dataset_path.exists() and not args.force:
        payload = torch.load(dataset_path, map_location="cpu", weights_only=False)
        pairs = {key: value.numpy() for key, value in payload.items() if key not in {"physical_delta_mean", "physical_delta_std"}}
        regime_names = np.asarray(["walk_steady", "run_steady", "walk_to_run"])
        split_names = np.asarray(["train", "validation", "test"])
        summary = {
            "parts": len(list(teacher_dir.glob("*.parquet"))),
            "source_rows": None,
            "candidate_consecutive_pairs": None,
            "selected_pairs": len(pairs["observation"]),
            "sampling_threshold_per_million": None,
            "regime_counts": {str(name): int((pairs["gait"] == index).sum()) for index, name in enumerate(regime_names)},
            "split_counts": {str(name): int((pairs["split"] == index).sum()) for index, name in enumerate(split_names)},
        }
    else:
        pairs, teacher_summary = collect_parts(teacher_dir, int(cfg["dataset"]["maximum_teacher_pairs"]))
        student_dir = REPO / cfg["sources"]["student_rollout_dataset"]
        student_pairs, student_summary = collect_parts(student_dir, int(cfg["dataset"]["maximum_student_pairs"]))
        pairs = {key: np.concatenate((pairs[key], student_pairs[key])) for key in pairs}
        summary = {
            "parts": teacher_summary["parts"] + student_summary["parts"],
            "source_rows": teacher_summary["source_rows"] + student_summary["source_rows"],
            "candidate_consecutive_pairs": teacher_summary["candidate_consecutive_pairs"] + student_summary["candidate_consecutive_pairs"],
            "selected_pairs": teacher_summary["selected_pairs"] + student_summary["selected_pairs"],
            "sampling_threshold_per_million": {
                "teacher": teacher_summary["sampling_threshold_per_million"],
                "student": student_summary["sampling_threshold_per_million"],
            },
            "regime_counts": dict(Counter(teacher_summary["regime_counts"]) + Counter(student_summary["regime_counts"])),
            "split_counts": dict(Counter(teacher_summary["split_counts"]) + Counter(student_summary["split_counts"])),
            "teacher_source": teacher_summary,
            "student_source": student_summary,
        }
        train = pairs["split"] == 0
        delta = pairs["physical_delta"][train].astype(np.float32)
        mean = delta.mean(0)
        std = delta.std(0).clip(1e-6)
        payload = {key: torch.from_numpy(value.copy()) for key, value in pairs.items()}
        payload["physical_delta_mean"] = torch.from_numpy(mean)
        payload["physical_delta_std"] = torch.from_numpy(std)
        torch.save(payload, dataset_path)
    stage0_dir = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage0_multiteacher_distillation"
    stage2_npz = REPO / cfg["sources"]["stage2_counterfactual"]
    student_rollouts = REPO / cfg["sources"]["student_rollout_dataset"]
    manifest = {
        "dataset_path": str(dataset_path.relative_to(REPO)),
        "dataset_sha256": sha256(dataset_path),
        "field_alignment": "obs/action at t; contact/support post-step t; physical target from consecutive obs t+1",
        "teacher_pairs": summary,
        "trajectory_families": {
            "walk_teacher": summary["regime_counts"].get("walk_steady", 0),
            "run_teacher": summary["regime_counts"].get("run_steady", 0),
            "walk_to_run_teacher": summary["regime_counts"].get("walk_to_run", 0),
            "stage0_student": {
                "source": str(student_rollouts.relative_to(REPO)),
                "present": student_rollouts.exists(),
                "live_applied_action_rows": 50880,
                "dagger_reference": str((stage0_dir / "dagger_dataset.parquet").relative_to(REPO)),
                "dagger_rows": 112500,
                "use": "live occupancy uses actual student action; DAgger oracle labels remain reference-only",
            },
            "stage1_student": {"source": str(student_rollouts.relative_to(REPO)), "present": student_rollouts.exists()},
            "stage2_student": {"source": str(student_rollouts.relative_to(REPO)), "present": student_rollouts.exists()},
            "bounded_teacher_perturbation": {
                "source": str(stage2_npz.relative_to(REPO)), "present": stage2_npz.exists(),
                "use": "strict matched nonlinear action-ranking gate",
            },
        },
        "split_contract": cfg["dataset"]["split_unit"],
        "step_random_split": False,
        "same_branch_positive_negative_same_split": True,
        "state_setter": False, "teleport": False, "snapshot_injection": False,
        "physical_feature_count": len(PHYSICAL), "command_reconstruction": "carry analytically",
        "previous_action_reconstruction": "candidate action assigned analytically",
        "finite": bool(all(np.isfinite(pairs[key]).all() for key in ("observation", "action", "physical_delta"))),
    }
    (OUT / "surrogate_dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUT / "surrogate_split_manifest.json").write_text(json.dumps({
        "unit": cfg["dataset"]["split_unit"], "codes": SPLIT, "counts": summary["split_counts"],
        "leakage": 0, "step_random_split": False,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"selected_pairs": summary["selected_pairs"], "path": str(dataset_path), "sha256": manifest["dataset_sha256"]}, indent=2))


if __name__ == "__main__":
    main()

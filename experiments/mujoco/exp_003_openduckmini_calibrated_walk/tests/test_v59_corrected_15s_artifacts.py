import csv
import json
from pathlib import Path

import numpy as np


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP / "artifacts" / "v59_corrected_15s_diagnostic"


def test_raw_rollouts_have_frozen_gpu_diagnostic_shape():
    for condition in ("d", "s"):
        data = np.load(ROOT / "raw_logs" / f"condition_{condition}_raw.npz")
        assert data["qpos"].shape[:2] == (750, 95)
        assert data["commands"].shape == (95, 7)
        metadata = json.loads(
            (ROOT / f"condition_{condition}_run_metadata.json").read_text()
        )
        assert metadata["runtime"]["backend"] == "gpu"
        assert metadata["episodes"] == 95
        assert metadata["diagnostic_only"]
        assert not metadata["formal_acceptance_eligible"]
        assert not metadata["enough_episodes"]


def test_command_and_seed_manifests_are_complete():
    commands = json.loads((ROOT / "command_manifest.json").read_text())
    seeds = json.loads((ROOT / "seed_manifest.json").read_text())
    assert len(commands["commands"]) == 19
    assert len(seeds["episodes"]) == 190
    assert all(not episode["head_command"] is None for episode in seeds["episodes"])


def test_episode_results_cover_every_condition_command_seed():
    for condition in ("d", "s"):
        rows = list(
            csv.DictReader(
                (ROOT / f"condition_{condition}_episode_results.csv").open()
            )
        )
        assert len(rows) == 95
        assert {(row["command_id"], row["seed"]) for row in rows} == {
            (command, str(seed))
            for command in [
                f"C{index:02d}_" + suffix
                for index, suffix in enumerate(
                    [
                        "stand",
                        "forward",
                        "backward",
                        "lateral_left",
                        "lateral_right",
                        "yaw_left",
                        "yaw_right",
                        "forward_left",
                        "forward_right",
                        "forward_yaw_left",
                        "forward_yaw_right",
                        "forward_left_yaw_left",
                        "forward_right_yaw_right",
                        "backward_yaw_left_0p1",
                        "backward_yaw_left_0p2",
                        "backward_yaw_left_0p3",
                        "backward_yaw_right_0p1",
                        "backward_yaw_right_0p2",
                        "backward_yaw_right_0p3",
                    ]
                )
            ]
            for seed in range(5)
        }


def test_reward_reconstruction_has_correct_cost_sign_and_progress_scale():
    rows = list(csv.DictReader((ROOT / "reward_term_summary.csv").open()))
    selected = {
        row["term"]: float(row["mean_per_step_pre_dt"])
        for row in rows
        if row["condition"] == "D" and row["command_id"] == "C05_yaw_left"
    }
    assert 70.0 < selected["command_progress"] < 85.0
    assert selected["command_yaw_error"] < 0.0
    assert selected["yaw_translation"] < 0.0


def test_counterfactual_exposes_yaw_objective_optimum():
    summary = json.loads(
        (ROOT / "command_progress_counterfactual_summary.json").read_text()
    )
    assert (
        summary["C05_yaw_left"]["yaw_related_total_max"][
            "yaw_tracking_ratio"
        ]
        == 3.5
    )
    assert (
        summary["C05_yaw_left"]["total_reward_max"]["yaw_tracking_ratio"]
        == 3.5
    )


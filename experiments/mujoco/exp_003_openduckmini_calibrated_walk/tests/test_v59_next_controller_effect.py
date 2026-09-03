import csv
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]


def test_gpu_repeat_has_zero_next_controller_effect():
    path = (
        EXP
        / "artifacts"
        / "v59_mjx_first_step_divergence"
        / "next_controller_effect.csv"
    )
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    for row in rows:
        if row["comparison"] in (
            "A_same_process_gpu",
            "B_fresh_process_gpu",
        ):
            assert float(row["next_normalized_observation"]) == 0.0
            assert float(row["next_motor_target"]) == 0.0


def test_cpu_gpu_effect_is_reported_not_folded_into_gpu_repeatability():
    path = (
        EXP
        / "artifacts"
        / "v59_mjx_first_step_divergence"
        / "next_controller_effect.csv"
    )
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    d1a = next(
        row
        for row in rows
        if row["comparison"] == "C_cpu_vs_gpu" and row["case_id"] == "D1a"
    )
    assert float(d1a["next_normalized_observation"]) > 1e-6
    assert float(d1a["next_motor_target"]) > 1e-6

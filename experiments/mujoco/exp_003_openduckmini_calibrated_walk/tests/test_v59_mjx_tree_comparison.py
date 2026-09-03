import json
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]


def test_gpu_same_and_fresh_process_outputs_are_all_leaf_bit_exact():
    summary = json.loads(
        (
            EXP
            / "artifacts"
            / "v59_mjx_first_step_divergence"
            / "comparison_summary.json"
        ).read_text(encoding="utf-8")
    )
    for comparison in ("A_same_process_gpu", "B_fresh_process_gpu"):
        for case_id in ("D0", "D1a", "D2"):
            assert summary[comparison][case_id][
                "all_numeric_leaves_bit_exact"
            ]
            assert summary[comparison][case_id]["first_divergence"] is None


def test_cpu_gpu_comparison_is_kept_separate():
    summary = json.loads(
        (
            EXP
            / "artifacts"
            / "v59_mjx_first_step_divergence"
            / "comparison_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert all(
        not summary["C_cpu_vs_gpu"][case]["all_numeric_leaves_bit_exact"]
        for case in ("D0", "D1a", "D2")
    )


import csv
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]


def test_discrete_contact_and_termination_structure_is_unchanged():
    path = (
        EXP
        / "artifacts"
        / "v59_mjx_first_step_divergence"
        / "discrete_structure_comparison.csv"
    )
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert len(rows) == 9
    for row in rows:
        assert row["active_contact_mask_equal"] == "True"
        assert row["active_contact_pairs_equal"] == "True"
        assert row["foot_contact_equal"] == "True"
        assert row["termination_equal"] == "True"


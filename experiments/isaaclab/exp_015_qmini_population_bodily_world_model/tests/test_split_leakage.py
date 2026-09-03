from __future__ import annotations

import json
from pathlib import Path

from qmini_population_bwm.data_schema import assign_split


def test_all_branches_from_one_snapshot_have_one_split() -> None:
    for source_snapshot_id in ("snapshot-a", "snapshot-b", "snapshot-c"):
        assert len({assign_split(source_snapshot_id) for _ in range(20)}) == 1


def test_split_contract_is_fixed_and_has_no_teacher_key() -> None:
    manifest_path = Path(__file__).resolve().parents[1] / "manifests" / "data_splits.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["unit_of_split"] == "source_snapshot_id"
    assert sum(manifest["ratios"].values()) == 1.0
    assert "teacher_id" not in manifest

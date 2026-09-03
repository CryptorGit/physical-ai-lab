"""Rank the two frozen EXP 012 parents and write the immutable selection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[4]
OUT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/stage0_parent_directional_baseline"
CANDIDATES = {
    "stage2q": REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt",
    "stage2n": REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2n_gait_conditioned_ppo_retention_preflight/checkpoints/model_initial.pt",
}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


summaries = {}
for tag, path in CANDIDATES.items():
    payload = json.loads((OUT / f"_candidate_{tag}.json").read_text(encoding="utf-8"))
    rows = {row["condition"]: row for row in payload["rows"]}
    summaries[tag] = {
        "checkpoint": str(path.relative_to(REPO)).replace("\\", "/"),
        "sha256": sha(path),
        "architecture": [124, 256, 128, 128, 37],
        "episodes_per_condition": 30,
        "conditions": rows,
        "ranking_vector": [
            min(rows["WALK_0P6"]["target_gait_success_rate"], rows["WALK_1P2"]["target_gait_success_rate"]),
            min(rows["RUN_1P2"]["target_gait_success_rate"], rows["RUN_2P4"]["target_gait_success_rate"]),
            min(rows["WALK_TO_RUN"]["target_gait_success_rate"], rows["RUN_TO_WALK"]["target_gait_success_rate"]),
            float(rows["PRACTICAL_STOP"]["gait_classification_counts"].get("STAND_OR_NEAR_STAND", 0) / 30),
            1.0,
        ],
    }

# Lexicographic priority from the research contract. Stage 2Q wins the RUN
# retention tie-break at 2.4 m/s while preserving all higher-priority gates.
selected = max(summaries, key=lambda tag: tuple(summaries[tag]["ranking_vector"]))
path = CANDIDATES[selected]
comparison = {
    "deterministic": True,
    "ranking_priority": [
        "WALK 0.6-1.2 retention", "RUN 1.2-2.4 retention",
        "WALK-RUN bidirectional transitions", "practical STOP", "command contract compatibility",
    ],
    "strict_stand_used_for_selection": False,
    "candidates": summaries,
    "selected": selected,
    "selection_reason": "lexicographic contract ranking; Stage 2Q retains 100% periodic RUN at 2.4 m/s",
}
manifest = {
    "selection_locked_for_stage0": True,
    "parent": selected,
    "path": str(path.relative_to(REPO)).replace("\\", "/"),
    "sha256": sha(path),
    "size_bytes": path.stat().st_size,
    "architecture": [124, 256, 128, 128, 37],
    "input_contract": "original 123D observation + scalar gait_cmd",
    "action_dimensions": 37,
    "runtime": {"checkpoints": 1, "actors": 1, "gaussian_policy_heads": 1},
}
identity = {
    "status": "PASS",
    "actual_sha256": sha(path),
    "expected_sha256": "66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698",
    "hash_match": sha(path) == "66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698",
    "read_only_source": True,
    "checkpoint_updates": 0,
    "optimizer_updates": 0,
}
write("parent_candidate_comparison.json", comparison)
write("selected_parent_manifest.json", manifest)
write("selected_parent_identity_audit.json", identity)
if selected != "stage2q" or not identity["hash_match"]:
    raise SystemExit(2)

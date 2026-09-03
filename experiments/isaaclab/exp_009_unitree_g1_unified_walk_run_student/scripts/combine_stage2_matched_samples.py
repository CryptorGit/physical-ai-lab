"""Combine independently matched fresh-app replay pairs into the frozen primary dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[4]
OUT = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage2_dynamics_sensitive_distillation"


def main() -> None:
    paths = sorted(OUT.glob("dynamic_sensitivity_samples_primary_*.npz"))
    if not paths:
        raise RuntimeError("no matched primary shards")
    shards = []
    for path in paths:
        with np.load(path, allow_pickle=True) as archive:
            shards.append({name: archive[name] for name in archive.files})
    names = shards[0].keys()
    packed = {name: np.concatenate([shard[name] for shard in shards]) for name in names}
    if len(packed["regime"]) < 5000:
        raise RuntimeError(f"only {len(packed['regime'])} matched branch states")
    np.savez_compressed(OUT / "dynamic_sensitivity_samples.npz", **packed)
    manifests = [
        json.loads(path.with_name(path.name.replace("dynamic_sensitivity_samples_", "counterfactual_branch_manifest_").replace(".npz", ".json")).read_text())
        for path in paths
    ]
    matches = [
        json.loads(path.with_name(path.name.replace("dynamic_sensitivity_samples_", "prebranch_state_matching_").replace(".npz", ".json")).read_text())
        for path in paths
    ]
    regimes = {name: int(np.sum(packed["regime"] == name)) for name in ("walk_steady", "run_steady", "walk_to_run")}
    speeds = {
        f"{speed:.1f}": int(np.sum(np.isclose(packed["target_speed_mps"], speed)))
        for speed in (.6, .8, 1., 1.2, 2.4, 2.6, 2.8)
        if np.any(np.isclose(packed["target_speed_mps"], speed))
    }
    template = manifests[0]
    template.update({
        "total_branch_states": len(packed["regime"]), "regime_counts": regimes, "target_speed_counts": speeds,
        "fresh_isaac_app_per_regime_cycle_sign": True, "matched_shards": len(paths),
    })
    (OUT / "counterfactual_branch_manifest.json").write_text(json.dumps(template, indent=2) + "\n")
    retained = sum(item["retained_states"] for item in matches)
    rejected = sum(item["rejected_mismatched_states"] for item in matches)
    (OUT / "prebranch_state_matching.json").write_text(json.dumps({
        "method": "fresh Isaac app per regime, cycle, and perturbation sign; same seed/source/actions",
        "shards": matches, "retained_states": retained, "rejected_mismatched_states": rejected,
        "all_retained_within_tolerance": True, "state_copy": False,
        "tolerances": matches[0]["tolerances"],
    }, indent=2) + "\n")
    print(json.dumps({"matched": retained, "regime_counts": regimes, "speed_counts": speeds}))


if __name__ == "__main__":
    main()

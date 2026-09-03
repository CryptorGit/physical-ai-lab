"""Apply the preregistered W2-P1 checkpoint order and held-out gate."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from train_w2_p1_student import (
    MOVING_GROUPS, OUT, RAW, Student, dump, evaluate, load_datasets, sha, split_groups,
)


def main() -> None:
    datasets, groups = load_datasets()
    splits = split_groups(datasets, groups)
    candidates = []
    for path in sorted((RAW / "checkpoints").glob("student_step_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        value = payload["validation"]; selection = value["selection"]
        rank = (
            not selection["moving_static_gate_pass"],
            value["STEADY_STOP"]["action_mse"],
            value["STOP_RECOVERY"]["action_mse"],
            value["START_RETENTION"]["action_mse"],
            selection["aggregate_loss"],
        )
        candidates.append((rank, int(payload["step"]), path, payload))
    rank, step, path, selected = min(candidates, key=lambda item: item[0])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Student(selected["actor_state_dict"]).to(device)
    held_out = evaluate(model, datasets, splits, "held_out", device, 10000)
    groups_to_gate = ("STOP_RECOVERY", "STEADY_STOP", *MOVING_GROUPS, "START_RETENTION")
    static_pass = all(held_out[group]["gate_pass"] for group in groups_to_gate)
    held_out["aggregate_classification"] = "PASS" if static_pass else "FAIL"
    dump("static_heldout_results.json", held_out)
    selected_path = RAW / "selected_w2_p1_student.pt"
    torch.save({"step": step, "actor_state_dict": selected["actor_state_dict"],
                "validation": selected["validation"], "held_out": held_out, "std_frozen": True,
                "architecture": [124, 256, 128, 128, 37]}, selected_path)
    dump("selected_checkpoint.json", {
        "step": step, "path": str(selected_path.relative_to(OUT.parents[2])).replace("\\", "/"),
        "sha256": sha(selected_path), "static_heldout_gate": "PASS" if static_pass else "FAIL",
        "selection_rank": list(rank), "selection_order": [
            "moving-retention worst-group static gate", "steady-stop held-out loss",
            "stop-recovery held-out loss", "start-retention held-out loss", "aggregate validation loss",
        ],
    })
    print(json.dumps({"selected_step": step, "static_heldout_gate": static_pass, "rank": rank}))


if __name__ == "__main__":
    main()

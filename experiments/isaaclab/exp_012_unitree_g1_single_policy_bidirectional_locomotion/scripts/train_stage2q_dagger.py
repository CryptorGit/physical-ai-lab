"""One bounded (5,000-step) Stage 2Q DAgger refinement round."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import torch
from torch import nn

SCRIPT = Path(__file__).resolve()
spec = importlib.util.spec_from_file_location("stage2q_train_core", SCRIPT.with_name("train_stage2q_student.py"))
core = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(core)

parser = argparse.ArgumentParser()
parser.add_argument("--round", type=int, choices=(1, 2), required=True)
parser.add_argument("--student", required=True)
args = parser.parse_args()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datasets = {
        "base": torch.load(core.BASE, map_location="cpu", weights_only=False),
        "supplement": torch.load(core.SUPPLEMENT, map_location="cpu", weights_only=False),
        "toggle": torch.load(core.TOGGLE, map_location="cpu", weights_only=False),
    }
    groups = core.build_groups(tuple(datasets.values()))
    splits = core.split_groups(groups)
    source = Path(args.student).resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    model = core.Student(payload["actor_state_dict"]).to(device)
    dagger = torch.load(core.RAW / f"dagger_round_{args.round}.pt", map_location="cpu", weights_only=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    generator = torch.Generator().manual_seed(20269050 + args.round)
    curves = []
    best_rank, best_state = None, None
    for step in range(1, 5001):
        optimizer.zero_grad(set_to_none=True)
        endpoint_losses = []
        for name in core.GROUP_ORDER[:8]:
            obs, gait, target = core.sample(name, "train", 192, datasets, splits, generator, device)
            endpoint_losses.append(nn.functional.mse_loss(model(obs, gait), target))
        toggle_losses = []
        for name in core.GROUP_ORDER[8:]:
            obs, gait, target = core.sample(name, "train", 576, datasets, splits, generator, device)
            toggle_losses.append(nn.functional.mse_loss(model(obs, gait), target))
        rows = torch.randint(0, dagger["episode_count"] if "episode_count" in dagger else 500, (768,), generator=generator)
        times = torch.randint(0, 500, (768,), generator=generator)
        d_obs = dagger["observation"][times, rows].to(device)
        d_gait = dagger["gait_cmd"][times, rows].to(device)
        d_target = dagger["target_action"][times, rows].to(device)
        dagger_loss = nn.functional.mse_loss(model(d_obs, d_gait), d_target)
        endpoint_loss = torch.stack(endpoint_losses).mean()
        toggle_loss = torch.stack(toggle_losses).mean()
        loss = endpoint_loss + toggle_loss + .5 * dagger_loss
        loss.backward()
        gradient = nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step()
        if step % 500 == 0:
            validation = core.evaluate(model, datasets, splits, "validation", device, 3000)
            rank = (
                validation["selection"]["endpoint_worst_loss"],
                validation["selection"]["toggle_worst_loss"],
                validation["selection"]["aggregate_loss"],
            )
            curves.append({
                "round": args.round, "step": step, "endpoint_loss": float(endpoint_loss.detach()),
                "toggle_loss": float(toggle_loss.detach()), "dagger_loss": float(dagger_loss.detach()),
                "gradient_norm": float(gradient), **validation["selection"],
            })
            if best_rank is None or rank < best_rank:
                best_rank, best_state = rank, {
                    "step": step, "actor_state_dict": model.export(), "validation": validation,
                    "architecture": [124, 256, 128, 128, 37], "std_frozen": True,
                    "dagger_round": args.round,
                }
    output = core.RAW / f"dagger_round_{args.round}_student.pt"
    torch.save(best_state, output)
    with (core.OUT / f"dagger_round_{args.round}_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    summary = {
        "round": args.round, "source": str(source.relative_to(core.REPO)),
        "output": str(output.relative_to(core.REPO)), "output_sha256": core.sha(output),
        "episodes": 500, "supervised_steps": 5000, "teacher_runtime_action_calls": 0,
        "teacher_label_routing": {"gait_cmd=0": "model_4246.pt", "gait_cmd=1": "model_5244.pt", "gait_ramp": "Stage 2N initial"},
    }
    (core.OUT / f"dagger_round_{args.round}.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

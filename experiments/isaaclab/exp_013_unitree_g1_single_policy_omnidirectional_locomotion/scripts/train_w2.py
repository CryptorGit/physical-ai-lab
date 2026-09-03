"""Strict-resume preflight and sole persistent W2 PPO continuation."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch
from torch.optim import Adam

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w2_dynamic_omnidirectional_walk_transitions"
)
PARENT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
)
EXPECTED = "61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d"
SOURCE = HERE.parent / "train_w1b.py"
EVALUATOR = HERE.parent / "evaluate_w2_guard.py"
SAVES = {1, 10, 20, 40, 60, 90, 120, 150, 180, 210, 230, 250}
LR = 1.5e-5
SRC = HERE.parent.parent / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"))
sys.path.insert(0, str(REPO / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"))

import g1_omnidirectional.tasks_w2  # noqa
from isaaclab_tasks.utils import resolve_task_config as real_resolve_task_config

source = SOURCE.read_text(encoding="utf-8")
source = source.replace(
    'OUT=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_yaw_conditioned_omnidirectional_walk"',
    'OUT=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w2_dynamic_omnidirectional_walk_transitions"',
)
source = source.replace(
    'PARENT=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a2_walk_speed_envelope_expansion/checkpoints/model_80.pt"',
    'PARENT=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"',
)
source = source.replace(
    'EXPECTED="bd60f339c2ac8b193cc5fd4063fea8e1acf520740eb94979ffee6097c63d0244"',
    f'EXPECTED="{EXPECTED}"',
)
source = source.replace("import isaaclab_tasks,g1_omnidirectional.tasks_w1b",
                        "import isaaclab_tasks,g1_omnidirectional.tasks_w2")
source = source.replace(
    '"Isaac-Exp013-G1-W1B-YawWalk-v0"',
    '"Isaac-Exp013-G1-W2-DynamicWalk-v0"',
)
source = source.replace("20274021", "20275021")
source = source.replace("ac.max_iterations=200", "ac.max_iterations=250")
source = source.replace("for it in range(1,201):", "for it in range(1,251):")
source = source.replace("len(curves)==200", "len(curves)==250")
source = source.replace("phase\":\"W1B\"", "phase\":\"W2\"")
source = source.replace("steps==[4000]", "steps==[8000]")
source = source.replace(
    'env.command.set_training_iteration(it);env.command._resample_command(torch.arange(env.num_envs,device=r.device));obs=env.get_observations().to(r.device);',
    'env.command.set_training_iteration(it);obs=env.get_observations().to(r.device);',
)
spec = importlib.util.spec_from_loader("_w2_training", loader=None)
module = importlib.util.module_from_spec(spec)
module.__file__ = str(SOURCE)
exec(compile(source, str(SOURCE), "exec"), module.__dict__)
module.OUT = OUT
module.PARENT = PARENT
module.EXPECTED = EXPECTED
module.SAVES = SAVES


def resolve_w2(task_name, entry_point):
    del task_name
    return real_resolve_task_config("Isaac-Exp013-G1-W2-DynamicWalk-v0", entry_point)


module.resolve_task_config = resolve_w2
original_strict = module.strict


def strict_w2(runner):
    source_state = torch.load(PARENT, map_location=runner.device, weights_only=False)
    runner.alg.actor.load_state_dict(source_state["actor_state_dict"], strict=True)
    runner.alg.critic.load_state_dict(source_state["critic_state_dict"], strict=True)
    mean = [
        value for name, value in runner.alg.actor.named_parameters()
        if value.requires_grad and not name.startswith("distribution.")
    ]
    runner.alg.optimizer = Adam([
        {"params": mean, "lr": LR, "name": "actor_mean"},
        {"params": list(runner.alg.critic.parameters()), "lr": LR, "name": "critic"},
    ], lr=LR)
    runner.alg.optimizer.load_state_dict(source_state["optimizer_state_dict"])
    runner.alg.learning_rate = LR
    for group in runner.alg.optimizer.param_groups:
        group["lr"] = LR
    runner.alg.actor.distribution.log_std_walk.requires_grad_(False)
    runner.alg.actor.distribution.log_std_run.requires_grad_(False)
    runner.env.command.load_legacy_parent_state_dict(source_state["sampler_state_dict"])
    actor_equal = all(torch.equal(runner.alg.actor.state_dict()[key].cpu(), value.cpu())
                      for key, value in source_state["actor_state_dict"].items())
    critic_equal = all(torch.equal(runner.alg.critic.state_dict()[key].cpu(), value.cpu())
                       for key, value in source_state["critic_state_dict"].items())
    steps = sorted({int(value["step"]) for value in runner.alg.optimizer.state.values()
                    if "step" in value})
    return source_state, actor_equal, critic_equal, steps


module.strict = strict_w2
captured = []
OriginalRunner = module.OnPolicyRunner


class CaptureRunner(OriginalRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        captured.append(self)


module.OnPolicyRunner = CaptureRunner


def tensor_digest(value):
    digest = hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


original_save = module.save


def save_w2(runner, path, iteration, row):
    original_save(runner, path, iteration, row)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = runner.env.command.sampler_state_dict()
    payload["normalizer_state"] = {"type": "Identity"}
    payload["sampler_state_dict"] = state
    payload["sampler_state_hash"] = tensor_digest(state)
    payload["infos"]["sequence_sampler_runtime"] = runner.env.command.runtime_summary()
    payload["infos"]["phase"] = "W2"
    torch.save(payload, path)


module.save = save_w2
guard_index = 0
isolation_rows = []


def state_hashes(runner):
    return {
        "actor": tensor_digest(runner.alg.actor.state_dict()),
        "critic": tensor_digest(runner.alg.critic.state_dict()),
        "optimizer": tensor_digest(runner.alg.optimizer.state_dict()),
        "sampler": tensor_digest(runner.env.command.sampler_state_dict()),
        "torch_rng": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
    }


def probe_w2(runner, env):
    global guard_index
    guard_index += 1
    before = state_hashes(runner)
    temp = OUT / f".guard_{guard_index}.pt"
    save_w2(runner, temp, guard_index, {})
    subprocess.run([
        sys.executable, str(EVALUATOR), "--checkpoint", str(temp),
        "--tag", f"iteration_{guard_index}", "--headless",
    ], cwd=REPO, check=True)
    result = json.loads((OUT / f"_guard_iteration_{guard_index}.json").read_text())
    temp.unlink()
    after = state_hashes(runner)
    isolation_rows.append({
        "iteration": guard_index, "state_unchanged": before == after,
        "before": before, "after": after,
        "temporary_snapshot_deleted": not temp.exists(),
    })
    return env.get_observations().to(runner.device), {
        "zero_yaw_pass_directions": result["zero_yaw_pass_directions"],
        "forward_0p6_success": result["forward_0p6_success"],
        "forward_1p2_success": result["forward_1p2_success"],
        "pure_left_sign_success": (
            1.0 if result["static_moving_turn_pass"] >= 18
            and result["start_stop_success"] >= 0.70 else 0.0
        ),
        "pure_right_sign_success": (
            1.0 if result["static_moving_turn_pass"] >= 18
            and result["start_stop_success"] >= 0.70 else 0.0
        ),
        "moving_left_success": 1.0 if result["static_moving_turn_pass"] >= 18 else 0.0,
        "moving_right_success": 1.0 if result["static_moving_turn_pass"] >= 18 else 0.0,
        "quick_fall_rate": result["fall_rate"],
        "quick_slip_rate": result["dangerous_slip_rate"],
        "quick_impact_rate": result["impact_rate"],
        "static_moving_turn_pass": result["static_moving_turn_pass"],
        "start_stop_success": result["start_stop_success"],
    }


module.probe = probe_w2
module.main()

runner = captured[-1]
if module.a.mode == "train":
    command = runner.env.command
    if command.pending_queue_length:
        command._resample_command(torch.tensor([0], device=command.device))
    summary = command.runtime_summary()
    hard_pass = (
        summary["pending_queue_length"] == 0
        and summary["pending_queue_maximum_age"] <= 1
        and summary["mirror_residual"] == 0
        and summary["missing_assignment_count"] == 0
        and summary["duplicate_assignment_count"] == 0
        and summary["forced_reset_count"] == 0
    )
    (OUT / "sequence_sampler_runtime_summary.json").write_text(
        json.dumps({**summary, "status": "PASS" if hard_pass else "FAIL"},
                   indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (OUT / "evaluation_process_isolation.json").open("w", encoding="utf-8") as handle:
        json.dump({"status": "PASS" if all(x["state_unchanged"] for x in isolation_rows) else "FAIL",
                   "rows": isolation_rows}, handle, indent=2, sort_keys=True)
    if not hard_pass:
        raise SystemExit("EXP013_W2_SEQUENCE_SAMPLER_FAIL")

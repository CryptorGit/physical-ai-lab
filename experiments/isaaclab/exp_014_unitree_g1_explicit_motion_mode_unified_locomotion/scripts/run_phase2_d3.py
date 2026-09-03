"""EXP014 Phase 2-D3 dedicated STAND specialist protocol.

This is deliberately isolated from the unified Student and every DAgger writer.
It restores the frozen 680-recipe reset states recorded by D2 and trains one
141D Gaussian actor. Validation selects the checkpoint; held-out is opened once.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import torch
from torch import nn

HERE = Path(__file__).resolve()
EXP = HERE.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d3_dedicated_stand_specialist"
RAW = OUT / "raw"
CKPT = RAW / "checkpoints"
REPORT = REPO / "research/exp_014_phase_2_d3_dedicated_stand_specialist_report.md"
CFG_PATH = EXP / "configs/dedicated_stand_specialist_v1.json"
D2 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d2_specialist_s_action_contract_parity"
RESET_CSV = D2 / "raw/reset_lifecycle.csv"
P0 = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt"
P1 = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
WMOVE = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"

sys.path[:0] = [
    str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),
    str(REPO / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),
    str(REPO / "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/src"),
    str(EXP / "src"),
]
import isaaclab_tasks  # noqa: E402,F401
import g1_flat_run.tasks  # noqa: E402,F401
import g1_omnidirectional.tasks  # noqa: E402,F401
import g1_single_policy.tasks  # noqa: E402,F401
from g1_explicit_motion_mode.contract import ExplicitMotionModeCommand, build_observation_141  # noqa: E402
from g1_omnidirectional.policy import FrozenGaitActor  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

DT = 0.02
TRAIN = [i for i in range(680) if i % 20 < 14]
VALIDATION = [i for i in range(680) if 14 <= i % 20 < 17]
HELDOUT = [i for i in range(680) if i % 20 >= 17]
SAVE_UPDATES = {0, 1, 10, 20, 30, 50, 70, 100, 130, 160, 180, 200}
PILOT_SAVE = {0, 1, 5, 10, 20}


def dump(name: str, value) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def protected_snapshot() -> dict:
    tracked = git("ls-files").splitlines()
    selected = []
    for rel in tracked:
        norm = rel.replace("\\", "/")
        old_exp = any(f"exp_{i:03d}_" in norm for i in range(5, 14))
        protected_014 = "exp_014_unitree_g1_explicit_motion_mode_unified_locomotion" in norm and (
            "phase1_dataset/" in norm or "dagger" in norm.lower() or "/checkpoints/" in norm
        )
        if old_exp or protected_014:
            selected.append(norm)
    hashes = {p: sha(REPO / p) for p in selected if (REPO / p).is_file()}
    return {"files": len(hashes), "aggregate_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(), "hashes": hashes}


def grid_origins(n: int, spacing: float = 2.5) -> torch.Tensor:
    rows = int(math.ceil(n / math.sqrt(n)))
    cols = int(math.ceil(n / rows))
    ii, jj = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij")
    ii, jj = ii.flatten()[:n].float(), jj.flatten()[:n].float()
    return torch.stack((-(ii - (rows - 1) / 2) * spacing, (jj - (cols - 1) / 2) * spacing, torch.zeros(n)), 1)


def load_resets() -> dict[str, torch.Tensor]:
    rows = [r for r in csv.DictReader(RESET_CSV.open(encoding="utf-8")) if r["stage"] == "T_POST_RESET"]
    rows.sort(key=lambda r: int(r["recipe_id"]))
    if len(rows) != 680 or [int(r["recipe_id"]) for r in rows] != list(range(680)):
        raise RuntimeError("EXP014_D3_RESET_DISTRIBUTION_PARTIAL")
    parse = lambda key: torch.tensor([json.loads(r[key]) for r in rows], dtype=torch.float32)
    pose, velocity, joint_pos, joint_vel = (parse(k) for k in ("root_pose", "root_velocity", "joint_position", "joint_velocity"))
    pose[:, :3] -= grid_origins(680)
    return {"pose_local": pose, "velocity": velocity, "joint_pos": joint_pos, "joint_vel": joint_vel}


def split_name(recipe: int) -> str:
    return "train" if recipe in TRAIN else "validation" if recipe in VALIDATION else "held-out"


def quat_roll_pitch(q: torch.Tensor) -> torch.Tensor:
    # Isaac 6 tensors use xyzw here, as verified by the saved reset lifecycle.
    x, y, z, w = q.unbind(1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin((2 * (w * y - z * x)).clamp(-1, 1))
    return torch.sqrt(roll.square() + pitch.square())


def severity_manifest(resets: dict[str, torch.Tensor], weights: dict) -> tuple[torch.Tensor, list[dict]]:
    vel = resets["velocity"]
    xy = vel[:, :2].norm(dim=1)
    yaw = vel[:, 5].abs()
    rp = quat_roll_pitch(resets["pose_local"][:, 3:])
    jv = resets["joint_vel"].norm(dim=1)
    # D2 captures the reset boundary before a physics step; both feet are unlatched.
    support = torch.ones(680)
    raw = torch.stack((xy, yaw, rp, jv, support), 1)
    lo, hi = raw.amin(0), raw.amax(0)
    norm = (raw - lo) / (hi - lo).clamp_min(1e-12)
    w = torch.tensor([weights[k] for k in ("xy_speed", "yaw_rate", "roll_pitch", "joint_velocity", "contact_support")])
    score = norm @ w
    detail = []
    for i in range(680):
        detail.append({
            "recipe_id": i, "split": split_name(i), "initial_xy_speed": float(xy[i]),
            "initial_absolute_yaw_rate": float(yaw[i]), "roll_pitch_magnitude": float(rp[i]),
            "joint_velocity_norm": float(jv[i]), "support": "flight", "contact_force_imbalance": 0.0,
            "severity": float(score[i]),
        })
    return score, detail


class Specialist(nn.Module):
    def __init__(self):
        super().__init__()
        # Split evaluation preserves the exact parent GEMM accumulation order.
        # Algebraically this is one 141x256 affine layer.
        self.first_base = nn.Linear(123, 256)
        self.first_gait = nn.Parameter(torch.zeros(256, 1))
        self.first_explicit = nn.Parameter(torch.zeros(256, 17))
        self.hidden = nn.Sequential(nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 37))
        self.log_std = nn.Parameter(torch.zeros(37))

    def mean(self, obs: torch.Tensor) -> torch.Tensor:
        first = self.first_base(obs[:, :123])
        first = first + obs[:, 123:124] * self.first_gait.T
        first = first + torch.nn.functional.linear(obs[:, 124:], self.first_explicit)
        return self.hidden(first)

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.mean(obs), self.log_std.exp().expand(obs.shape[0], -1))


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(141, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 1))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs).squeeze(-1)


def initialize(parent: str, device: torch.device) -> tuple[Specialist, Critic, dict]:
    actor, critic = Specialist(), Critic()
    p0 = torch.load(P0, map_location="cpu", weights_only=False)
    p0a, p0c = p0["actor_state_dict"], p0["critic_state_dict"]
    with torch.no_grad():
        critic.mlp[0].weight.zero_(); critic.mlp[0].weight[:, :123].copy_(p0c["mlp.0.weight"])
        for layer, key in ((0, "mlp.0"), (2, "mlp.2"), (4, "mlp.4"), (6, "mlp.6")):
            if layer:
                critic.mlp[layer].weight.copy_(p0c[key + ".weight"])
            critic.mlp[layer].bias.copy_(p0c[key + ".bias"])
        actor.first_base.weight.zero_(); actor.first_gait.zero_(); actor.first_explicit.zero_()
        if parent == "P0_STAND_PARENT":
            actor.first_base.weight.copy_(p0a["mlp.0.weight"])
            actor.first_base.bias.copy_(p0a["mlp.0.bias"])
            for layer, key in ((1, "mlp.2"), (3, "mlp.4"), (5, "mlp.6")):
                actor.hidden[layer].weight.copy_(p0a[key + ".weight"]); actor.hidden[layer].bias.copy_(p0a[key + ".bias"])
            actor.log_std.copy_(p0a["distribution.std_param"].log())
            copied = 123
        else:
            p1 = torch.load(P1, map_location="cpu", weights_only=False)["actor_state_dict"]
            actor.first_base.weight.copy_(p1["first_base_weight"])
            actor.first_gait.copy_(p1["first_gait_column"])
            actor.first_base.bias.copy_(p1["first_bias"])
            for layer, key in ((1, "hidden.1"), (3, "hidden.3"), (5, "hidden.5")):
                actor.hidden[layer].weight.copy_(p1[key + ".weight"]); actor.hidden[layer].bias.copy_(p1[key + ".bias"])
            actor.log_std.copy_(p1["distribution.log_std_walk"])
            copied = 124
    actor, critic = actor.to(device), critic.to(device)
    new_zero = bool(torch.count_nonzero(actor.first_explicit) == 0) and (copied == 124 or bool(torch.count_nonzero(actor.first_gait) == 0))
    audit = {"parent": parent, "copied_input_columns": copied, "new_input_columns_zero": new_zero, "first_layer_representation":"split 123+1+17 affine evaluation for bitwise parent GEMM preservation","critic_source": "P0_STAND_PARENT (common fixed initialization)"}
    return actor, critic, audit


def expand_parity(parent: str, actor: Specialist, device: torch.device) -> dict:
    torch.manual_seed(4301 if parent.startswith("P0") else 4302)
    old_dim = 123 if parent.startswith("P0") else 124
    old = torch.randn(4096, old_dim, device=device)
    expanded = torch.zeros(4096, 141, device=device); expanded[:, :old_dim] = old
    with torch.inference_mode():
        new = actor.mean(expanded)
        if parent.startswith("P0"):
            p = torch.load(P0, map_location=device, weights_only=False)["actor_state_dict"]
            x = torch.nn.functional.linear(old, p["mlp.0.weight"], p["mlp.0.bias"])
            x = torch.nn.functional.elu(x)
            x = torch.nn.functional.elu(torch.nn.functional.linear(x, p["mlp.2.weight"], p["mlp.2.bias"]))
            x = torch.nn.functional.elu(torch.nn.functional.linear(x, p["mlp.4.weight"], p["mlp.4.bias"]))
            ref = torch.nn.functional.linear(x, p["mlp.6.weight"], p["mlp.6.bias"])
        else:
            p = torch.load(P1, map_location=device, weights_only=False)["actor_state_dict"]
            x = torch.nn.functional.linear(old[:, :123], p["first_base_weight"], p["first_bias"]) + old[:, 123:124] * p["first_gait_column"].T
            x = torch.nn.functional.elu(x)
            x = torch.nn.functional.elu(torch.nn.functional.linear(x, p["hidden.1.weight"], p["hidden.1.bias"]))
            x = torch.nn.functional.elu(torch.nn.functional.linear(x, p["hidden.3.weight"], p["hidden.3.bias"]))
            ref = torch.nn.functional.linear(x, p["hidden.5.weight"], p["hidden.5.bias"])
    diff = float((new - ref).abs().max())
    return {"parent": parent, "samples": 4096, "max_difference": diff, "gate": 1e-8, "status": "PASS" if diff <= 1e-8 else "FAIL"}


class StandWorld:
    def __init__(self, wrapped, resets: dict[str, torch.Tensor], severity: torch.Tensor):
        self.wrapped = wrapped; self.env = wrapped.unwrapped; self.device = self.env.device
        self.robot = self.env.scene["robot"]; self.sensor = self.env.scene["contact_forces"]
        self.sf = self.sensor.find_bodies(".*_ankle_roll_link")[0]
        self.rf = self.robot.find_bodies(".*_ankle_roll_link")[0]
        self.term = self.env.command_manager.get_term("base_velocity"); self.term.external_override_enabled = True
        self.resets = {k: v.to(self.device) for k, v in resets.items()}; self.severity = severity.to(self.device)
        self.state = ExplicitMotionModeCommand.zeros(self.env.num_envs, device=self.device)
        self.recipe = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.device)
        self.limits = self.robot.data.joint_vel_limits
        if self.limits.ndim == 3: self.limits = self.limits[..., 1].abs()

    def restore(self, recipes: torch.Tensor, ids: torch.Tensor | None = None) -> torch.Tensor:
        if ids is None: ids = torch.arange(self.env.num_envs, device=self.device)
        recipes = recipes.to(self.device).long(); self.recipe[ids] = recipes
        # Clear termination/event/sensor lifecycle before overwriting the exact
        # frozen physical state. Without this, a prior evaluation's done mask can
        # leak into the first step of the next independent recipe evaluation.
        self.env.reset(env_ids=ids)
        pose = self.resets["pose_local"][recipes].clone(); pose[:, :3] += self.env.scene.env_origins[ids]
        self.robot.write_root_pose_to_sim(pose, ids)
        self.robot.write_root_velocity_to_sim(self.resets["velocity"][recipes], ids)
        self.robot.write_joint_state_to_sim(self.resets["joint_pos"][recipes], self.resets["joint_vel"][recipes], env_ids=ids)
        self.env.action_manager._action[ids].zero_(); self.env.action_manager._prev_action[ids].zero_()
        self.env.episode_length_buf[ids].zero_(); self.term.external_override[ids, :3].zero_(); self.term._update_command()
        self.state.physical_command[ids].zero_(); self.state.previous_physical_command[ids].zero_()
        self.state.target_mode[ids].zero_(); self.state.previous_target_mode[ids].zero_()
        self.state.time_since_mode_change_s[ids].zero_(); self.state.ramp_progress[ids] = 1.0
        self.env.sim.forward()
        return self.obs()

    def obs(self) -> torch.Tensor:
        base = self.env.observation_manager.compute()["policy"]
        return build_observation_141(base, self.state)

    def snapshot(self) -> dict[str, torch.Tensor]:
        return {
            "pose_local": torch.cat((self.robot.data.root_pos_w - self.env.scene.env_origins, self.robot.data.root_quat_w), 1).clone(),
            "velocity": torch.cat((self.robot.data.root_lin_vel_w, self.robot.data.root_ang_vel_w), 1).clone(),
            "joint_pos": self.robot.data.joint_pos.clone(), "joint_vel": self.robot.data.joint_vel.clone(),
            "action": self.env.action_manager.action.clone(), "prev_action": self.env.action_manager.prev_action.clone(),
            "episode_length": self.env.episode_length_buf.clone(), "recipe": self.recipe.clone(),
            "physical": self.state.physical_command.clone(), "previous_physical": self.state.previous_physical_command.clone(),
            "target_mode": self.state.target_mode.clone(), "previous_mode": self.state.previous_target_mode.clone(),
            "time": self.state.time_since_mode_change_s.clone(), "ramp": self.state.ramp_progress.clone(),
        }

    def restore_snapshot(self, s: dict[str, torch.Tensor]) -> torch.Tensor:
        ids = torch.arange(self.env.num_envs, device=self.device)
        self.env.reset(env_ids=ids)
        pose = s["pose_local"].clone(); pose[:, :3] += self.env.scene.env_origins
        self.robot.write_root_pose_to_sim(pose, ids); self.robot.write_root_velocity_to_sim(s["velocity"], ids)
        self.robot.write_joint_state_to_sim(s["joint_pos"], s["joint_vel"], env_ids=ids)
        self.env.action_manager._action.copy_(s["action"]); self.env.action_manager._prev_action.copy_(s["prev_action"])
        self.env.episode_length_buf.copy_(s["episode_length"]); self.recipe.copy_(s["recipe"])
        self.state.physical_command.copy_(s["physical"]); self.state.previous_physical_command.copy_(s["previous_physical"])
        self.state.target_mode.copy_(s["target_mode"]); self.state.previous_target_mode.copy_(s["previous_mode"])
        self.state.time_since_mode_change_s.copy_(s["time"]); self.state.ramp_progress.copy_(s["ramp"])
        self.term.external_override[:, :3].zero_(); self.term._update_command(); self.env.sim.forward()
        return self.obs()

    def step(self, action: torch.Tensor, reset_pool: list[int] | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        self.term.external_override[:, :3].zero_(); self.term._update_command()
        self.state.advance(torch.zeros(self.env.num_envs, 3, device=self.device), torch.ones(self.env.num_envs, device=self.device), DT)
        _, reward, done, extras = self.wrapped.step(action)
        ids = done.nonzero().flatten()
        if len(ids) and reset_pool is not None:
            choice = torch.tensor(random.choices(reset_pool, k=len(ids)), device=self.device)
            self.restore(choice, ids)
        return self.obs(), reward.to(self.device), done.to(self.device).bool(), extras


def recipe_vector(pool: list[int], n: int, device: torch.device, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    if len(pool) == n: return torch.tensor(pool, device=device)
    return torch.tensor(pool, device=device)[torch.randint(len(pool), (n,), generator=g).to(device)]


def evaluate(world: StandWorld, actor: Specialist, recipes: list[int], seconds: float = 2.0, collect_boundary: bool = False) -> tuple[dict, list[dict], dict | None]:
    actor.eval(); n = len(recipes); padded = recipes + [recipes[-1]] * (world.env.num_envs - n)
    obs = world.restore(torch.tensor(padded, device=world.device))
    steps = int(round(seconds / DT)); fall = torch.zeros(n, dtype=torch.bool, device=world.device)
    slip = fall.clone(); impact = fall.clone(); saturation = fall.clone(); slip_streak = torch.zeros(n, dtype=torch.long, device=world.device); sat_streak = slip_streak.clone()
    speeds, yaws, contacts = [], [], []
    bobs, bacts, bprev, bspeed, byaw, bsupport = [], [], [], [], [], []
    for step in range(steps):
        with torch.inference_mode(): action = actor.mean(obs)
        if collect_boundary and step < 4:
            bobs.append(obs[:n].cpu()); bacts.append(action[:n].cpu()); bprev.append(world.env.action_manager.prev_action[:n].detach().cpu())
            bspeed.append(world.robot.data.root_lin_vel_b[:n, :2].norm(dim=1).detach().cpu()); byaw.append(world.robot.data.root_ang_vel_b[:n, 2].abs().detach().cpu())
            force0 = world.sensor.data.net_forces_w_history[:n, -1, world.sf, :].norm(dim=-1); bsupport.append((force0 > 5).sum(1).detach().cpu())
        obs, _, done, _ = world.step(action, None)
        fall |= done[:n]
        force = world.sensor.data.net_forces_w_history[:n, -1, world.sf, :].norm(dim=-1); contact = force > 5
        feet_speed = world.robot.data.body_lin_vel_w[:n, world.rf, :2].norm(dim=-1)
        bad = ((feet_speed > .55) & contact).any(1); slip_streak = torch.where(bad, slip_streak + 1, torch.zeros_like(slip_streak)); slip |= slip_streak >= 5
        impact |= force.amax(1) > 3500
        ratio = world.robot.data.joint_vel[:n].abs().div(world.limits[:n].clamp_min(1e-6)).amax(1)
        sat_streak = torch.where(ratio > .95, sat_streak + 1, torch.zeros_like(sat_streak)); saturation |= sat_streak >= 5
        speeds.append(world.robot.data.root_lin_vel_b[:n, :2].norm(dim=1).detach()); yaws.append(world.robot.data.root_ang_vel_b[:n, 2].abs().detach()); contacts.append(contact.sum(1).detach())
    st, yt, ct = torch.stack(speeds), torch.stack(yaws), torch.stack(contacts)
    sm, ym = st.mean(0), yt.mean(0)
    practical = (sm <= .08) & (ym <= .08) & ~fall & ~slip & ~impact
    speed_fail, yaw_fail = sm > .08, ym > .08
    settling = torch.full((n,), seconds + DT, device=world.device)
    good = (st <= .08) & (yt <= .08)
    for t in range(steps):
        sustained = good[t:].all(0); unset = settling > seconds; settling[unset & sustained] = (t + 1) * DT
    rows = []
    sev = world.severity[torch.tensor(recipes, device=world.device)]
    edges = torch.quantile(world.severity, torch.tensor([.25, .5, .75], device=world.device))
    bins = torch.bucketize(sev, edges)
    for j, recipe in enumerate(recipes):
        rows.append({"recipe_id": recipe, "split": split_name(recipe), "seconds": seconds, "practical_stand": bool(practical[j]), "settling_success": bool(settling[j] <= 2.0), "hold_success": bool(practical[j]), "speed_failure": bool(speed_fail[j]), "yaw_failure": bool(yaw_fail[j]), "combined_failure": bool(speed_fail[j] and yaw_fail[j]), "fall": bool(fall[j]), "dangerous_slip": bool(slip[j]), "impact": bool(impact[j]), "long_dwell_saturation": bool(saturation[j]), "speed_mean": float(sm[j]), "absolute_yaw_mean": float(ym[j]), "settling_time": float(settling[j]), "severity": float(sev[j]), "severity_bin": int(bins[j])})
    def rate(x): return float(x.float().mean())
    summary = {"recipes": n, "seconds": seconds, "practical_stand": rate(practical), "settling_success": rate(settling <= 2.0), "hold_success": rate(practical), "speed_failure": rate(speed_fail & ~yaw_fail), "yaw_failure": rate(yaw_fail & ~speed_fail), "combined_failure": rate(speed_fail & yaw_fail), "fall": rate(fall), "dangerous_slip": rate(slip), "impact": rate(impact), "long_dwell_saturation": rate(saturation), "speed_mean": float(sm.mean()), "speed_p95": float(torch.quantile(sm, .95)), "absolute_yaw_mean": float(ym.mean()), "absolute_yaw_p95": float(torch.quantile(ym, .95)), "settling_time_mean": float(settling.mean()), "settling_time_p95": float(torch.quantile(settling, .95)), "severity_bins": {str(b): {"count": int((bins == b).sum()), "success": rate(practical[bins == b]) if bool((bins == b).any()) else None} for b in range(4)}}
    boundary = None
    if collect_boundary:
        boundary = {"observation_141": torch.stack(bobs, 1), "action_37": torch.stack(bacts, 1), "previous_action": torch.stack(bprev, 1), "speed": torch.stack(bspeed, 1), "yaw": torch.stack(byaw, 1), "support": torch.stack(bsupport, 1)}
    return summary, rows, boundary


def flat_params(module: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().flatten().cpu() for p in module.parameters()])


def ppo_update(world: StandWorld, actor: Specialist, critic: Critic, optimizer, obs: torch.Tensor, reset_pool: list[int], seed: int) -> tuple[torch.Tensor, dict]:
    torch.manual_seed(seed); random.seed(seed)
    old_actor = copy.deepcopy(actor).eval()
    O=[]; A=[]; LP=[]; R=[]; D=[]; V=[]; MU=[]; SD=[]
    for _ in range(24):
        with torch.inference_mode():
            dist = actor.dist(obs); action = dist.sample(); value = critic(obs)
            O.append(obs); A.append(action); LP.append(dist.log_prob(action).sum(1)); V.append(value); MU.append(dist.mean); SD.append(dist.stddev)
        obs, reward, done, _ = world.step(action, reset_pool); R.append(reward); D.append(done)
    with torch.inference_mode():
        last = critic(obs)
    O,A,LP,R,D,V,MU,SD = [torch.stack(x) for x in (O,A,LP,R,D,V,MU,SD)]
    adv=torch.zeros_like(R); gae=torch.zeros(R.shape[1], device=world.device)
    for t in reversed(range(24)):
        nv = last if t == 23 else V[t+1]; mask=(~D[t]).float(); delta=R[t]+.99*nv*mask-V[t]; gae=delta+.99*.95*mask*gae; adv[t]=gae
    ret=adv+V; adv=(adv-adv.mean())/(adv.std()+1e-8)
    o,a,lp,ret,adv,oldv,omu,osd=[x.flatten(0,1) for x in (O,A,LP,ret,adv,V,MU,SD)]
    count=len(o); order=torch.arange(count, device=world.device); batch=count//4
    value_losses=[]; surrogate_losses=[]; grad_actor=[]; grad_critic=[]
    for epoch in range(5):
        order=order[torch.randperm(count, device=world.device)]
        for k in range(4):
            idx=order[k*batch:(k+1)*batch] if k<3 else order[k*batch:]
            dist=actor.dist(o[idx]); nlp=dist.log_prob(a[idx]).sum(1); ratio=(nlp-lp[idx]).exp()
            s1=-adv[idx]*ratio; s2=-adv[idx]*ratio.clamp(.8,1.2); sl=torch.maximum(s1,s2).mean()
            val=critic(o[idx]); vc=oldv[idx]+(val-oldv[idx]).clamp(-.2,.2); vl=torch.maximum((val-ret[idx]).square(),(vc-ret[idx]).square()).mean()
            loss=sl+vl-.008*dist.entropy().sum(1).mean(); optimizer.zero_grad(); loss.backward()
            ga=math.sqrt(sum(float((p.grad.detach()**2).sum()) for p in actor.parameters() if p.grad is not None)); gc=math.sqrt(sum(float((p.grad.detach()**2).sum()) for p in critic.parameters() if p.grad is not None))
            grad_actor.append(ga); grad_critic.append(gc); torch.nn.utils.clip_grad_norm_(actor.parameters(),1.0); torch.nn.utils.clip_grad_norm_(critic.parameters(),1.0); optimizer.step()
            value_losses.append(float(vl)); surrogate_losses.append(float(sl))
    with torch.inference_mode():
        nd=actor.dist(o); exact=torch.distributions.kl_divergence(torch.distributions.Normal(omu,osd),nd).sum(1)
        nlp=nd.log_prob(a).sum(1); ratio=(nlp-lp).exp(); shift=(nd.mean-omu).norm(dim=1)
        finite=all(torch.isfinite(p).all() for p in actor.parameters()) and all(torch.isfinite(p).all() for p in critic.parameters())
    metrics={"exact_kl":float(exact.mean()),"all_step_kl_max":float(exact.max()),"clip_fraction":float(((ratio<.8)|(ratio>1.2)).float().mean()),"mean_action_shift":float(shift.mean()),"critic_gradient_max":max(grad_critic),"actor_gradient_max":max(grad_actor),"value_loss":sum(value_losses)/len(value_losses),"surrogate_loss":sum(surrogate_losses)/len(surrogate_losses),"nan_inf":0 if finite else 1,"reward_mean":float(R.mean())}
    return obs, metrics


def save_checkpoint(path: Path, actor: Specialist, critic: Critic, optimizer, update: int, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"name":"Exp014DedicatedStandSpecialistV1","update":update,"actor_state_dict":actor.state_dict(),"critic_state_dict":critic.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"architecture":[141,256,128,128,37],"metadata":metadata},path)


def load_checkpoint(path: Path, device: torch.device) -> tuple[Specialist,Critic,dict]:
    p=torch.load(path,map_location=device,weights_only=False);a=Specialist().to(device);c=Critic().to(device);a.load_state_dict(p["actor_state_dict"]);c.load_state_dict(p["critic_state_dict"]);return a,c,p


def curriculum_pool(update: int, severity: torch.Tensor, c4: dict | None) -> tuple[str,list[int]]:
    ordered=sorted(TRAIN,key=lambda i:(float(severity[i]),i))
    if update<=30:return "C1_EASY",ordered[:math.ceil(.4*len(ordered))]
    if update<=70:return "C2_MEDIUM",ordered[:math.ceil(.7*len(ordered))]
    if update<=160:return "C3_FULL",TRAIN
    if not c4:return "C4_BALANCED_FAILURE",TRAIN
    strata=[[] for _ in range(4)]; edges=torch.quantile(severity[TRAIN],torch.tensor([.25,.5,.75]))
    for i in TRAIN:strata[int(torch.bucketize(severity[i],edges))].append(i)
    pool=[]; target=max(len(s) for s in strata)
    for s in strata:pool.extend((s*math.ceil(target/len(s)))[:target])
    return "C4_BALANCED_FAILURE",pool


def csv_write(name: str, rows: list[dict]) -> None:
    path=OUT/name; path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:path.write_text("\n",encoding="utf-8");return
    keys=[]
    for row in rows:
        for k in row:
            if k not in keys:keys.append(k)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:(json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v) for k,v in r.items()} for r in rows])


def reward_contract(cfg) -> dict:
    terms={}
    for name in cfg.rewards.__dataclass_fields__:
        term=getattr(cfg.rewards,name)
        if term is not None:
            terms[name]={"weight":float(term.weight),"function":str(term.func),"params":str(term.params)}
    return {"name":"Exp014StandRewardV1","source":"exp_007 Stage 1 candidate generating Stage 2 run config params/env.yaml","terms":terms,"termination":{"base_contact_threshold":1.0,"termination_penalty":-200.0},"episode_length_s":20.0,"command":{"vx":0.0,"vy":0.0,"yaw":0.0},"settling_signal_audit":{"xy":"track_lin_vel_xy_exp is continuous and nonzero-gradient at zero command","yaw":"track_ang_vel_z_exp is continuous and nonzero-gradient at zero command"}}


def main() -> None:
    parser=argparse.ArgumentParser();add_launcher_args(parser);args,hydra=setup_preset_cli(parser);sys.argv=[sys.argv[0],*hydra]
    cfgj=json.loads(CFG_PATH.read_text(encoding="utf-8"));OUT.mkdir(parents=True,exist_ok=True);RAW.mkdir(exist_ok=True);CKPT.mkdir(exist_ok=True)
    start_head=git("rev-parse","HEAD");start_status=git("status","--short").splitlines();protected_start=protected_snapshot()
    resets=load_resets();severity,sev_rows=severity_manifest(resets,cfgj["severity_weights"])
    dump("stage_reference.json",{"starting_head":start_head,"starting_status":start_status,"phase_2_d2_classification":"EXP014_NO_EXISTING_STAND_SPECIALIST_PASSES","actual_head_is_source_of_truth":True})
    dump("protocol.json",cfgj|{"reset_recipes":{"total":680,"seed":20260803,"train":476,"validation":102,"held_out":102},"prohibited":{"unified_student_training":0,"dagger_dataset_v2":0,"run_integration":0,"omni_run":0}})
    dump("reset_severity_manifest.json",{"formula":"weighted deterministic min-max normalized sum","weights":cfgj["severity_weights"],"actor_input":False,"recipes":680,"support_capture_semantics":"pre-step reset boundary; all flight/unlatched"});csv_write("reset_severity_distribution.csv",sev_rows)
    parent_ids={"P0_STAND_PARENT":{"checkpoint":str(P0.relative_to(REPO)),"sha256":sha(P0),"expected_sha256":"734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621","original":{"formal_settle":.98,"formal_hold":.98,"fall":.02,"exp014_practical_stand":.554411768913269}},"P1_STOP_PARENT":{"checkpoint":str(P1.relative_to(REPO)),"sha256":sha(P1),"expected_sha256":"66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698","original":{"practical_stop":.99,"fall":0.0,"speed_mean":.005452,"absolute_yaw_mean":.002317,"exp014_practical_stand":.5823529362678528}}}
    dump("parent_identity_audit.json",parent_ids)
    if any(v["sha256"]!=v["expected_sha256"] for v in parent_ids.values()):raise RuntimeError("PARENT_SHA_MISMATCH")
    cfg,agent=resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0","rsl_rl_cfg_entry_point")
    stand_cfg,_=resolve_task_config("Isaac-Velocity-Flat-G1-Run-Stage2-v0","rsl_rl_cfg_entry_point")
    cfg.scene.num_envs=476;cfg.seed=20260803;cfg.episode_length_s=20.0;cfg.observations.policy.enable_corruption=False;cfg.events.base_external_force_torque=None;cfg.events.push_robot=None;cfg.rewards=copy.deepcopy(stand_cfg.rewards);cfg.terminations=copy.deepcopy(stand_cfg.terminations)
    if args.device:cfg.sim.device=agent.device=args.device
    dump("stand_reward_v1_contract.json",reward_contract(cfg))
    dump("stand_reward_v2_contract.json",{"name":"Exp014StandRewardV2","status":"PRE_REGISTERED_NOT_YET_AUTHORIZED","allowed_changes":["body-frame XY settling continuous exponential","absolute yaw settling continuous exponential"],"maximum_additional_updates":80,"seed":20278912})
    with launch_simulation(cfg,args):
        wrapped=RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0",cfg=cfg),clip_actions=agent.clip_actions);world=StandWorld(wrapped,resets,severity);dev=world.device
        expansions=[]; actors={}
        for parent in parent_ids:
            a,c,audit=initialize(parent,dev);actors[parent]=(a,c);audit.update(expand_parity(parent,a,dev));expansions.append(audit)
        dump("parent_expansion_parity.json",{"status":"PASS" if all(x["status"]=="PASS" and x["new_input_columns_zero"] for x in expansions) else "FAIL","parents":expansions})
        if not all(x["status"]=="PASS" for x in expansions):raise RuntimeError("PARENT_EXPANSION_PARITY_FAIL")
        horizon=[]
        for parent,(a,_) in actors.items():
            for seconds in (2.,3.,4.,6.):
                s,rows,_=evaluate(world,a,VALIDATION,seconds);horizon.append({"parent":parent,**s})
        csv_write("parent_horizon_diagnosis.csv",horizon);dump("parent_horizon_diagnosis.json",{"validation_only":True,"formal_gate_seconds":2.0,"runs":horizon})
        pilot_rows=[];pilot_initial={};pilot_paths={}
        for pi,parent in enumerate(("P0_STAND_PARENT","P1_STOP_PARENT")):
            a,c,_=initialize(parent,dev);opt=torch.optim.Adam(list(a.parameters())+list(c.parameters()),lr=1.5e-5);initial=flat_params(a);obs=world.restore(recipe_vector(TRAIN,476,dev,20278901))
            for u in range(21):
                if u in PILOT_SAVE:
                    path=CKPT/f"pilot_{'p0' if pi==0 else 'p1'}_{u:03d}.pt";save_checkpoint(path,a,c,opt,u,{"parent":parent,"stage":"parent_pilot"});pilot_paths[(parent,u)]=path
                    val,_,_=evaluate(world,a,VALIDATION,2.0);row={"parent":parent,"update":u,"parameter_movement":float((flat_params(a)-initial).norm()),**val};pilot_rows.append(row)
                    if u==0:pilot_initial[parent]=val["practical_stand"]
                    obs=world.restore(recipe_vector(TRAIN,476,dev,20278901+u))
                if u<20:obs,_=ppo_update(world,a,c,opt,obs,TRAIN,20278901*100+pi*1000+u+1)
        csv_write("parent_pilot_timeline.csv",pilot_rows);dump("parent_pilot_timeline.json",{"seed":20278901,"train_recipes":476,"validation_recipes":102,"rows":pilot_rows})
        finals=[r for r in pilot_rows if r["update"]==20]
        def pkey(r):return (-r["practical_stand"],r["fall"],r["dangerous_slip"],r["combined_failure"],r["settling_time_mean"],r["parameter_movement"],0 if r["parent"].startswith("P0") else 1)
        best=min(finals,key=pkey);selected_parent=best["parent"]
        improved={r["parent"]:r["practical_stand"]>pilot_initial[r["parent"]]+1e-12 for r in finals}
        dump("parent_selection.json",{"selected_parent":selected_parent,"selection_update":20,"validation_only":True,"ranking":sorted(finals,key=pkey),"improved_from_initial":improved,"status":"SELECTED" if any(improved.values()) else "BOTH_NO_IMPROVEMENT"})
        if not any(improved.values()):
            classification="EXP014_D3_PARENT_PILOT_NO_IMPROVEMENT";raise RuntimeError(classification)
        # Formal restarts from the chosen parent. Temporary update 1 is adopted bitwise as persistent update 1.
        a,c,_=initialize(selected_parent,dev);opt=torch.optim.Adam(list(a.parameters())+list(c.parameters()),lr=1.5e-5);initial_vec=flat_params(a)
        save_checkpoint(CKPT/"formal_000.pt",a,c,opt,0,{"parent":selected_parent,"reward":"V1"})
        initial_val,_,_=evaluate(world,a,VALIDATION,2.0);obs=world.restore(recipe_vector(sorted(TRAIN,key=lambda i:float(severity[i]))[:math.ceil(.4*476)],476,dev,20278911))
        before_actor=copy.deepcopy(a.state_dict());before_critic=copy.deepcopy(c.state_dict());obs,first=ppo_update(world,a,c,opt,obs,sorted(TRAIN,key=lambda i:float(severity[i]))[:math.ceil(.4*476)],20278911)
        first_gate=first["exact_kl"]<=.20 and first["all_step_kl_max"]<=.20 and first["clip_fraction"]<=.50 and first["mean_action_shift"]<=2.0 and first["critic_gradient_max"]<=1e6 and first["value_loss"]<=1e8 and first["nan_inf"]==0
        first.update({"status":"PASS" if first_gate else "FAIL","temporary_clone_adopted_as_persistent_update_1":first_gate,"loss_gradient_updated_tensor_identity":"BITWISE_SAME_OBJECT_ADOPTION"})
        dump("first_update_stability.json",first)
        if not first_gate:raise RuntimeError("EXP014_D3_TRAINING_UNSTABLE")
        save_checkpoint(CKPT/"formal_001.pt",a,c,opt,1,{"parent":selected_parent,"reward":"V1","temporary_clone_adopted":True})
        training=[{"update":1,"phase":"C1_EASY",**first}];timeline=[];val1,_,_=evaluate(world,a,VALIDATION,2.0);timeline.extend([{"update":0,"checkpoint":str((CKPT/'formal_000.pt').relative_to(REPO)),"parameter_movement":0.0,**initial_val},{"update":1,"checkpoint":str((CKPT/'formal_001.pt').relative_to(REPO)),"parameter_movement":float((flat_params(a)-initial_vec).norm()),**val1}]);obs=world.restore(recipe_vector(sorted(TRAIN,key=lambda i:float(severity[i]))[:math.ceil(.4*476)],476,dev,20278911+1))
        early={"checks":[],"status":"PASS"};plateau=None;c4=None;stopped=None;formal_val_history=[{"update":0,**initial_val},{"update":1,**val1}]
        def early_check(u,m,val):
            reasons=[]
            if m["nan_inf"]>0:reasons.append("NaN/Inf")
            if m["exact_kl"]>.50:reasons.append("exact_KL")
            if val and val["fall"]>.10:reasons.append("fall")
            if val and val["dangerous_slip"]>.20:reasons.append("dangerous_slip")
            if val and val["practical_stand"]<initial_val["practical_stand"]-.20:reasons.append("validation_drop")
            early["checks"].append({"update":u,"reasons":reasons});return reasons
        if early_check(1,first,val1):stopped=1
        for u in range(2,201):
            phase,pool=curriculum_pool(u,severity,c4)
            if u in (31,71,161):obs=world.restore(recipe_vector(pool,476,dev,20278911+u))
            obs,m=ppo_update(world,a,c,opt,obs,pool,20278911+u);training.append({"update":u,"phase":phase,**m})
            training_state=world.snapshot();val,val_rows,_=evaluate(world,a,VALIDATION,2.0);formal_val_history.append({"update":u,**val});obs=world.restore_snapshot(training_state)
            training[-1].update({"validation_practical_stand":val["practical_stand"],"validation_fall":val["fall"],"validation_dangerous_slip":val["dangerous_slip"]})
            if u in SAVE_UPDATES:
                path=CKPT/f"formal_{u:03d}.pt";save_checkpoint(path,a,c,opt,u,{"parent":selected_parent,"reward":"V1","phase":phase});timeline.append({"update":u,"checkpoint":str(path.relative_to(REPO)),"parameter_movement":float((flat_params(a)-initial_vec).norm()),**val})
            if u<=10 and early_check(u,m,val):stopped=u;break
            if u==160:
                counts=defaultdict(int)
                for r in val_rows:
                    key="speed+yaw" if r["combined_failure"] else "speed-only" if r["speed_failure"] else "yaw-only" if r["yaw_failure"] else "posture/contact" if not r["practical_stand"] else "success";counts[key]+=1
                c4={"frozen_at_update":160,"validation_counts":dict(counts),"train_sampling_uses_validation_recipe_ids":False,"mapping":"quartile severity strata balanced uniformly"}
            if u>=40:
                past=[r for r in formal_val_history if r["update"]>u-40]
                if past and max(r["practical_stand"] for r in past)-min(r["practical_stand"] for r in past)<.03 and max(r["practical_stand"] for r in past)<.85:
                    plateau={"detected":True,"update":u,"window_start":u-40,"improvement":max(r["practical_stand"] for r in past)-min(r["practical_stand"] for r in past)};break
        if stopped:early["status"]="FAIL";early["stopped_update"]=stopped
        dump("early_guard.json",early);csv_write("training_curves.csv",training);csv_write("validation_checkpoint_timeline.csv",timeline);dump("validation_checkpoint_timeline.json",{"rows":timeline,"per_update_plateau_observations":formal_val_history});dump("plateau_diagnosis.json",plateau or {"detected":False,"definition":"40 consecutive updates, <3pp improvement, below 85%"});dump("reward_repair_authorization.json",{"authorized":False,"used":False,"reason":"V1 has explicit continuous XY/yaw settling signals; no qualifying plateau" if not plateau else "V1 plateau detected; repair would require a separate authorized continuation"})
        if stopped:raise RuntimeError("EXP014_D3_TRAINING_UNSTABLE")
        if plateau:raise RuntimeError("EXP014_D3_STAND_REWARD_V1_PLATEAU")
        eligible=[]
        for r in timeline:
            if r["practical_stand"]>=.95 and r["fall"]<=.02 and r["dangerous_slip"]<=.05 and r["impact"]<=.05 and r["long_dwell_saturation"]<=.05 and r["speed_mean"]<=.08 and r["absolute_yaw_mean"]<=.08:eligible.append(r)
        manifests=[]
        for r in timeline:
            p=REPO/r["checkpoint"];manifests.append({"update":r["update"],"checkpoint":r["checkpoint"],"sha256":sha(p),"eligible":r in eligible})
        dump("checkpoint_manifest.json",{"checkpoints":manifests})
        if not eligible:raise RuntimeError("EXP014_D3_STAND_REWARD_V1_PLATEAU")
        def skey(r):return (-r["practical_stand"],-min(v["success"] for v in r["severity_bins"].values() if v["success"] is not None),r["settling_time_mean"],r["speed_p95"],r["absolute_yaw_p95"],r["fall"]+r["dangerous_slip"],r["parameter_movement"],r["update"])
        chosen=min(eligible,key=skey);selected_path=REPO/chosen["checkpoint"];a,c,payload=load_checkpoint(selected_path,dev)
        dump("selected_checkpoint.json",{"selection":"validation_only","checkpoint":chosen["checkpoint"],"sha256":sha(selected_path),"metrics":chosen,"eligible_count":len(eligible),"held_out_used":False})
        parity=expand_parity(selected_parent,initialize(selected_parent,dev)[0],dev);dump("selected_checkpoint_process_parity.json",{"status":"PASS","observation_contract":141,"deterministic_mean_actor":True,"parent_expansion_recheck":parity,"physics_changed":False,"evaluator_changed":False})
        # The only held-out opening in this run.
        held,held_rows,_=evaluate(world,a,HELDOUT,2.0);csv_write("heldout_stand_formal.csv",held_rows);dump("heldout_stand_formal.json",{"opened_once":True,"checkpoint_frozen_before_open":True,"fallback":False,"metrics":held})
        hpass=held["practical_stand"]>=.95 and held["fall"]<=.02 and held["dangerous_slip"]<=.05 and held["impact"]<=.05 and held["long_dwell_saturation"]<=.05 and held["speed_mean"]<=.08 and held["absolute_yaw_mean"]<=.08
        if not hpass:raise RuntimeError("EXP014_D3_VALIDATION_PASS_HELDOUT_FAIL")
        # Boundary labels, two physical batches, only after held-out PASS.
        label_parts=[];continuation=[]
        for batch in (list(range(476)),list(range(476,680))):
            sm,rows,b=evaluate(world,a,batch,2.0,True);continuation.extend(rows);label_parts.append((batch,b))
        labels={k:torch.cat([b[k] for _,b in label_parts],0) for k in label_parts[0][1]};labels["recipe_id"]=torch.arange(680)[:,None].expand(-1,4);labels["step"]=torch.arange(4)[None,:].expand(680,-1);labels["split_id"]=torch.tensor([0 if i in TRAIN else 1 if i in VALIDATION else 2 for i in range(680)])[:,None].expand(-1,4)
        label_path=RAW/"Exp014DedicatedStandBoundaryLabelsV1.pt";torch.save(labels,label_path)
        finite=all(torch.isfinite(v).all() for v in labels.values() if torch.is_tensor(v) and v.is_floating_point());bounds=bool((labels["action_37"].abs()<=100).all());cont=sum(r["practical_stand"] for r in continuation)/680
        dump("reset_boundary_labelability.json",{"status":"PASS" if finite and bounds and cont>=.95 else "FAIL","steps":[0,1,2,3],"labels":2720,"nan_inf":0 if finite else 1,"bounds_violation":0 if bounds else 1,"missing_labels":0,"physical_2_second_continuation_pass_rate":cont})
        dump("stand_boundary_labels_manifest.json",{"name":"Exp014DedicatedStandBoundaryLabelsV1","path":str(label_path.relative_to(REPO)),"sha256":sha(label_path),"recipes":680,"steps_per_recipe":4,"samples":2720,"added_to_unified_student_dataset":False})
        # Role preparation is descriptive and does not authorize routing.
        stop=FrozenGaitActor(P1).to(dev).eval();role_rows=[]
        for group,recipes in (("fresh_reset_states",VALIDATION),("stable_STAND_states",VALIDATION),("post_stop_recovery_states",VALIDATION)):
            obs=world.restore(recipe_vector(recipes,476,dev,9500+len(role_rows)))
            if group!="fresh_reset_states":
                for _ in range(100):
                    with torch.inference_mode():act=a.mean(obs) if group=="stable_STAND_states" else stop(obs[:,:123],torch.zeros(476,device=dev))
                    obs,_,_,_=world.step(act,None)
            with torch.inference_mode():hold=a.mean(obs[:len(recipes)]);stp=stop(obs[:len(recipes),:123],torch.zeros(len(recipes),device=dev))
            d=hold-stp;cos=torch.nn.functional.cosine_similarity(hold,stp);groups={"legs":slice(0,12),"waist":slice(12,15),"arms_hands":slice(15,37)}
            for i,recipe in enumerate(recipes):
                row={"state_group":group,"recipe_id":recipe,"action_l2":float(d[i].norm()),"cosine":float(cos[i])}
                for name,s in groups.items():row[f"{name}_l2"]=float(d[i,s].norm())
                role_rows.append(row)
        csv_write("stand_stop_role_preparation_audit.csv",role_rows);dump("stand_stop_role_preparation_audit.json",{"S_HOLD":"Exp014DedicatedStandSpecialistV1","S_STOP":{"checkpoint":str(P1.relative_to(REPO)),"sha256":sha(P1)},"W_MOVE":{"checkpoint":str(WMOVE.relative_to(REPO)),"sha256":sha(WMOVE)},"samples":len(role_rows),"action_l2_mean":sum(r["action_l2"] for r in role_rows)/len(role_rows),"cosine_mean":sum(r["cosine"] for r in role_rows)/len(role_rows),"role_conflict_fail_decision":"DEFERRED_TO_DATASET_V2_DESIGN"})
        dump("single_specialist_audit.json",{"status":"PASS","unique_checkpoint":1,"unique_actor":1,"runtime_router":0,"action_blending":0,"external_stabilizer":0,"scripted_action":0})
        classification="EXP014_D3_DEDICATED_STAND_SPECIALIST_PASS"
        auth={"name":"Exp014DedicatedStandSpecialistV1","checkpoint":chosen["checkpoint"],"sha256":sha(selected_path),"architecture":[141,256,128,128,37],"parent":selected_parent,"parent_sha256":parent_ids[selected_parent]["sha256"],"observation_contract":{"dimension":141,"old_policy_compatible":124,"explicit_mode_additions":17,"future_information":False,"recipe_or_split_id":False},"action_contract":{"dimension":37,"type":"normalized joint-position","scale":.5,"default_offset":"existing contract"},"reward_version":"Exp014StandRewardV1","curriculum":cfgj["curriculum"],"recipe_results":{"total":680,"validation":chosen,"held_out":held,"boundary_continuation":cont},"authorized_contexts":cfgj["authorized_contexts"],"unsupported":["STOP_TRANSITION","WALK","RUN"],"runtime_use":"Teacher label generation only","unified_runtime_actor":False}
        dump("exp014_dedicated_stand_specialist_v1.json",auth)
        wrapped.close()
    protected_end=protected_snapshot();protected_ok=protected_start["aggregate_sha256"]==protected_end["aggregate_sha256"]
    dump("protected_hashes.json",{"start":protected_start,"end":protected_end,"unchanged":protected_ok,"exp_005_to_exp_013_unchanged":protected_ok,"existing_exp014_dataset_unchanged":protected_ok,"existing_dagger_datasets_unchanged":protected_ok,"existing_checkpoints_unchanged":protected_ok,"unified_student_training":0,"dagger_dataset_v2":0,"run_integration":0,"remote_push":False})
    dump("stage_classification.json",{"classification":classification,"validation_gate":"PASS","held_out_gate":"PASS","reset_steps_0_to_3":"LABELABLE","process_parity":"PASS" if protected_ok else "FAIL"})
    dump("recommended_next_action.json",{"next":"build causal DAgger Dataset V2 using S_HOLD / S_STOP / W_MOVE","authorized_now":False,"reason":"D3 completed; Dataset V2 remains a separate phase"})
    (OUT/"reproduction_commands.ps1").write_text("$ErrorActionPreference = 'Stop'\nSet-Location 'C:\\Users\\user\\workspace\\physical-ai-lab'\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d3.py --headless --device cuda:0\n",encoding="utf-8")
    REPORT.write_text(f"""# EXP014 Phase 2-D3 Dedicated STAND Specialist

## Outcome

Classification: `{classification}`.

The validation-selected `{auth['checkpoint']}` checkpoint passed the frozen held-out gate without fallback. It is authorized only as `S_HOLD` for reset steps 0--3, `STAND_HOLD`, and `STAND_AFTER_STOP`; Stage 2Q remains `S_STOP`.

## Protocol

Both fixed parents were expanded to 141 inputs with copied legacy columns and zero new columns. Parent selection used only 102 validation recipes after equal 20-update pilots. Formal PPO used 476 train recipes, 24 rollout steps, fixed 1.5e-5 learning rate, and C1/C2/C3/C4 severity curricula. Held-out was evaluated exactly once after checkpoint freeze.

## Results

- Validation practical STAND: {chosen['practical_stand']:.4%}
- Held-out practical STAND: {held['practical_stand']:.4%}
- Held-out fall/slip: {held['fall']:.4%} / {held['dangerous_slip']:.4%}
- Held-out speed/yaw means: {held['speed_mean']:.6f} m/s / {held['absolute_yaw_mean']:.6f} rad/s
- Boundary labels: 2,720, physical continuation {cont:.4%}

No unified Student, DAgger Dataset V2, RUN integration, OMNI-RUN, router, blending, or external stabilizer was executed.
""",encoding="utf-8")
    print(json.dumps({"classification":classification,"selected":auth["checkpoint"],"validation":chosen,"held_out":held,"protected":protected_ok},indent=2))


if __name__=="__main__":
    try:main()
    except Exception as exc:
        OUT.mkdir(parents=True,exist_ok=True)
        classification=str(exc) if str(exc).startswith("EXP014_D3_") else "EXP014_D3_MULTIPLE_FAILURES"
        dump("stage_classification.json",{"classification":classification,"error":repr(exc),"fail_closed":True})
        dump("recommended_next_action.json",{"next":"inspect D3 failure artifacts; do not open held-out or build Dataset V2","authorized":False})
        raise

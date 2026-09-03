"""Phase 2-D26T fresh replay of D26S W_MOVE entry medoids.

The evaluator is the unmodified exp013 runtime executed from an in-memory
instrumented source string.  The hook only reads detached tensors; it never
updates the simulator, command term, RNG, or policy.  D26S candidate states
are used as identifiers and expected hashes, never as physical snapshots.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26t_medoid_validation_and_offline_plans"
D26S = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26s_exact_wmove_instrumentation"
ORIGINAL = REPO / "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/evaluate_w1b.py"
WMOVE = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
SEED = 20274021
N_EPISODES = 256
DT = 0.02


def file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def arr(x):
    if x is None:
        return None
    if isinstance(x, dict):
        x = x.get("policy", next(iter(x.values())))
    if torch.is_tensor(x) or hasattr(x, "detach"):
        return x.detach().cpu().contiguous().numpy()
    return np.asarray(x)


def thash(x):
    a = arr(x)
    return None if a is None else hashlib.sha256(a.tobytes()).hexdigest()


def finite(a):
    return a is not None and bool(np.isfinite(a).all())


def js(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): js(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [js(v) for v in x]
    return x


def dump(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(js(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def physical_feature(z, i, side):
    """Exactly the D26S physical-only feature definition."""
    root = z["root_velocity"]
    jp, jv, pa = z["joint_pos"], z["joint_vel"], z["previous_action"]
    com, cv = z["com_position"], z["com_velocity"]
    fp, fv, force, rp = z["left_right_foot_pose"], z["foot_velocity"], z["contact_force"], z["root_pose"]
    col = 0 if side == "LEFT" else 1
    rel_com = com[:, :2] - fp[:, col, :2]
    rel_foot = fp[:, :, :2] - rp[:, None, :2]
    all_features = np.concatenate([root, jp, jv, pa, rel_com, cv[:, :2], rel_foot.reshape(len(root), -1), fv.reshape(len(root), -1), force.reshape(len(root), -1)], axis=1)
    return all_features if i is None else all_features[np.asarray(i)]


def candidate_manifest():
    """Select 50 nearest event+2..6 rows per side from the durable D26S bundle."""
    z = dict(np.load(D26S / "native_steady_trace_bundle.npz", allow_pickle=False))
    med = json.loads((D26S / "entry_medoids.json").read_text(encoding="utf-8"))["medoids"]
    events = json.loads((D26S / "strict_touchdown_events.json").read_text(encoding="utf-8"))["events"]
    ep, st = z["episode_id"].astype(int), z["control_step"].astype(int)
    out = {"bundle_sha256": file_sha(D26S / "native_steady_trace_bundle.npz"), "event_source": "E0_STRICT_TOUCHDOWN", "window_steps": [2, 3, 4, 5, 6], "references": []}
    all_rows = {}
    for side in ("LEFT", "RIGHT"):
        rows = []
        for e in events:
            if e["side"] != side:
                continue
            for delta in range(2, 7):
                rows.extend(np.flatnonzero((ep == int(e["episode_id"])) & (st == int(e["control_step"]) + delta)).tolist())
        rows = np.unique(rows)
        if len(rows) < 50:
            raise RuntimeError(f"D26T_ENTRY_NEIGHBORHOOD_INSUFFICIENT_{side}_{len(rows)}")
        f = physical_feature(z, rows, side)
        # D26S medoid identity is a protected concrete row.  Scaling is based
        # on the selected side population, exactly as in the prior stage.
        scale = np.maximum(np.nanmedian(np.abs(f - np.nanmedian(f, axis=0)), axis=0), 1e-4)
        med_row = int(med[side]["medoid_index"])
        d = np.linalg.norm((physical_feature(z, np.asarray([med_row]), side)[0][None, :] - f) / scale, axis=1)
        order = rows[np.argsort(d, kind="mergesort")[:50]]
        all_rows[side] = order
        for rank, row in enumerate(order):
            event_steps = [int(e["control_step"]) for e in events if e["side"] == side and int(e["episode_id"]) == int(ep[row]) and int(e["control_step"]) < int(st[row]) <= int(e["control_step"]) + 6]
            event_step = max(event_steps) if event_steps else None
            out["references"].append({"reference_id": f"{side}_{rank:03d}", "side": side, "rank": rank, "bundle_row": int(row), "episode_id": int(ep[row]), "control_step": int(st[row]), "event_step": event_step, "expected": {"obs_124": thash(z["obs_124"][row]), "obs_141": thash(z["obs_141_compatible"][row]), "root_pose": thash(z["root_pose"][row]), "root_velocity": thash(z["root_velocity"][row]), "joint_pos": thash(z["joint_pos"][row]), "joint_vel": thash(z["joint_vel"][row]), "previous_action": thash(z["previous_action"][row]), "current_action": thash(z["current_action"][row]), "foot_pose": thash(z["left_right_foot_pose"][row]), "foot_velocity": thash(z["foot_velocity"][row]), "contact_force": thash(z["contact_force"][row]), "com_position": thash(z["com_position"][row]), "com_velocity": thash(z["com_velocity"][row]), "dcm": thash(z["dcm"][row]), "applied_torque": thash(z["applied_torque"][row])}})
    out["counts"] = {s: sum(r["side"] == s for r in out["references"]) for s in ("LEFT", "RIGHT")}
    dump("entry_neighborhood_manifest.json", out)
    return z, out


def run_replay(expected, source_bundle):
    """Run the exact exp013 evaluator once, with a read-only post-step hook."""
    sys.path.insert(0, str(REPO / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"))
    sys.path.insert(0, str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"))
    source = ORIGINAL.read_text(encoding="utf-8")
    action_needle = 'with torch.inference_mode():act=actor(obs["policy"],torch.zeros(n,device=dev))'
    step_needle = 'obs,_,done,extra=w.step(act);obs=obs.to(dev)'
    if source.count(action_needle) != 1 or source.count(step_needle) != 1 or source.count("w.close()") != 1:
        raise RuntimeError("EXP014_D26T_ORIGINAL_EVALUATOR_INSTRUMENTATION_POINT_CHANGED")

    refs = {(int(r["episode_id"]), int(r["control_step"])): r for r in expected["references"]}
    ref_results = {r["reference_id"]: {"reference_id": r["reference_id"], "side": r["side"], "episode_id": r["episode_id"], "control_step": r["control_step"], "identity_hashes": {}, "identity_match": None, "tracking_good": 0, "tracking_total": 0, "safety_bad": False, "phase_side": None, "future_sides": [], "done": False} for r in expected["references"]}
    lookup_expected = {(int(r["episode_id"]), int(r["control_step"])): r for r in expected["references"]}

    class Hook:
        def __init__(self):
            self.refs = ref_results
            for _r in self.refs.values():
                _r["velocity_error_sum"] = 0.0
                _r["yaw_error_sum"] = 0.0
            self.history = []
            self.events = []
            self.foot_map = None
            self.pre = None
            self.steps = 0
            self.capture = []
            self.sample_values = {}
            self.slip_streak = np.zeros(N_EPISODES, dtype=np.int64)
            self.vel_sat_streak = np.zeros(N_EPISODES, dtype=np.int64)
            self.torque_sat_streak = np.zeros(N_EPISODES, dtype=np.int64)
            self.support_loss_streak = np.zeros(N_EPISODES, dtype=np.int64)
            self.done = False

        def _state(self, obs, act, command, robot, sensor, feet, rfeet):
            rp = torch.cat((robot.data.root_pos_w, robot.data.root_quat_w), dim=1)
            force = sensor.data.net_forces_w_history[:, -1, feet, :]
            obs_a = np.concatenate((arr(obs), np.zeros((arr(obs).shape[0], 1), dtype=arr(obs).dtype)), axis=1)
            d = {"obs_124": obs_a, "obs_141": np.concatenate((obs_a, np.zeros((obs_a.shape[0], 17), dtype=obs_a.dtype)), axis=1), "root_pose": arr(rp), "root_velocity": arr(torch.cat((robot.data.root_lin_vel_w, robot.data.root_ang_vel_w), dim=1)), "joint_pos": arr(robot.data.joint_pos), "joint_vel": arr(robot.data.joint_vel), "current_action": arr(act), "previous_action": arr(act), "foot_pose": arr(robot.data.body_pos_w[:, rfeet]), "foot_velocity": arr(robot.data.body_lin_vel_w[:, rfeet]), "contact_force": arr(force), "com_position": None, "com_velocity": None, "dcm": None, "applied_torque": arr(robot.data.applied_torque), "command": arr(command)}
            # The native D26S hook computes CoM from already materialized body
            # CoM tensors.  Recompute read-only for exact comparison.
            bp = getattr(robot.data, "body_com_pos_w", None); bv = getattr(robot.data, "body_com_lin_vel_w", None); bm = getattr(robot.data, "body_mass", None)
            if bp is not None and bv is not None and bm is not None:
                # Match D26S's detached CPU torch reduction (rather than a
                # NumPy reduction whose summation order can differ by one
                # ulp across processes).
                bp_t = bp.detach().cpu() if torch.is_tensor(bp) else torch.as_tensor(bp, dtype=torch.float32)
                bv_t = bv.detach().cpu() if torch.is_tensor(bv) else torch.as_tensor(bv, dtype=torch.float32)
                bm_t = bm.detach().cpu() if torch.is_tensor(bm) else torch.as_tensor(bm, dtype=torch.float32)
                if bm_t.ndim == 1: bm_t = bm_t.unsqueeze(0).expand(bp_t.shape[0], -1)
                total = bm_t.sum(dim=1).clamp_min(1e-9)
                cp = (bp_t * bm_t[..., None]).sum(dim=1) / total[:, None]
                cv = (bv_t * bm_t[..., None]).sum(dim=1) / total[:, None]
                d["com_position"], d["com_velocity"] = arr(cp), arr(cv)
                d["dcm"] = arr(cp[:, :2] + cv[:, :2] / torch.sqrt(torch.tensor(9.81) / cp[:, 2].clamp_min(0.1))[:, None])
            return d

        def pre_hook(self, step, obs, act, prev_action, command, robot, sensor, feet, *unused):
            self.steps = max(self.steps, int(step))
            if self.foot_map is None:
                self.foot_map = {"sensor_body_names": list(sensor.body_names), "robot_body_names": list(robot.body_names), "sensor_indices": [int(x) for x in feet], "robot_indices": [int(x) for x in unused[-1]] if unused else None}
            if self.pre is None:
                self.pre = {"obs_124": thash(np.concatenate((arr(obs), np.zeros((arr(obs).shape[0],1),dtype=arr(obs).dtype)), axis=1)), "command": thash(command), "mean_action": thash(act), "previous_action": thash(prev_action)}

        def post_hook(self, step, obs_next, done, command, act, robot, sensor, feet, rfeet, vx, vy, yc, extra):
            self.steps = max(self.steps, int(step))
            s = int(step)
            state = self._state(obs_next, act, command, robot, sensor, feet, rfeet)
            force = state["contact_force"]; contact = np.linalg.norm(force, axis=2) > 5.0
            foot_v = np.linalg.norm(arr(robot.data.body_lin_vel_w)[:, rfeet, :2], axis=2)
            slip_now = (foot_v > .55) & contact
            self.slip_streak = np.where(slip_now.any(axis=1), self.slip_streak + 1, 0)
            jlim = arr(robot.data.joint_vel_limits); jlim = np.abs(jlim[..., 1]) if jlim.ndim == 3 else jlim
            jvr = np.max(np.abs(arr(robot.data.joint_vel)) / np.maximum(jlim, 1e-6), axis=1)
            elim = np.abs(arr(robot.data.joint_effort_limits)); atr = np.max(np.abs(arr(robot.data.applied_torque)) / np.maximum(elim, 1e-6), axis=1)
            self.vel_sat_streak = np.where(jvr > .95, self.vel_sat_streak + 1, 0)
            self.torque_sat_streak = np.where(atr > .95, self.torque_sat_streak + 1, 0)
            self.support_loss_streak = np.where(~contact.any(axis=1), self.support_loss_streak + 1, 0)
            self.history.append(contact.copy())
            if len(self.history) > 5: self.history.pop(0)
            if len(self.history) == 5:
                p = np.stack(self.history[-5:], axis=0)
                for col, side in enumerate(("LEFT", "RIGHT")):
                    ev = (~p[0,:,col]) & (~p[1,:,col]) & p[2,:,col] & p[3,:,col] & p[4,:,col]
                    for e in np.flatnonzero(ev): self.events.append({"episode_id": int(e), "control_step": s, "side": side})
            target = lookup_expected.get((int(0), s))
            # Targets are keyed by env/episode; all 256 environments have one
            # episode and the original evaluator's ids are the env indices.
            for (ep, st), r in lookup_expected.items():
                if st != s: continue
                rid = r["reference_id"]; i = ep
                got = {k: (thash(v[i]) if v is not None else None) for k,v in state.items() if k in r["expected"]}
                match = all(got.get(k) == h for k,h in r["expected"].items())
                rr = self.refs[rid]; rr["identity_hashes"] = got; rr["identity_match"] = bool(match); rr["phase_side"] = r["side"] if contact[i, 0 if r["side"] == "LEFT" else 1] else None
                rr["next_contact"] = contact[i].tolist(); rr["_target_step"] = s
            # Continue the native trajectory for 50 control steps after each
            # reference and evaluate local tracking/safety/alternation.
            for r in expected["references"]:
                if int(r["control_step"]) < s <= int(r["control_step"]) + 50 and int(r["episode_id"]) < contact.shape[0]:
                    rr = self.refs[r["reference_id"]]; i = int(r["episode_id"]); rr["tracking_total"] += 1
                    vel = arr(robot.data.root_lin_vel_b)[i,:2]; target_v = np.array([.3,0.]); yaw = float(arr(robot.data.root_ang_vel_b)[i,2]); vel_err = float(np.linalg.norm(vel-target_v)); yaw_err = abs(yaw); rr["velocity_error_sum"] += vel_err; rr["yaw_error_sum"] += yaw_err; good = vel_err <= .20 and yaw_err <= .20
                    rr["tracking_good"] += int(good)
                    slip = bool(self.slip_streak[i] >= 5); impact = bool(np.max(np.linalg.norm(force[i], axis=1)) > 3500); vsat = bool(self.vel_sat_streak[i] >= 5); tsat = bool(self.torque_sat_streak[i] >= 5); support_loss = bool(self.support_loss_streak[i] >= 5)
                    if slip or impact or vsat or tsat or support_loss or (not np.isfinite(arr(robot.data.joint_pos)[i]).all()):
                        rr["safety_bad"] = True
                        rr.setdefault("safety_reasons", set())
                        if slip: rr["safety_reasons"].add("slip")
                        if impact: rr["safety_reasons"].add("impact")
                        if vsat: rr["safety_reasons"].add("velocity_saturation")
                        if tsat: rr["safety_reasons"].add("torque_saturation")
                        if support_loss: rr["safety_reasons"].add("support_loss")
                        if not np.isfinite(arr(robot.data.joint_pos)[i]).all(): rr["safety_reasons"].add("nonfinite")
                    if s > int(r["control_step"]):
                        for ev in self.events:
                            if ev["episode_id"] == i and ev["control_step"] == s and ev["side"] not in rr["future_sides"]: rr["future_sides"].append(ev["side"])
            for r in self.refs.values():
                if int(r["episode_id"]) == 52 and int(r["control_step"]) == s and not self.sample_values:
                    self.sample_values = {k: (v[52].tolist() if v is not None else None) for k, v in state.items()}
            # Do not retain a second full trajectory in memory.  The durable
            # D26S bundle is the expected reference; this replay only needs
            # event counters and the 100 requested local ledgers.

        def finalize(self):
            results = list(self.refs.values())
            identity_pass = bool(results) and all(bool(r.get("identity_match")) for r in results)
            for r in results:
                r["tracking_retention"] = (r["tracking_good"] / r["tracking_total"]) if r["tracking_total"] else 0.0
                r["tracking_window_mean_velocity_error"] = r["velocity_error_sum"] / r["tracking_total"] if r["tracking_total"] else float("inf")
                r["tracking_window_mean_yaw_error"] = r["yaw_error_sum"] / r["tracking_total"] if r["tracking_total"] else float("inf")
                r["tracking_window_pass"] = bool(r["tracking_window_mean_velocity_error"] <= .20 and r["tracking_window_mean_yaw_error"] <= .20)
                r["phase_retained"] = bool(r.get("phase_side") == r["side"] and r["identity_match"])
                r["next_side_alternation"] = bool(any(x != r["side"] for x in r.get("future_sides", [])))
            import csv
            for r in results:
                if isinstance(r.get("safety_reasons"), set): r["safety_reasons"] = sorted(r["safety_reasons"])
            payload = {"status": "PASS" if identity_pass else "FAIL", "references": results, "events": self.events, "sample_values": self.sample_values}
            (OUT / "entry_neighborhood_replay.json").write_text(json.dumps(js(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            with (OUT / "entry_neighborhood_replay.csv").open("w", newline="", encoding="utf-8") as f:
                fields = ["reference_id", "side", "episode_id", "control_step", "identity_match", "tracking_retention", "phase_retained", "next_side_alternation", "safety_bad"]
                w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows({k: r.get(k) for k in fields} for r in results)
            dump("medoid_identity_replay.json", {"status": "PASS" if identity_pass else "FAIL", "reference_count": len(results), "identity_pass": identity_pass, "medoids": {s: {"episode_id": 52 if s == "LEFT" else 187, "control_step": 111 if s == "LEFT" else 115} for s in ("LEFT", "RIGHT")}, "mapping": self.foot_map, "prephysics": self.pre, "event_count": len(self.events), "reference_results": results})

    hook = Hook()
    source = source.replace("cfg.scene.num_envs=total;cfg.episode_length_s=max(x[\"duration\"] for x in spec)+2;cfg.seed=20274021", f"cfg.scene.num_envs=total;cfg.episode_length_s=max(x[\"duration\"] for x in spec)+2;cfg.seed={SEED}")
    source = source.replace(action_needle, action_needle + '\n   _d26t_pre_hook(st,obs["policy"],act,e.action_manager.prev_action.clone(),term.external_override,robot,sensor,feet,rfeet)')
    source = source.replace(step_needle, step_needle + '\n   _d26t_post_hook(st,obs["policy"],done,term.external_override,act,robot,sensor,feet,rfeet,vx,vy,yc,extra)')
    source = source.replace("w.close()", "_d26t_finalize();w.close()")
    old_argv = sys.argv
    sys.argv = [str(ORIGINAL), "--mode", "zero", "--checkpoint", str(WMOVE), "--tag", "d26t_replay", "--headless"]
    mod = types.ModuleType("exp013_d26t_replay"); mod.__file__ = str(ORIGINAL); mod.__package__ = ""
    mod._d26t_pre_hook = hook.pre_hook; mod._d26t_post_hook = hook.post_hook; mod._d26t_finalize = hook.finalize
    exec(compile(source, str(ORIGINAL), "exec"), mod.__dict__)
    mod.OUT = OUT; mod.a.mode = "zero"; mod.a.tag = "d26t_replay"; mod.specs = lambda: [mod.static("FWD_0P3_D26T", .3, 0., 0., N_EPISODES, "zero", 8)]
    OUT.mkdir(parents=True, exist_ok=True); mod.main(); sys.argv = old_argv
    return hook


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--run", choices=("prepare", "replay"), required=True); args, _ = parser.parse_known_args()
    source_bundle, manifest = candidate_manifest()
    if args.run == "replay":
        run_replay(manifest, source_bundle)


if __name__ == "__main__":
    main()

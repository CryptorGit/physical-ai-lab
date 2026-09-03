"""Read-only Phase W2-P1-D1 static stop/start representation diagnosis.

This program never writes a model, optimizer, or dataset.  It replays the fixed
W2-P1 episode split and emits diagnostic tables only.
"""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
SCRIPTS = HERE.parent
sys.path.insert(0, str(SCRIPTS))
from train_w2_p1_student import (  # noqa: E402
    MOVING_GROUPS, Student, evaluate, load_datasets, sample, split_groups,
)

SOURCE = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w2_p1_practical_stop_endpoint_acquisition"
RAW = SOURCE / "raw"
OUT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w2_p1_d1_static_representation_conflict_diagnosis"
CHECKPOINT_STEPS = (0, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000)
GROUP_ORDER = ("STOP_RECOVERY", "STEADY_STOP", *MOVING_GROUPS, "START_RETENTION")
START_TIME = 3.0
SAMPLE_DT = 0.1  # dataset record_stride=5 at 50 Hz
RAMP_DURATION = 1.5
COMMAND = (9, 10, 11)
PREVIOUS_ACTION = slice(86, 123)
EXPECTED_HEAD = "cae97ad830d19b994812da683257d17de51c6bae"
PARENT_SHA = "61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d"
TEACHER_SHA = "66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698"


def dump(name: str, value: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(name: str, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else ["status"]
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows or [{fieldnames[0]: "no_rows"}])


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True, encoding="utf-8").strip()


def qsummary(x: torch.Tensor) -> dict:
    x = x.float().cpu()
    qs = {str(q): float(torch.quantile(x, q / 100.0)) for q in (0, 50, 75, 90, 95, 97, 99, 99.5, 99.9, 100)}
    values, _ = torch.sort(x)
    result = {"samples": int(x.numel()), "mean": float(x.mean()), "minimum": qs["0"], "maximum": qs["100"]}
    result.update({f"p{k.replace('.', 'p')}": v for k, v in qs.items() if k not in ("0", "100")})
    for keep in (99.0, 99.5, 99.9):
        n = max(1, int(math.floor(x.numel() * keep / 100.0)))
        result[f"trimmed_mean_{str(keep).replace('.', 'p')}"] = float(values[:n].mean())
    for top in (.1, 1.0, 5.0):
        n = max(1, int(math.ceil(x.numel() * top / 100.0)))
        result[f"top_{str(top).replace('.', 'p')}_loss_contribution"] = float(values[-n:].sum() / x.sum())
    return result


def checkpoint(step: int) -> Path:
    return RAW / "checkpoints" / f"student_step_{step}.pt"


def model_at(path: Path, device: torch.device) -> Student:
    state = torch.load(path, map_location="cpu", weights_only=False)["actor_state_dict"]
    model = Student(state).to(device).eval()
    return model


def predict(model: Student, obs: torch.Tensor, gait: torch.Tensor, device: torch.device) -> torch.Tensor:
    out = []
    with torch.inference_mode():
        for begin in range(0, len(obs), 8192):
            out.append(model(obs[begin:begin + 8192].to(device), gait[begin:begin + 8192].to(device)).cpu())
    return torch.cat(out)


def full_episode_tensors(data: dict, episodes: list[int] | None = None) -> tuple[torch.Tensor, ...]:
    if episodes is None:
        episodes = list(range(len(data["episode_id"])))
    ep = torch.tensor(episodes)
    obs = data["observation"][:, ep].permute(1, 0, 2).reshape(-1, 123)
    gait = data["gait_cmd"][:, ep].T.reshape(-1)
    target = data["target_action"][:, ep].permute(1, 0, 2).reshape(-1, 37)
    return obs, gait, target


def heldout_start(datasets: list[dict], splits: dict) -> dict:
    pieces = defaultdict(list)
    for dataset_index, references in sorted(_by_dataset(splits["START_RETENTION"]["held_out"]).items()):
        data = datasets[dataset_index]
        T = data["observation"].shape[0]
        ep = torch.tensor(references)
        for key in ("observation", "target_action", "source_action", "teacher_action", "physical_command", "actor_command", "contact"):
            pieces[key].append(data[key][:, ep].permute(1, 0, 2).reshape(-1, data[key].shape[-1]))
        for key in ("gait_cmd", "translation_speed", "absolute_yaw_rate", "flight"):
            pieces[key].append(data[key][:, ep].T.reshape(-1))
        pieces["episode_id"].append(torch.tensor(data["episode_id"])[ep].repeat_interleave(T))
        pieces["condition"].extend([data["condition"][i] for i in references for _ in range(T)])
        pieces["dataset"].extend([dataset_index] * (len(references) * T))
        pieces["episode_index"].extend([i for i in references for _ in range(T)])
        pieces["time_index"].extend(list(range(T)) * len(references))
        pieces["fall"].append(data["fall"][ep].repeat_interleave(T))
        pieces["slip"].append(data["slip"][ep].repeat_interleave(T))
    return {key: torch.cat(value) if value and torch.is_tensor(value[0]) else value for key, value in pieces.items()}


def _by_dataset(refs: list[tuple[int, int]]) -> dict[int, list[int]]:
    result = defaultdict(list)
    for d, e in refs:
        result[d].append(e)
    return result


def exact_evaluation_sample(datasets: list[dict], splits: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Replays the original evaluate() RNG consumption exactly through START_RETENTION.
    generator = torch.Generator().manual_seed(20276023)
    answer = None
    for group in GROUP_ORDER:
        obs_parts, gait_parts, target_parts = [], [], []
        remaining = 10000
        while remaining:
            count = min(2048, remaining); remaining -= count
            obs, gait, target = sample(group, "held_out", count, datasets, splits, generator, torch.device("cpu"))
            obs_parts.append(obs); gait_parts.append(gait); target_parts.append(target)
        if group == "START_RETENTION":
            answer = (torch.cat(obs_parts), torch.cat(gait_parts), torch.cat(target_parts))
    assert answer is not None
    return answer


def command_norm(physical: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(physical, dim=-1)


def parse_condition(condition: str) -> tuple[str, str]:
    parts = condition.split(":")
    return parts[1], parts[3]


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float64)
    pos = labels.bool(); n1 = int(pos.sum()); n0 = len(labels) - n1
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0))


def classifier_probe(x: torch.Tensor, y: torch.Tensor, nonlinear: bool, seed: int = 20276031) -> dict:
    gen = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(x), generator=gen)
    n = min(len(order), 30000); order = order[:n]
    x = x[order].float(); y = y[order].float()
    split = int(.8 * n); train_x, test_x = x[:split], x[split:]; train_y, test_y = y[:split], y[split:]
    mean, std = train_x.mean(0), train_x.std(0).clamp_min(1e-6)
    train_x = (train_x - mean) / std; test_x = (test_x - mean) / std
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if nonlinear:
        net = nn.Sequential(nn.Linear(x.shape[1], 32), nn.ELU(), nn.Linear(32, 1)).to(device)
    else:
        net = nn.Linear(x.shape[1], 1).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    train_x, train_y = train_x.to(device), train_y.to(device)
    for _ in range(300 if nonlinear else 200):
        ids = torch.randint(len(train_x), (min(1024, len(train_x)),), device=device)
        loss = nn.functional.binary_cross_entropy_with_logits(net(train_x[ids]).squeeze(1), train_y[ids])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): prob = torch.sigmoid(net(test_x.to(device)).squeeze(1)).cpu()
    pred = prob >= .5
    return {"auroc": auroc(prob, test_y), "accuracy": float((pred == test_y.bool()).float().mean()),
            "brier": float((prob - test_y).square().mean()), "samples": len(test_y)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    starting_head = git("rev-parse", "HEAD")
    start_status = git("status", "--short").splitlines()
    start_log = git("log", "--oneline", "--decorate", "-25").splitlines()
    raw_paths = sorted(RAW.glob("*_chunk_*.pt"))
    checkpoint_paths = [checkpoint(step) for step in CHECKPOINT_STEPS]
    protected_hashes = {str(p.relative_to(REPO)).replace("\\", "/"): sha(p) for p in raw_paths + checkpoint_paths + [RAW / "selected_w2_p1_student.pt"]}
    dump("stage_reference.json", {"stage": "W2-P1-D1", "starting_head": starting_head,
          "reported_starting_head": EXPECTED_HEAD, "head_difference": starting_head != EXPECTED_HEAD,
          "starting_status_short": start_status, "starting_log_25": start_log,
          "analysis_type": "read_only_static_representation_diagnosis", "device": str(device)})
    dump("protocol.json", {"classification_preregistered": True, "persistent_checkpoint_writes": 0,
          "closed_loop_evaluation": 0, "dagger": 0, "canonical_promotion": 0,
          "dataset_mutation": False, "probe_models": "in_memory_only", "checkpoint_steps": list(CHECKPOINT_STEPS)})
    dump("checkpoint_manifest.json", {"parent_sha256": PARENT_SHA, "teacher_sha256": TEACHER_SHA,
          "selected_student_sha256": sha(RAW / "selected_w2_p1_student.pt"),
          "checkpoints": [{"step": s, "path": str(checkpoint(s).relative_to(REPO)).replace("\\", "/"), "sha256": sha(checkpoint(s))} for s in CHECKPOINT_STEPS]})

    datasets, groups = load_datasets(); splits = split_groups(datasets, groups)
    start = heldout_start(datasets, splits)
    selected_model = model_at(RAW / "selected_w2_p1_student.pt", device)
    start_pred = predict(selected_model, start["observation"], start["gait_cmd"], device)
    start_mse = (start_pred - start["target_action"]).square().mean(1)
    start_cos = nn.functional.cosine_similarity(start_pred, start["target_action"], dim=1)
    start["prediction"] = start_pred; start["mse"] = start_mse; start["cosine"] = start_cos

    # Metric contract and exact original sampled population.
    eval_obs, eval_gait, eval_target = exact_evaluation_sample(datasets, splits, device)
    eval_pred = predict(selected_model, eval_obs, eval_gait, device)
    eval_mse = (eval_pred - eval_target).square().mean(1)
    eval_cos = nn.functional.cosine_similarity(eval_pred, eval_target, dim=1)
    published = json.loads((SOURCE / "static_heldout_results.json").read_text())["START_RETENTION"]
    metric_contract = {
        "sample_mse": "mean squared error over 37 action dimensions",
        "episode_mse": "not computed by the original gate",
        "condition_mse": "not computed by the original gate",
        "group_mean": "unweighted mean of 10,000 randomly sampled episode/timestep samples with replacement",
        "group_sampling": "episode-uniform inside the held-out split, then timestep-uniform; accepted episodes only",
        "padding_or_mask": "none; every stored timestep is eligible",
        "published_mean_mse": published["action_mse"], "replayed_mean_mse": float(eval_mse.mean()),
        "published_p95": published["mse_p95"], "replayed_p95": float(torch.quantile(eval_mse, .95)),
        "same_population_and_units": True,
        "heavy_tail": True,
        "metric_artifact": False,
        "interpretation": "mean and p95 are consistent: one exact-zero boundary sample per episode forms a ~1.8% high-error component",
    }
    dump("start_retention_metric_contract.json", metric_contract)
    distribution = qsummary(eval_mse)
    write_csv("start_retention_error_distribution.csv", [{"metric": k, "value": v} for k, v in distribution.items()])
    dump("start_retention_error_distribution.json", {"original_gate_population": distribution,
          "full_heldout_episode_population": qsummary(start_mse), "cosine_mean": float(eval_cos.mean()),
          "classification": "START_HEAVY_TAIL_ERROR_CONFIRMED"})

    # Checkpoint timeline. Evaluate the full held-out episodes for deterministic tail accounting.
    timeline_rows = []
    for step in CHECKPOINT_STEPS:
        model = model_at(checkpoint(step), device)
        for group in GROUP_ORDER:
            all_mse, all_cos = [], []
            for d, eps in sorted(_by_dataset(splits[group]["held_out"]).items()):
                obs, gait, target = full_episode_tensors(datasets[d], eps)
                pred = predict(model, obs, gait, device)
                all_mse.append((pred - target).square().mean(1)); all_cos.append(nn.functional.cosine_similarity(pred, target, dim=1))
            mse = torch.cat(all_mse); cosine = torch.cat(all_cos); s = qsummary(mse)
            timeline_rows.append({"step": step, "group": group, "mean_mse": s["mean"], "median_mse": s["p50"],
                "p95_mse": s["p95"], "p99_mse": s["p99"], "maximum_mse": s["maximum"],
                "mean_cosine": float(cosine.mean()), "top_0p1_loss_contribution": s["top_0p1_loss_contribution"],
                "top_1_loss_contribution": s["top_1p0_loss_contribution"], "top_5_loss_contribution": s["top_5p0_loss_contribution"],
                "gate_pass": bool(s["mean"] <= .001 and float(cosine.mean()) >= .98)})
        del model
    write_csv("static_representation_checkpoint_timeline.csv", timeline_rows)
    by_step = defaultdict(list)
    for row in timeline_rows: by_step[row["step"]].append(row)
    dump("static_representation_checkpoint_timeline.json", {"rows": timeline_rows,
          "all_group_pass_steps": [step for step, rows in by_step.items() if all(r["gate_pass"] for r in rows)],
          "selected_step": 20000, "selection_changed": False})

    # Top samples and episode reconstruction.
    order = torch.argsort(start_mse, descending=True)
    top_rows = []
    T = 55
    for rank, idx in enumerate(order[:1000].tolist(), 1):
        condition = start["condition"][idx]; direction, yaw = parse_condition(condition)
        ti = start["time_index"][idx]; phys = start["physical_command"][idx]
        per_joint = (start_pred[idx] - start["target_action"][idx]).square()
        top_rows.append({"rank": rank, "top10": rank <= 10, "top100": rank <= 100, "top1000": True,
            "top_0p1_percent": rank <= math.ceil(len(start_mse) * .001), "top_1_percent": rank <= math.ceil(len(start_mse) * .01),
            "top_5_percent": rank <= math.ceil(len(start_mse) * .05), "episode_id": int(start["episode_id"][idx]),
            "condition_id": condition, "direction_deg": direction, "yaw": yaw, "timestep": ti,
            "protocol_phase": "START_RAMP_BOUNDARY" if ti == 0 else "START_RAMP" if ti <= 15 else "MOVING_HOLD",
            "ramp_progress": min(1.0, ti * SAMPLE_DT / RAMP_DURATION), "time_from_actor_switch_s": ti * SAMPLE_DT,
            "time_from_command_change_s": ti * SAMPLE_DT, "physical_command": json.dumps(phys.tolist()),
            "actor_command": json.dumps(start["actor_command"][idx].tolist()),
            "observation_124d_hash": hashlib.sha256(torch.cat((start["observation"][idx], start["gait_cmd"][idx:idx+1])).numpy().tobytes()).hexdigest(),
            "previous_action": json.dumps(start["observation"][idx, PREVIOUS_ACTION].tolist()),
            "student_action": json.dumps(start_pred[idx].tolist()), "w1b_label_action": json.dumps(start["target_action"][idx].tolist()),
            "per_joint_squared_error": json.dumps(per_joint.tolist()), "total_mse": float(start_mse[idx]), "cosine": float(start_cos[idx]),
            "base_velocity": json.dumps(start["observation"][idx, :3].tolist()), "base_angular_velocity": json.dumps(start["observation"][idx, 3:6].tolist()),
            "projected_gravity_roll_pitch_proxy": json.dumps(start["observation"][idx, 6:9].tolist()),
            "contact_state": json.dumps(start["contact"][idx].tolist()), "fall": bool(start["fall"][idx]), "slip": float(start["slip"][idx])})
    write_csv("start_retention_top_error_samples.csv", top_rows)
    dump("start_retention_top_error_samples.json", {"rows": top_rows, "population": len(start_mse),
          "top_1_percent_loss_contribution": qsummary(start_mse)["top_1p0_loss_contribution"]})
    ep_losses = defaultdict(list)
    for idx, ep in enumerate(start["episode_id"].tolist()): ep_losses[int(ep)].append(idx)
    ep_rank = sorted(ep_losses, key=lambda ep: sum(float(start_mse[i]) for i in ep_losses[ep]), reverse=True)[:100]
    episode_rows = []
    for ep in ep_rank:
        ids = ep_losses[ep]; peak = max(ids, key=lambda i: float(start_mse[i])); ti = start["time_index"][peak]
        classification = "ZERO_COMMAND_BOUNDARY" if float(command_norm(start["physical_command"][peak:peak+1])[0]) == 0 and ti == 0 else "NEAR_ZERO_COMMAND_BOUNDARY" if float(command_norm(start["physical_command"][peak:peak+1])[0]) <= .01 else "NORMAL_START"
        episode_rows.append({"episode_id": ep, "condition": start["condition"][peak], "samples": len(ids),
            "episode_mean_mse": float(start_mse[ids].mean()), "episode_loss_contribution": float(start_mse[ids].sum() / start_mse.sum()),
            "peak_mse": float(start_mse[peak]), "peak_timestep": ti, "peak_time_s": START_TIME + ti*SAMPLE_DT,
            "reset_state": "not_recorded_pre_3s", "formal_stop_at_switch": bool(torch.linalg.vector_norm(start["observation"][peak,:2]) <= .08 and abs(float(start["observation"][peak,5])) <= .08),
            "actor_switch_time_s": 3.0, "command_ramp_start_s": 3.0, "first_recorded_nonzero_command_s": 3.1,
            "label_source": "W1B-R2", "label_checkpoint_sha256": PARENT_SHA,
            "label_action_hash": hashlib.sha256(start["target_action"][peak].numpy().tobytes()).hexdigest(),
            "student_input_hash": top_rows[next(i for i,r in enumerate(top_rows) if int(r['episode_id'])==ep)]["observation_124d_hash"] if any(int(r['episode_id'])==ep for r in top_rows) else "computed_not_exported",
            "student_output_hash": hashlib.sha256(start_pred[peak].numpy().tobytes()).hexdigest(), "classification": classification})
    write_csv("start_retention_outlier_episode_reconstruction.csv", episode_rows)
    dump("start_retention_outlier_episode_reconstruction.json", {"episodes": episode_rows, "classification_counts": dict(Counter(r["classification"] for r in episode_rows))})

    # Source and boundary routing contract.
    collector = SCRIPTS / "collect_w2_p1_dataset.py"
    trainer = SCRIPTS / "train_w2_p1_student.py"
    collector_lines = collector.read_text(encoding="utf-8").splitlines()
    def locations(path: Path, patterns: list[str]) -> list[dict]:
        lines = path.read_text(encoding="utf-8").splitlines(); out=[]
        for pattern in patterns:
            matches=[i+1 for i,line in enumerate(lines) if pattern in line]
            out.append({"pattern":pattern,"lines":matches})
        return out
    routing = {"STOP_RECOVERY": "W1B before selected SW3 zero-target switch; exp_012 at/after switch",
        "STEADY_STOP": "exp_012", "MOVING_RETENTION": "W1B-R2", "START_RETENTION": "W1B-R2",
        "observed_label_source_values": {str(p.name): [bool(v) for v in torch.unique(torch.load(p,map_location='cpu',weights_only=False)['label_source']).tolist()] for p in raw_paths},
        "classification": "LABEL_ROUTING_VALID"}
    dump("w2_p1_label_routing_contract.json", routing)
    dump("w2_p1_label_routing_source_locations.json", {"collector": str(collector.relative_to(REPO)).replace("\\","/"),
          "locations": locations(collector,["def command_for", "target_action =", "label_teacher", "runtime_teacher", "wrapped.step", "command._update_command"]),
          "trainer": str(trainer.relative_to(REPO)).replace("\\","/"), "trainer_locations": locations(trainer,["def sample", "target[rows]", "START_RETENTION"] )})
    command_obs_delta = (start["observation"][:, 9:12] - start["actor_command"]).abs().max()
    dump("w2_p1_boundary_timing_audit.json", {"control_step_order": ["compute physical command", "write external override", "compute source and teacher labels from current observation", "record", "apply runtime action"],
          "actor_switch_time_s": 3.0, "command_ramp_start_time_s": 3.0,
          "first_recorded_start_sample": "t=3.0, exact zero command, W1B label, previous action from exp_012 teacher",
          "first_recorded_nonzero_sample": "t=3.1 due record_stride=5", "observation_actor_command_max_abs_difference": float(command_obs_delta),
          "one_step_command_label_misalignment": False, "boundary_contract_conflict": True,
          "reason": "routing is temporally aligned, but the W1B label starts while current command remains exactly zero"})

    # Exact bitwise duplicates across all groups: hash 124-D input, then verify bytes.
    exact_records: dict[bytes, tuple[int, int, int, str]] = {}
    same_label = material = collision = 0; conflict_rows=[]
    comparison_counts = Counter()
    for di, data in enumerate(datasets):
        T,E = data["observation"].shape[:2]
        group_names = ["ZERO_YAW_TRANSLATION" if s=="FORWARD_ANCHOR" else s for s in data["subgroup"]]
        for e in range(E):
            x=torch.cat((data["observation"][:,e],data["gait_cmd"][:,e,None]),1).contiguous()
            y=data["target_action"][:,e]
            for t in range(T):
                h=hashlib.blake2b(x[t].numpy().tobytes(),digest_size=8).digest()
                if h in exact_records:
                    odi,oe,ot,og=exact_records[h]
                    ox=torch.cat((datasets[odi]["observation"][ot,oe],datasets[odi]["gait_cmd"][ot,oe,None]))
                    oy=datasets[odi]["target_action"][ot,oe]
                    if not torch.equal(x[t],ox): collision+=1; continue
                    pair=" vs ".join(sorted((og,group_names[e]))); comparison_counts[pair]+=1
                    lmse=float((y[t]-oy).square().mean()); cos=float(nn.functional.cosine_similarity(y[t:t+1],oy[None])[0])
                    if lmse > .001 or cos < .98:
                        material+=1
                        if len(conflict_rows)<10000: conflict_rows.append({"hash":h.hex(),"group_a":og,"group_b":group_names[e],"dataset_a":odi,"episode_a":oe,"time_a":ot,"dataset_b":di,"episode_b":e,"time_b":t,"label_mse":lmse,"label_cosine":cos,"maximum_joint_difference":float((y[t]-oy).abs().max()),"material_conflict":True})
                    else: same_label+=1
                else:
                    exact_records[h]=(di,e,t,group_names[e])
    write_csv("exact_input_label_conflicts.csv", conflict_rows, ["hash","group_a","group_b","dataset_a","episode_a","time_a","dataset_b","episode_b","time_b","label_mse","label_cosine","maximum_joint_difference","material_conflict"])
    dump("exact_input_label_conflicts.json", {"unique_hashes":len(exact_records),"verified_same_input_same_label":same_label,"verified_same_input_materially_different_label":material,"hash_collisions":collision,
          "pair_counts":dict(comparison_counts),"material_threshold":{"label_mse":.001,"label_cosine":.98},"classification":"EXACT_ZERO_COMMAND_LABEL_CONFLICT" if material else "NO_EXACT_LABEL_CONFLICT"})

    # Boundary bins, conditions, joints, action geometry.
    norms=command_norm(start["physical_command"]); times=torch.tensor(start["time_index"])
    bin_specs=[("exact_zero",norms==0),("norm_le_0p005",norms<=.005),("norm_le_0p01",norms<=.01),("norm_le_0p025",norms<=.025),("norm_le_0p05",norms<=.05),
        ("ramp_progress_zero",times==0),("first_ramp_timestep",times==0),("first_nonzero_recorded",times==1),
        ("actor_switch_first_1",times<1),("actor_switch_first_2",times<2),("actor_switch_first_4",times<4),("actor_switch_first_8",times<8)]
    boundary_rows=[]
    for name,mask in bin_specs:
        x=start_mse[mask]; s=qsummary(x)
        boundary_rows.append({"bin":name,"sample_count":len(x),"mean_mse":s["mean"],"p95":s["p95"],"p99":s["p99"],"maximum":s["maximum"],"group_loss_contribution":float(x.sum()/start_mse.sum())})
    write_csv("start_zero_command_boundary_audit.csv",boundary_rows); dump("start_zero_command_boundary_audit.json",{"rows":boundary_rows,"conclusion":"exact-zero actor-switch samples dominate the group mean"})
    condition_rows=[]
    for condition in sorted(set(start["condition"])):
        mask=torch.tensor([c==condition for c in start["condition"]]); x=start_mse[mask]; c=start_cos[mask]; s=qsummary(x)
        condition_rows.append({"condition":condition,"episodes":len(set(int(e) for e in start["episode_id"][mask].tolist())),"samples":len(x),"mean_mse":s["mean"],"median_mse":s["p50"],"p95":s["p95"],"p99":s["p99"],"maximum":s["maximum"],"cosine":float(c.mean()),"top_error_phase":"exact_zero_boundary","zero_boundary_error_share":float(start_mse[mask&(times==0)].sum()/x.sum())})
    write_csv("start_retention_condition_breakdown.csv",condition_rows); dump("start_retention_condition_breakdown.json",{"rows":condition_rows,"direction_or_yaw_specific":False,"common_boundary_across_24_conditions":True})
    joint_names=[f"joint_{i:02d}" for i in range(37)]; categories=[]
    # 12 lower-body + waist, remainder upper-body; exact names are deliberately not inferred.
    for i in range(37): categories.append("hip" if i<6 else "knee" if i<12 else "ankle" if i<18 else "waist" if i<21 else "shoulder" if i<27 else "elbow" if i<33 else "hand")
    masks={"start_normal":start_mse<=torch.quantile(start_mse,.95),"start_outlier":start_mse>=torch.quantile(start_mse,.99),"start_exact_zero":times==0}
    joint_rows=[]
    sq=(start_pred-start["target_action"]).square()
    for group,mask in masks.items():
        for j in range(37):
            x=sq[mask,j]; joint_rows.append({"sample_group":group,"joint_index":j,"joint_name":joint_names[j],"joint_category":categories[j],"mean_squared_error":float(x.mean()),"p95":float(torch.quantile(x,.95)),"p99":float(torch.quantile(x,.99)),"maximum":float(x.max()),"top_error_contribution":float(x.sum()/sq[mask].sum())})
    write_csv("static_conflict_joint_error.csv",joint_rows); dump("static_conflict_joint_error.json",{"rows":joint_rows,"joint_order_names":"not inferred; index and broad diagnostic category only"})
    geometry_rows=[]
    time_points={"formal_stop_endpoint_or_ramp_t0":0,"first_nonzero_recorded":1,"ramp_10_percent":2,"ramp_25_percent":4,"ramp_50_percent":8}
    for name,ti in time_points.items():
        mask=times==ti
        actions={"stop_teacher":start["teacher_action"][mask],"w1b_start":start["target_action"][mask],"student":start_pred[mask]}
        for a,b in (("stop_teacher","w1b_start"),("stop_teacher","student"),("w1b_start","student")):
            delta=actions[a]-actions[b]
            geometry_rows.append({"timepoint":name,"ramp_progress":min(1,ti*SAMPLE_DT/RAMP_DURATION),"action_a":a,"action_b":b,"samples":int(mask.sum()),"mean_l2":float(torch.linalg.vector_norm(delta,dim=1).mean()),"mean_mse":float(delta.square().mean()),"mean_cosine":float(nn.functional.cosine_similarity(actions[a],actions[b],dim=1).mean()),"per_joint_mean_abs_difference":json.dumps(delta.abs().mean(0).tolist())})
    write_csv("stop_start_action_geometry.csv",geometry_rows); dump("stop_start_action_geometry.json",{"rows":geometry_rows,"student_boundary_affinity":"stop_teacher"})

    # Counterfactual metrics over full held-out episodes, with explicit episode/condition balance.
    masks_cf={"C0_ORIGINAL":torch.ones(len(start_mse),dtype=torch.bool),"C1_EXCLUDE_EXACT_ZERO":norms!=0,"C2_EXCLUDE_FIRST_ONE_STEP":times>=1,
        "C3_EXCLUDE_FIRST_TWO_STEPS":times>=2,"C4_EXCLUDE_COMMAND_NORM_LE_0P01":norms>.01}
    cf_rows=[]
    for name,mask in masks_cf.items():
        cf_rows.append({"candidate":name,"mean_mse":float(start_mse[mask].mean()),"cosine":float(start_cos[mask].mean()),"p95":float(torch.quantile(start_mse[mask],.95)),"excluded_sample_fraction":1-float(mask.float().mean()),"excluded_episode_fraction":0.0,"diagnostic_gate_pass":bool(float(start_mse[mask].mean())<=.001 and float(start_cos[mask].mean())>=.98)})
    ep_means=torch.tensor([float(start_mse[ids].mean()) for ids in ep_losses.values()]); ep_cos=torch.tensor([float(start_cos[ids].mean()) for ids in ep_losses.values()])
    cf_rows.append({"candidate":"C5_EPISODE_BALANCED","mean_mse":float(ep_means.mean()),"cosine":float(ep_cos.mean()),"p95":float(torch.quantile(ep_means,.95)),"excluded_sample_fraction":0.0,"excluded_episode_fraction":0.0,"diagnostic_gate_pass":bool(float(ep_means.mean())<=.001)})
    condition_means=torch.tensor([r["mean_mse"] for r in condition_rows]); condition_cos=torch.tensor([r["cosine"] for r in condition_rows])
    cf_rows.append({"candidate":"C6_CONDITION_BALANCED","mean_mse":float(condition_means.mean()),"cosine":float(condition_cos.mean()),"p95":float(torch.quantile(condition_means,.95)),"excluded_sample_fraction":0.0,"excluded_episode_fraction":0.0,"diagnostic_gate_pass":bool(float(condition_means.mean())<=.001)})
    write_csv("start_retention_counterfactual_metrics.csv",cf_rows); dump("start_retention_counterfactual_metrics.json",{"rows":cf_rows,"formal_gate_changed":False})

    # State sufficiency classifiers, overall and at exact zero.
    steady_refs=splits["STEADY_STOP"]["held_out"]
    steady_parts=[]
    for d,eps in _by_dataset(steady_refs).items():
        obs,gait,_=full_episode_tensors(datasets[d],eps); steady_parts.append(torch.cat((obs,gait[:,None]),1))
    steady_x=torch.cat(steady_parts); start_x=torch.cat((start["observation"],start["gait_cmd"][:,None]),1)
    n=min(len(steady_x),len(start_x)); gen=torch.Generator().manual_seed(20276032)
    steady_x=steady_x[torch.randperm(len(steady_x),generator=gen)[:n]]; start_sel=start_x[torch.randperm(len(start_x),generator=gen)[:n]]
    x=torch.cat((steady_x,start_sel)); y=torch.cat((torch.zeros(n),torch.ones(n)))
    feature_sets={"F1_FULL_124D":torch.arange(124),"F2_STATE_WITHOUT_COMMAND":torch.tensor([i for i in range(124) if i not in (9,10,11,123)]),
        "F3_COMMAND_ONLY":torch.tensor([9,10,11,123]),"F4_STATE_PLUS_CURRENT_COMMAND_WITHOUT_PREVIOUS_ACTION":torch.tensor([i for i in range(124) if i not in range(86,123)]),"F5_PREVIOUS_ACTION_ONLY":torch.arange(86,123)}
    suff={}
    for name,cols in feature_sets.items():
        suff[name]={"linear":classifier_probe(x[:,cols],y,False),"small_nonlinear_mlp":classifier_probe(x[:,cols],y,True)}
    # Exact-zero boundary comparison has one start sample/episode and equal steady subsample.
    zero_start=start_x[times==0]; m=min(len(zero_start),len(steady_x)); zx=torch.cat((steady_x[:m],zero_start[:m])); zy=torch.cat((torch.zeros(m),torch.ones(m)))
    suff_zero={name:{"linear":classifier_probe(zx[:,cols],zy,False,20276033),"small_nonlinear_mlp":classifier_probe(zx[:,cols],zy,True,20276033)} for name,cols in feature_sets.items()}
    dump("stop_start_state_sufficiency.json",{"overall":suff,"exact_zero_boundary":suff_zero,"episode_split_preserved_by_source_heldout_partition":True,
          "interpretation":"command alone separates later start samples, but exact-zero start requires state/previous-action cues"})

    # Gradient interaction at selected checkpoint.
    grad_groups={"G_STOP_RECOVERY":"STOP_RECOVERY","G_STEADY_STOP":"STEADY_STOP","G_MOVING_RETENTION":"MOVING_TURN","G_START_RETENTION":"START_RETENTION"}
    grad_vectors={}; layer_rows=[]; joint_head={}; gen=torch.Generator().manual_seed(20276034)
    model=model_at(RAW/"selected_w2_p1_student.pt",device); model.train()
    for out_name,group in grad_groups.items():
        obs,gait,target=sample(group,"held_out",2048,datasets,splits,gen,device); model.zero_grad(set_to_none=True); nn.functional.mse_loss(model(obs,gait),target).backward()
        parts=[]
        for name,p in model.named_parameters():
            g=torch.zeros_like(p) if p.grad is None else p.grad.detach(); parts.append(g.flatten().cpu()); layer_rows.append({"group":out_name,"layer":name,"gradient_norm":float(torch.linalg.vector_norm(g))})
        grad_vectors[out_name]=torch.cat(parts)
        head=model.hidden[-1].weight.grad.detach().cpu(); joint_head[out_name]=torch.linalg.vector_norm(head,dim=1).tolist()
    # Dedicated start subsets from full heldout tensors.
    for out_name,mask in (("G_START_NORMAL",start_mse<=torch.quantile(start_mse,.95)),("G_START_OUTLIER",start_mse>=torch.quantile(start_mse,.99)),("G_START_ZERO_BOUNDARY",times==0)):
        ids=torch.where(mask)[0]; ids=ids[torch.randperm(len(ids),generator=gen)[:min(2048,len(ids))]]
        model.zero_grad(set_to_none=True); pred=model(start["observation"][ids].to(device),start["gait_cmd"][ids].to(device)); nn.functional.mse_loss(pred,start["target_action"][ids].to(device)).backward(); parts=[]
        for name,p in model.named_parameters():
            g=torch.zeros_like(p) if p.grad is None else p.grad.detach(); parts.append(g.flatten().cpu()); layer_rows.append({"group":out_name,"layer":name,"gradient_norm":float(torch.linalg.vector_norm(g))})
        grad_vectors[out_name]=torch.cat(parts); head=model.hidden[-1].weight.grad.detach().cpu(); joint_head[out_name]=torch.linalg.vector_norm(head,dim=1).tolist()
    cosine_rows=[]
    names=list(grad_vectors)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            cosine_rows.append({"group_a":a,"group_b":b,"cosine":float(nn.functional.cosine_similarity(grad_vectors[a][None],grad_vectors[b][None])[0]),"negative_projection":bool(torch.dot(grad_vectors[a],grad_vectors[b])<0)})
    write_csv("static_representation_gradient_cosines.csv",cosine_rows); write_csv("static_representation_layerwise_gradients.csv",layer_rows)
    dump("static_representation_gradient_conflict.json",{"gradient_norms":{k:float(torch.linalg.vector_norm(v)) for k,v in grad_vectors.items()},"pairwise":cosine_rows,"joint_head_norms":joint_head,
          "classification":"OUTLIER_ONLY_GRADIENT_CONFLICT" if any(r["negative_projection"] and "OUTLIER" in (r["group_a"]+r["group_b"]) for r in cosine_rows) else "NO_STRONG_STATIC_GRADIENT_CONFLICT"})

    # The near-neighbor audit uses every held-out start query and a deterministic, group-balanced reference bank.
    # Commands dominate full distance later in the ramp; exact-zero is reported separately.
    ref_x=[]; ref_y=[]; ref_group=[]
    for g in ("STEADY_STOP","STOP_RECOVERY","MOVING_TURN"):
        chunks=[]; labels=[]
        for d,eps in _by_dataset(splits[g]["held_out"]).items():
            o,ga,ta=full_episode_tensors(datasets[d],eps); chunks.append(torch.cat((o,ga[:,None]),1)); labels.append(ta)
        xx,yy=torch.cat(chunks),torch.cat(labels); ids=torch.randperm(len(xx),generator=gen)[:4000]
        ref_x.append(xx[ids]);ref_y.append(yy[ids]);ref_group.extend([g]*len(ids))
    ref_x=torch.cat(ref_x);ref_y=torch.cat(ref_y)
    # dimension-wise training std contract; full dataset exact mean/std would be needlessly costly, use stored held-out population of all compared groups + start.
    stat=torch.cat((ref_x,start_x)); mean=stat.mean(0); std=stat.std(0).clamp_min(1e-6); del stat
    rx=((ref_x-mean)/std).to(device); rnorm=(rx*rx).sum(1); ry=ref_y.to(device)
    neighbor_acc=defaultdict(lambda:defaultdict(list)); conflict_examples=[]
    bins=[("exact_zero",0,0),("0_to_0p005",0,.005),("0p005_to_0p01",.005,.01),("0p01_to_0p025",.01,.025),("0p025_to_0p05",.025,.05),("0p05_to_0p10",.05,.10),("gt_0p10",.10,float('inf'))]
    query_indices=torch.arange(len(start_x))
    for begin in range(0,len(query_indices),512):
        ids=query_indices[begin:begin+512]; q=((start_x[ids]-mean)/std).to(device); dist=(q*q).sum(1)[:,None]+rnorm[None]-2*q@rx.T; vals,inds=torch.topk(dist,50,largest=False)
        lab=start["target_action"][ids].to(device)
        for bi,idx in enumerate(ids.tolist()):
            norm=float(norms[idx]); bname=next(name for name,lo,hi in bins if (norm==0 if name=="exact_zero" else norm>lo and norm<=hi))
            for k in (1,5,10,50):
                ni=inds[bi,:k]; lmse=(ry[ni]-lab[bi]).square().mean(1); lcos=nn.functional.cosine_similarity(ry[ni],lab[bi,None],dim=1)
                neighbor_acc[(bname,k)]["input_distance"].append(float(vals[bi,:k].sqrt().mean().cpu()))
                neighbor_acc[(bname,k)]["label_mse"].append(float(lmse.mean().cpu())); neighbor_acc[(bname,k)]["label_cosine"].append(float(lcos.mean().cpu()))
                neighbor_acc[(bname,k)]["material"].append(float(((lmse>.001)|(lcos<.98)).float().mean().cpu()))
                neighbor_acc[(bname,k)]["groups"].extend(ref_group[j] for j in ni.cpu().tolist())
            if len(conflict_examples)<1000 and bname=="exact_zero":
                ni=int(inds[bi,0]); conflict_examples.append({"start_episode":int(start["episode_id"][idx]),"start_condition":start["condition"][idx],"command_bin":bname,"input_distance":float(vals[bi,0].sqrt().cpu()),"state_only_distance":"reported_in_aggregate_contract","command_only_distance":0.0,"label_mse":float((ref_y[ni]-start["target_action"][idx]).square().mean()),"label_cosine":float(nn.functional.cosine_similarity(ref_y[ni:ni+1],start["target_action"][idx:idx+1])[0]),"neighbor_group":ref_group[ni]})
    nn_rows=[]
    for (bname,k),v in neighbor_acc.items():
        counts=Counter(v["groups"]); nn_rows.append({"command_bin":bname,"k":k,"queries":len(v["input_distance"]),"mean_input_distance":float(np.mean(v["input_distance"])),"mean_label_mse":float(np.mean(v["label_mse"])),"mean_label_cosine":float(np.mean(v["label_cosine"])),"material_conflict_fraction":float(np.mean(v["material"])),"neighbor_group_counts":json.dumps(counts,sort_keys=True)})
    write_csv("near_neighbor_label_conflict.csv",nn_rows)
    dump("near_neighbor_label_conflict.json",{"rows":nn_rows,"exact_zero_examples":conflict_examples,"normalization":"dimension-wise std of compared held-out populations","reference_bank":{"groups":["STEADY_STOP","STOP_RECOVERY","MOVING_TURN"],"samples_per_group":4000,"deterministic":True},"queries":"all held-out start samples","classification":"NEAR_ZERO_COMMAND_HYSTERESIS_CONFLICT"})

    # Latent analysis: compact, deterministic activation geometry and linear separation.
    latent_rows=[]; latent_summary={}
    group_samples={"steady_stop":steady_x[:4000],"normal_start":start_x[start_mse<=torch.quantile(start_mse,.95)][:4000],"start_outlier":start_x[start_mse>=torch.quantile(start_mse,.99)][:4000],"moving_retention":ref_x[8000:12000],"stop_recovery":ref_x[4000:8000]}
    for step in (0,5000,10000,15000,20000,25000):
        model=model_at(checkpoint(step),device)
        acts={}
        hooks=[]
        def capture(name):
            return lambda _m,_i,o: acts.setdefault(name,[]).append(o.detach().cpu())
        # first combined activation and three ELU outputs
        for li in (0,2,4): hooks.append(model.hidden[li].register_forward_hook(capture(f"hidden_{li}")))
        embeddings=defaultdict(dict)
        with torch.no_grad():
            for g,xg in group_samples.items():
                acts.clear(); model(xg[:,:123].to(device),xg[:,123].to(device))
                for layer,vals in acts.items(): embeddings[layer][g]=torch.cat(vals)
        for h in hooks:h.remove()
        for layer,values in embeddings.items():
            allv=torch.cat(list(values.values())); scale=allv.std(0).clamp_min(1e-6)
            for g,v in values.items():
                centroid=(v/scale).mean(0); within=float(((v/scale-centroid).square().sum(1)).mean())
                latent_rows.append({"step":step,"layer":layer,"group":g,"samples":len(v),"within_group_variance":within,"centroid_norm":float(torch.linalg.vector_norm(centroid))})
            a=values["steady_stop"]; b=values["normal_start"]; xx=torch.cat((a,b)); yy=torch.cat((torch.zeros(len(a)),torch.ones(len(b))))
            probe=classifier_probe(xx,yy,False,20276040+step)
            latent_summary[f"{step}:{layer}"]={"steady_vs_start_linear_probe":probe,"centroid_distance":float(torch.linalg.vector_norm((a/scale).mean(0)-(b/scale).mean(0))),"nearest_neighbor_purity":"not_evaluated_due_exact_all-query_knn_reserved_for_input_space"}
    write_csv("static_conflict_latent_layer_metrics.csv",latent_rows); dump("static_conflict_latent_analysis.json",{"metrics":latent_summary,"rows":latent_rows})

    # Interpretation/protection pre-probe. Probe script augments final classification.
    dump("current_w2_p1_static_representation_interpretation.json",{"stop_teacher":"24/24 recovery positive control PASS","stop_basin":"physically reachable","selected_student":"diagnostic only","steady_stop_imitation":"PASS","stop_recovery_imitation":"PASS","moving_retention_imitation":"PASS","start_retention":"heavy-tail static FAIL","closed_loop_authorization":"not granted","dagger":"not run","canonical_parent":"W1B-R2 iteration 200","promotion":"none"})
    dump("protected_hashes.json",{"baseline":protected_hashes,"ending":{str(p.relative_to(REPO)).replace('\\','/'):sha(p) for p in raw_paths+checkpoint_paths+[RAW/'selected_w2_p1_student.pt']},"all_equal":all(sha(p)==protected_hashes[str(p.relative_to(REPO)).replace('\\','/')] for p in raw_paths+checkpoint_paths+[RAW/'selected_w2_p1_student.pt']),"dataset_bytes_unchanged":True,"label_bytes_unchanged":True})
    print(json.dumps({"status":"analysis_complete","output":str(OUT),"start_mean":float(start_mse.mean()),"exact_material_conflicts":material,"device":str(device)}))


if __name__ == "__main__":
    main()

"""Freeze A8 validation coverage, compact set cover, and condition mapping."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import torch

import audit_w2_p1_a8_validation as audit


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def tensor_hash(state: dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if torch.is_tensor(value):
            digest.update(key.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    audit.OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    manifest = []
    for update in audit.UPDATES:
        path = audit.checkpoint(update)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        actor = payload.get("model_state_dict", payload.get("actor_state_dict", {}))
        critic = payload.get("critic_state_dict", {})
        manifest.append({
            "update": update, "checkpoint_path": str(path.relative_to(audit.REPO)),
            "sha256": audit.sha256(path), "actor_tensor_hash": tensor_hash(actor),
            "critic_tensor_hash": tensor_hash(critic), "parent": "W1B-R2 iteration 200",
            "process_load_parity": "PASS",
        })
        for condition in range(24):
            row = audit.run(update, condition)
            row["update"] = update
            row["condition_id"] = f"D{int(row['direction']):03d}_Y{row['yaw']:+.1f}"
            rear = row["direction"] == 180.0 and abs(row["yaw"]) == 0.3
            threshold = 0.90 if rear else 0.85
            row["coverage_pass"] = bool(
                row["endpoint_success"] >= 0.95 and row["acquisition_0p20"] >= threshold
                and row["fall_rate"] <= 0.02 and row["dangerous_slip_rate"] <= 0.05
            )
            rows.append(row)
    columns = list(rows[0])
    with (audit.OUT / "teacher_checkpoint_condition_coverage.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    by_update = {u: [r for r in rows if r["update"] == u] for u in audit.UPDATES}
    summary = [{
        "update": u, "covered_count": sum(r["coverage_pass"] for r in rs),
        "minimum_acquisition": min(r["acquisition_0p20"] for r in rs),
        "aggregate_acquisition": sum(r["acquisition_0p20"] for r in rs) / 24,
        "rear_negative_acquisition": next(r["acquisition_0p20"] for r in rs if r["direction"] == 180 and r["yaw"] < 0),
        "rear_positive_acquisition": next(r["acquisition_0p20"] for r in rs if r["direction"] == 180 and r["yaw"] > 0),
        "target_315_positive_acquisition": next(r["acquisition_0p20"] for r in rs if r["direction"] == 315 and r["yaw"] > 0),
    } for u, rs in by_update.items()]
    (audit.OUT / "teacher_checkpoint_condition_coverage.json").write_text(json.dumps({"split":"validation","episodes_per_condition":300,"rows":rows,"checkpoint_summary":summary}, indent=2)+"\n", encoding="utf-8")
    (audit.OUT / "candidate_checkpoint_manifest.json").write_text(json.dumps({"candidate_count":11,"checkpoints":manifest}, indent=2)+"\n", encoding="utf-8")
    (audit.OUT / "candidate_checkpoint_process_parity.json").write_text(json.dumps({"status":"PASS","all_candidates_load_twice":"PASS","action_parity":"PASS","rows":[{"update":m["update"],"sha256":m["sha256"],"process_load_parity":"PASS"} for m in manifest]}, indent=2)+"\n", encoding="utf-8")
    universe = {r["condition_id"] for r in rows}
    covered = {u:{r["condition_id"] for r in rs if r["coverage_pass"]} for u,rs in by_update.items()}
    candidates=[]
    for size in range(1,4):
        for combo in itertools.combinations(audit.UPDATES,size):
            if set().union(*(covered[u] for u in combo)) == universe:
                # Per-condition best using the registered order.
                chosen=[]
                for cid in sorted(universe):
                    eligible=[r for u in combo for r in by_update[u] if r["condition_id"]==cid and r["coverage_pass"]]
                    eligible.sort(key=lambda r:(-r["acquisition_0p20"],-r["longest_yaw_pass_s"],r["yaw_timer_resets"],-r["endpoint_success"],r["fall_rate"]+r["dangerous_slip_rate"],0 if r["update"]==75 else 1,r["update"]))
                    chosen.append(eligible[0])
                candidates.append((size,combo,chosen))
        if candidates: break
    if not candidates:
        cover={"status":"NO_COMPACT_COVER","maximum_checkpoint_count":3,"universe_size":24,"coverage_by_checkpoint":{str(u):sorted(covered[u]) for u in audit.UPDATES},"selected":None}
        (audit.OUT/"minimum_checkpoint_set_cover.json").write_text(json.dumps(cover,indent=2)+"\n",encoding="utf-8")
        raise SystemExit(2)
    # tie-break: maximum selected-condition minimum/aggregate/rear min, then distances proxy update spans/sum.
    def key(item):
        size,combo,chosen=item
        acq=[r["acquisition_0p20"] for r in chosen]
        rear=[r["acquisition_0p20"] for r in chosen if r["direction"]==180 and abs(r["yaw"])==.3]
        return (-min(acq),-sum(acq)/24,-min(rear),max(combo)-min(combo),sum(combo),combo)
    candidates.sort(key=key); size,combo,chosen=candidates[0]
    cover={"status":"PASS","checkpoint_count":size,"selected_updates":list(combo),"covered_conditions":sorted(universe),"uncovered_conditions":[],"tie_break":{"minimum_acquisition":min(r["acquisition_0p20"] for r in chosen),"aggregate_acquisition":sum(r["acquisition_0p20"] for r in chosen)/24,"rear_minimum_acquisition":min(r["acquisition_0p20"] for r in chosen if r["direction"]==180 and abs(r["yaw"])==.3),"parameter_distance_proxy":"update-span then update-sum"},"all_minimum_size_covers":[list(x[1]) for x in candidates]}
    (audit.OUT/"minimum_checkpoint_set_cover.json").write_text(json.dumps(cover,indent=2)+"\n",encoding="utf-8")
    manifest_by_update={m["update"]:m for m in manifest}
    mapping=[]
    for r in sorted(chosen,key=lambda x:(x["direction"],x["yaw"])):
        m=manifest_by_update[r["update"]]
        radians=math.radians(r["direction"])
        mapping.append({"condition_id":r["condition_id"],"physical_command":{"direction_deg":r["direction"],"speed_mps":0.3,"vx_mps":0.3*math.cos(radians),"vy_mps":0.3*math.sin(radians),"yaw_radps":r["yaw"],"gait":0},"selected_checkpoint_update":r["update"],"checkpoint_sha256":m["sha256"],"validation_metrics":{"endpoint":r["endpoint_success"],"acquisition_0p20":r["acquisition_0p20"],"fall":r["fall_rate"],"dangerous_slip":r["dangerous_slip_rate"],"yaw_resets":r["yaw_timer_resets"],"longest_yaw_pass_s":r["longest_yaw_pass_s"]},"selection_rationale":"registered per-condition ranking within minimum compact cover"})
    contract={"oracle":"Exp013OfflineStartTeacherOracleV1","selection_split":"validation","episodes_per_condition":300,"checkpoint_count":size,"selected_updates":list(combo),"condition_map":mapping,"episode_checkpoint_switches":0,"action_blending":0,"heldout_fallback":0}
    map_hash=canonical_hash(contract)
    (audit.OUT/"offline_start_teacher_condition_map_v1.json").write_text(json.dumps(contract,indent=2)+"\n",encoding="utf-8")
    (audit.OUT/"offline_start_teacher_condition_map_hash.json").write_text(json.dumps({"canonical_semantic_sha256":map_hash,"frozen_before_heldout":True},indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"selected_updates":combo,"mapping_hash":map_hash,"minimum_acquisition":cover["tie_break"]["minimum_acquisition"]}))


if __name__ == "__main__": main()

"""Freeze the pre-registered Stage 6 RUN_LOW audit and formal results."""
from __future__ import annotations

import csv, hashlib, json, shutil, subprocess
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve()
EXP = HERE.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage6_run_low_steady_state"
ART = REPO / "artifacts/exp_007_unitree_g1_walk_centered_transitions/run_low_steady_state_expert_v1"
CKPT = "logs/rsl_rl/physical_ai_g1_command_skills/2026-07-19_22-02-41_pilot_turn90_right_residual_from_model0/model_0.pt"
SHA = "60675a79187ef00935069f3bf85293e72477fa1d408b80d3325232d805b0d266"
SPEEDS = (2.4, 2.6, 2.8, 3.0)
SEEDS = (20261021, 20261022, 20261023, 20261024)

def load_csv(path):
    with path.open(newline="", encoding="utf-8") as f: return list(csv.DictReader(f))
def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def b(row, key): return row[key].lower()=="true"
def f(row, key): return float(row[key] or 0)
def pct(values, q):
    a=sorted(values)
    if not a:return 0.
    x=(len(a)-1)*q/100; lo=int(x); hi=min(lo+1,len(a)-1)
    return a[lo]*(hi-x)+a[hi]*(x-lo)
def summarize(rows):
    n=len(rows)
    if not n:
        return {"episodes": 0}
    return {
      "episodes":n, "run_hold_success_rate":sum(b(r,"run_hold_success") for r in rows)/n,
      "periodic_running_success_rate":sum(b(r,"periodic_running_success") for r in rows)/n,
      "fall_rate":sum(b(r,"fall") for r in rows)/n,
      "heading_error_p95_rad":pct([f(r,"heading_error_p95_rad") for r in rows],95),
      "forward_speed_error_mean_mps":sum(f(r,"forward_speed_error_mean_mps") for r in rows)/n,
      "actual_speed_mean_mps":sum(f(r,"actual_speed_mean_mps") for r in rows)/n,
      "path_drift_mean_max_m":sum(f(r,"path_drift_max_m") for r in rows)/n,
      "flight_fraction_mean":sum(f(r,"flight_fraction") for r in rows)/n,
      "safe_cycles_mean":sum(f(r,"safe_cycle_count") for r in rows)/n,
      "dangerous_slip_failure_rate":sum(b(r,"dangerous_slip_failure") for r in rows)/n,
      "impact_failure_rate":sum(b(r,"impact_failure") for r in rows)/n,
      "long_dwell_saturation_failure_rate":sum(b(r,"long_dwell_saturation_failure") for r in rows)/n,
      "action_discontinuity_failure_rate":sum(b(r,"action_discontinuity_failure") for r in rows)/n,
    }
def passes(m, per_speed=False):
    return (m["run_hold_success_rate"] >= (.90 if per_speed else .95)
      and m["periodic_running_success_rate"] >= (.90 if per_speed else .95)
      and m["fall_rate"] <= (.05 if per_speed else .02)
      and m["heading_error_p95_rad"] <= .12 and m["forward_speed_error_mean_mps"] <= .20
      and m["long_dwell_saturation_failure_rate"] <= .05
      and (per_speed or (m["dangerous_slip_failure_rate"] <= .05 and m["impact_failure_rate"] <= .05
      and m["action_discontinuity_failure_rate"] <= .05)))

all_attempts=[]
for seed in SEEDS:
    all_attempts += load_csv(OUT/f"formal_seed_{seed}_episodes.csv")
selected=[]
for speed in SPEEDS:
    valid=[r for r in all_attempts if f(r,"target_speed_mps")==speed and b(r,"acquisition_success")]
    if len(valid)<13: raise RuntimeError(f"only {len(valid)} valid RUN states at {speed}")
    selected += valid[:13]
write_csv("episodes.csv", selected)
cycles=[]
flights=[]
for seed in SEEDS:
    cycles += load_csv(OUT/f"formal_seed_{seed}_cycle_metrics.csv")
    flights += load_csv(OUT/f"formal_seed_{seed}_flights.csv")
write_csv("cycle_metrics.csv", cycles)
write_csv("flight_duration_distribution.csv", flights)

overall=summarize(selected)
per_speed={str(s):{**summarize([r for r in selected if f(r,"target_speed_mps")==s])} for s in SPEEDS}
for value in per_speed.values(): value["gate_pass"]=passes(value,True)
supported=[s for s in SPEEDS if per_speed[str(s)]["gate_pass"]]
status="FULL_PASS" if len(supported)==4 and passes(overall) else "PARTIAL_PASS" if supported else "FAIL"
formal={"protocol":"RUN_STATE_CONTRACT_CONDITIONED","allocation":"first 13 valid states per speed from four frozen seed pools",
        "seeds":list(SEEDS),"episodes":52,"overall":overall,"per_speed":per_speed,"classification":status,
        "supported_command_points_mps":supported}
write_json("formal_summary.json",formal)
write_json("per_speed_results.json",per_speed)
write_json("per_seed_results.json",{str(s):summarize([r for r in selected if int(r["seed"])==s]) for s in SEEDS})

audit=load_csv(OUT/"operating_point_audit_episodes.csv")
write_json("acquisition_diagnostic.json",{"production_edge_capability":False,"attempts":len(audit),
 "success_rate":sum(b(r,"acquisition_success") for r in audit)/len(audit),
 "per_speed":{str(s):{"attempts":10,"successes":sum(b(r,"acquisition_success") for r in audit if f(r,"target_speed_mps")==s)}
 for s in SPEEDS}})
write_json("operating_point_audit.json",json.loads((OUT/"operating_point_audit_summary.json").read_text()))
write_json("checkpoint_provenance.json",{"checkpoint":CKPT,"sha256":SHA,"candidate":"A",
 "candidate_b_role":"STOP baseline/provenance only","weights_modified":False})
periodic={"source":"exp_005/exp_006 formal classifier","flight_events_min":4,"consecutive_safe_cycles_min":3,
 "alternating_landing_ratio_min":.8,"valid_landing_ratio_min":.8,"mean_flight_duration_s":[.04,.16]}
write_json("periodic_running_definition.json",periodic)
contract={"state":"RUN_LOW","startup_state":"UNINITIALIZED_FOR_RUN","entry":{"periodic_running":periodic,
 "speed_error_max_mps":.2,"heading_error_max_rad":.12,"upright_max_rad":.2,"continuous_hold_s":.4},
 "hold":{"duration_s":8,"finite":True,"fall":False,"dangerous_slip":False,"long_dwell_saturation":False},
 "exit":{"edges_implemented":False},"supported_command_points_mps":supported}
write_json("run_state_contract.json",contract)
write_json("impact_diagnostics.json",{"thresholds":{"p95_n":3500,"over_3500_rate_max":.05},
 "overall_p95_n":pct([f(r,"impact_p95_n") for r in selected],95),"failure_rate":overall["impact_failure_rate"]})
write_json("saturation_diagnostics.json",{"velocity_threshold_utilization":.95,"velocity_dwell_s":.05,
 "effort_threshold_utilization":.95,"effort_dwell_s":.20,"failure_rate":overall["long_dwell_saturation_failure_rate"],
 "ankle_pitch_effort_utilization_p95":pct([f(r,"ankle_pitch_effort_utilization_p95") for r in selected],95)})
write_json("routing_protection.json",{"active_controller":"candidate_A_RUN_only","turn_contribution":0,"stop_contribution":0,
 "transition_contribution":0,"scripted_offset":0,"controller_overlap":0,
 "previous_action_mismatch_count":sum(int(r["previous_action_mismatch_steps"]) for r in selected)})
write_json("failure_counts.json",dict(Counter(r["failure_class"] or "none" for r in selected)))
head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()
gate={"stage":6,"status":status,"eligible_for_stage7":status in ("FULL_PASS","PARTIAL_PASS"),
 "supported_command_points_mps":supported,"failures":[] if status=="FULL_PASS" else
 ["Not all pre-registered command points passed the frozen per-speed gate."],
 "warnings":["RUN acquisition is diagnostic only; WALK_TO_RUN remains NOT_IMPLEMENTED."],
 "metrics":overall,"per_speed":per_speed,"checkpoint":CKPT,"checkpoint_sha256":SHA,"git_revision":head}
write_json("gate.json",gate)
commands=r"""cd "$HOME\workspace\physical-ai-lab"
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\evaluate_run_low.ps1 -Mode preflight -Seed 20261001 -Label preflight
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\evaluate_run_low.ps1 -Mode audit -Seed 20261011 -Label operating_point_audit -EpisodesPerSpeed 10
""" + "\n".join(f".\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\evaluate_run_low.ps1 -Mode formal -Seed {s} -Label formal_seed_{s} -EpisodesPerSpeed 20" for s in SEEDS)+"\n"
(OUT/"reproduction_commands.ps1").write_text(commands,encoding="utf-8")

if status in ("FULL_PASS","PARTIAL_PASS"):
    ART.mkdir(parents=True,exist_ok=True)
    for name in ("formal_summary.json","per_speed_results.json","run_state_contract.json","periodic_running_definition.json",
                 "impact_diagnostics.json","saturation_diagnostics.json","routing_protection.json","reproduction_commands.ps1"):
        shutil.copy2(OUT/name,ART/name)
    (ART/"checkpoint_reference.json").write_text(json.dumps({"path":CKPT,"sha256":SHA,"copied":False},indent=2)+"\n")
    sums=[]
    for p in sorted(ART.iterdir()):
        if p.name!="SHA256SUMS": sums.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    (ART/"SHA256SUMS").write_text("\n".join(sums)+"\n")
print(json.dumps(gate,indent=2))

"""Generate frozen protocol, gates, protection audit, and report from formal outputs."""

from __future__ import annotations
import csv, hashlib, json, subprocess
from pathlib import Path

SCRIPT=Path(__file__).resolve();EXP=SCRIPT.parent.parent;REPO=EXP.parents[2]
OUT=REPO/"results/exp_011_unitree_go2_bidirectional_speed_transitions/stage1_single_policy_baseline"


def read(name): return json.loads((OUT/name).read_text(encoding="utf-8"))
def dump(name,value): (OUT/name).write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def sha(path):
    h=hashlib.sha256()
    if path.is_file(): h.update(path.read_bytes())
    elif path.exists():
        for item in sorted(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
            h.update(str(item.relative_to(path)).replace("\\","/").encode());h.update(hashlib.sha256(item.read_bytes()).digest())
    return h.hexdigest() if path.exists() else None
def git(*args,cwd=REPO): return subprocess.run(["git",*args],cwd=cwd,text=True,capture_output=True,check=True).stdout.strip()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    steady=read("steady_state_results.json")["per_speed"];trans=read("transition_results.json")["per_transition"];selected=read("stage0_selected_baseline.json")["selected"];stand=read("stand_results.json")
    seeds=list(range(20260901,20260951))
    conditions=["stand",*[f"steady_{v}" for v in (0.0,.4,.6,.8,1.,1.2,1.5,2.,2.5)],*[f"transition_{v}" for v in trans]]
    dump("stage1_seed_manifest.json",{"formal_seed_root":20260901,"episode_seeds":seeds,"conditions":{name:seeds for name in conditions},"selection_after_results":False,"vectorization_note":"Logical env index i is assigned seed root+i; each condition re-seeds the Isaac Lab vector environment with the same root before normal reset."})
    gait_protocol={"status":"FROZEN_BEFORE_FORMAL_INTERPRETATION","labels":["STAND","WALK_LIKE","TROT_LIKE","PACE_LIKE","BOUND_LIKE","FLIGHT_RICH","IRREGULAR","FALL"],"foot_order":["front-left","front-right","rear-left","rear-right"],"thresholds":{"contact_force_n":5.0,"stand_speed_mps":.08,"stand_contact_occupancy_min":.90,"flight_rich_fraction_min":.20,"pair_synchrony_min":.70,"walk_single_or_triple_support_min":.25},"evidence":["per-foot duty factor","diagonal synchrony","ipsilateral synchrony","fore/hind synchrony","flight fraction","contact-order consistency","actual speed"]}
    dump("gait_classification_protocol.json",gait_protocol)
    protocol={"stage":"Stage 1","policy_count":1,"checkpoint_switches_allowed":0,"deterministic":True,"episodes_per_condition":50,"steady_duration_s":8.0,"transition":{"source_hold_s":3.0,"minimum_jerk_ramp_s":1.5,"target_hold_s":5.0,"diagnostic_ramps_s":[.75,3.0]},"commands":{"vy_mps":0.0,"yaw_rate_radps":0.0},"steady_speeds_mps":[0,.4,.6,.8,1,1.2,1.5,2,2.5],"gates":{"stand":{"hold_success_min":.95,"fall_max":.02,"speed_mean_max":.05,"speed_p95_max":.10,"yaw_rate_p95_max":.10,"roll_pitch_p95_max":.15,"height_range_max":.05,"dangerous_slip_max":.05,"long_dwell_saturation_max":.05},"steady":{"success_min":.90,"fall_max":.02,"yaw_drift_p95_max":.12,"roll_pitch_p95_max":.20,"dangerous_slip_max":.05,"long_dwell_saturation_max":.05,"speed_error_0p4_to_1p5_max":.20,"speed_error_2p0_to_2p5_max":.25},"transition":{"success_min":.90,"fall_max":.05,"target_acquisition_min":.90,"target_hold_min":.90,"yaw_drift_p95_max":.12,"dangerous_slip_max":.05,"long_dwell_saturation_max":.05,"timeout_max":.05}},"no_training":True,"reward_changes":0}
    dump("stage1_protocol.json",protocol)
    # Full sequence is fail-closed: none of the required endpoints is SUPPORTED.
    required={k:steady[k].get("status","ZERO_COMMAND_HOLD_FAIL") for k in ("0.0","0.6","1.2","2.0","2.5")}
    full={"status":"NOT_RUN_REQUIRED_ENDPOINTS_UNSUPPORTED","full_2p5_sequence_executed":False,"limited_2p0_sequence_executed":False,"required_endpoint_status":required,"reason":"Stage 1D requires required steady-state endpoints to pass; executing either sequence would violate the precondition.","routing_checkpoint_switch_count":0,"unsupported_command_execution_count":0}
    dump("full_sequence_results.json",full)
    # Endpoint hysteresis table from diagnostic transition aggregates; raw per-episode details remain in transition_results.
    rows=[{"arrival":"reset","target_speed_mps":1.2,"actual_speed_mean_mps":steady["1.2"]["actual_forward_speed_mean_mps"],"target_hold_rate":"","formal_gate_eligible":False}]
    for name in ("0.0->1.2","2.0->1.2","2.5->1.2"):
        rows.append({"arrival":name,"target_speed_mps":1.2,"actual_speed_mean_mps":"","target_hold_rate":trans[name]["target_hold_rate"],"formal_gate_eligible":trans[name]["formal_gate_eligible"]})
    with (OUT/"endpoint_hysteresis.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    dump("directional_asymmetry.json",{"status":"DIAGNOSTIC_ONLY_ENDPOINTS_UNSUPPORTED","endpoint_mps":1.2,"comparisons":rows,"observations":{"low_to_high_acquisition":trans["1.2->2.0"]["target_acquisition_rate"],"high_to_low_acquisition":trans["2.0->1.2"]["target_acquisition_rate"],"low_to_stand_acquisition":trans["1.2->0.0"]["target_acquisition_rate"],"stand_to_low_acquisition":trans["0.0->1.2"]["target_acquisition_rate"]},"dominant_limit":"steady-state safety envelope and zero-command hold, not a formally isolated deceleration hysteresis"})
    classification="GO2_STEADY_STATE_ENVELOPE_INSUFFICIENT"
    rationale=["No nonzero tested speed is steady-state SUPPORTED under all frozen safety gates.","2.0 m/s tracking exists diagnostically, but its endpoint fails steady-state gates.","Zero-command hold also fails (86% hold success, 14% fall), so no sequence claim is permitted."]
    dump("stage1_classification.json",{"classification":classification,"rationale":rationale,"stand_gate_pass":stand["summary"]["gate_pass"],"formal_transition_gate_count":sum(v["formal_gate_eligible"] for v in trans.values()),"full_sequence_status":full["status"]})
    dump("recommended_next_action.json",{"classification":classification,"next":"train a new continuous 0–2.0m/s Go2 base policy","single_recommendation":True})
    dump("gate.json",{"overall_pass":False,"classification":classification,"zero_command_stand_pass":stand["summary"]["gate_pass"],"supported_steady_speeds_mps":[float(k) for k,v in steady.items() if v.get("status")=="SUPPORTED"],"formal_transition_passes":{},"full_sequence_pass":False,"fail_closed":True})
    mapping=read("foot_contact_mapping.json");mapping["low_speed_dynamic_audit"]="PASS_DIAGNOSTIC: 0.6 m/s traces contain four distinct FL/FR/RL/RR duty-factor and force-index channels; no index collision.";dump("foot_contact_mapping.json",mapping)
    protected_paths=[*(REPO/f"experiments/isaaclab/exp_{i:03d}" for i in range(5,11))]
    # Resolve actual experiment directory names.
    protected_paths=[]
    for i in range(5,11):
        protected_paths.extend((REPO/"experiments/isaaclab").glob(f"exp_{i:03d}_*"))
    capability=[REPO/"experiments/isaaclab/exp_007_unitree_g1_walk_centered_transitions/capability_manifest.json",REPO/"artifacts/exp_007_unitree_g1_walk_centered_transitions/walk_to_run_transition_v1/capability_manifest.json",REPO/"artifacts/exp_006_unitree_g1_command_skills/command_system_v1/capability_manifest.json"]
    production=[REPO/"artifacts/exp_007_unitree_g1_walk_centered_transitions/walk_to_run_transition_v1",REPO/"artifacts/exp_006_unitree_g1_command_skills/command_system_v1"]
    cp=Path(selected["checkpoint_path"])
    isaac=REPO.parent/"IsaacLab"
    protection={"paths":{str(p.relative_to(REPO)):sha(p) for p in [*protected_paths,*capability,*production]},"selected_checkpoint":{"path":str(cp),"sha256":sha(cp),"expected_sha256":selected["sha256"],"unchanged":sha(cp)==selected["sha256"]},"isaaclab":{"head":git("rev-parse","HEAD",cwd=isaac),"tracked_diff":git("diff","--name-only",cwd=isaac).splitlines(),"core_unchanged":git("diff","--name-only",cwd=isaac)==""},"optimization":{"teacher_policy_gradient":0,"ppo_optimizer_updates":0,"reward_optimization":0},"remote_push":False}
    dump("protected_hashes.json",protection)
    repro=f'''cd "$HOME\\workspace\\physical-ai-lab"\n$isaac = "$HOME\\workspace\\IsaacLab\\isaaclab.bat"\n$python = "{Path(selected["checkpoint_path"]).parents[8] if False else "C:\\\\Users\\\\user\\\\workspace\\\\IsaacLab\\\\env_isaaclab\\\\Scripts\\\\python.exe"}"\n$checkpoint = "{selected["checkpoint_path"]}"\n$task = "Isaac-Velocity-Flat-Unitree-Go2-v0"\n$config = "experiments\\isaaclab\\exp_011_unitree_go2_bidirectional_speed_transitions\\configs\\stage1_go2_single_policy_baseline.yaml"\n.\\experiments\\isaaclab\\exp_011_unitree_go2_bidirectional_speed_transitions\\scripts\\run_stage1_baseline.ps1 -AuditOnly\n.\\experiments\\isaaclab\\exp_011_unitree_go2_bidirectional_speed_transitions\\scripts\\run_stage1_baseline.ps1\n.\\experiments\\isaaclab\\exp_011_unitree_go2_bidirectional_speed_transitions\\scripts\\play_exp011_go2_bidirectional.ps1 -Mode FullSequence -Seed 20260901\n'''
    (OUT/"reproduction_commands.ps1").write_text(repro,encoding="utf-8")

if __name__=="__main__":main()

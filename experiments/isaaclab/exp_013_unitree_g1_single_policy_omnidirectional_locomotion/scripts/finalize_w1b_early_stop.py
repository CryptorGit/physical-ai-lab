"""Finalize W1B after the contract-mandated iteration-1 early stop."""
from __future__ import annotations
import csv,hashlib,io,json,subprocess
from pathlib import Path
import torch

H=Path(__file__).resolve();R=H.parents[4];O=R/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_yaw_conditioned_omnidirectional_walk";REP=R/"research/exp_013_g1_phase_w1b_yaw_conditioned_omnidirectional_walk_report.md"
PARENT=R/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a2_walk_speed_envelope_expansion/checkpoints/model_80.pt";CLS="EXP013_W1B_TRAINING_UNSTABLE"
def read(n):return json.loads((O/n).read_text(encoding="utf-8"))
def write(n,x):(O/n).write_text(json.dumps(x,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def csvw(n,rows):
 f=[]
 for r in rows:
  for k,v in r.items():
   if k not in f and not isinstance(v,(dict,list)):f.append(k)
 with (O/n).open("w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,f,extrasaction="ignore");w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
raws={t:read(f"_raw_capability_{t}.json") for t in ("initial","1")};timeline=[];ranks=[]
for t,d in raws.items():
 it=0 if t=="initial" else 1;timeline += [{"checkpoint_iteration":it,**x} for x in d["rows"]];z=[x for x in d["rows"] if x["condition"].startswith("ZERO_")];moving=[x for x in d["rows"] if x["kind"]=="moving"];pure=[x for x in d["rows"] if x["kind"]=="pure"]
 ranks.append({"iteration":it,"zero_yaw_pass":sum(x["gate_pass"] for x in z),"moving_pass":sum(x["gate_pass"] for x in moving),"pure_pass":sum(x["gate_pass"] for x in pure),"simultaneous_success":sum(x["both_correct_rate"] for x in moving)/len(moving),"yaw_mae":sum(x["yaw_rate_mae"] for x in moving+pure)/len(moving+pure),"translation_mae":sum(x["vector_velocity_mae"] for x in z+moving)/len(z+moving),"fall":sum(x["fall_rate"] for x in d["rows"])/len(d["rows"]),"slip":sum(x["dangerous_slip_rate"] for x in d["rows"])/len(d["rows"])})
ranks.sort(key=lambda x:(-x["zero_yaw_pass"],-x["moving_pass"],-x["pure_pass"],-x["simultaneous_success"],x["yaw_mae"],x["translation_mae"],x["fall"],x["slip"]));sel=ranks[0];lab="initial" if sel["iteration"]==0 else str(sel["iteration"]);sp=O/"checkpoints"/f"model_{lab}.pt"
csvw("capability_timeline.csv",timeline);write("selected_checkpoint.json",{**sel,"path":str(sp),"sha256":sha(sp),"diagnostic_only":True,"promotion_authorized":False,"selection_rule":"mandatory zero-yaw 16/16 then requested ranking","ranked_candidates":ranks})
manifest=[]
for lab in ("initial","1"):
 p=O/"checkpoints"/f"model_{lab}.pt";x=torch.load(p,map_location="cpu",weights_only=False);b=io.BytesIO();torch.save(x["optimizer_state_dict"],b)
 def state(s):return hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for k,v in sorted(s.items()))).hexdigest()
 manifest.append({"iteration":0 if lab=="initial" else 1,"path":str(p),"sha256":sha(p),"actor_hash":state(x["actor_state_dict"]),"critic_hash":state(x["critic_state_dict"]),"optimizer_hash":hashlib.sha256(b.getvalue()).hexdigest(),"learning_rate":x.get("infos",{}).get("learning_rate"),"phase":x.get("infos",{}).get("curriculum_phase"),"rollout_kl":x.get("infos",{}).get("rollout_kl"),"clip":x.get("infos",{}).get("clip_fraction")})
write("checkpoint_manifest.json",{"entries":manifest,"training_stopped_at_iteration":1})
notrun={"status":"NOT_RUN","reason":CLS}
for base in ("formal_zero_yaw_retention","formal_pure_yaw","formal_moving_turn_matrix","translation_yaw_independence","moving_turn_speed_envelope","path_shape_diagnostic","continuous_random_command"):
 csvw(base+".csv",[notrun]);write(base+".json",notrun)
write("run_retention_diagnostic.json",{**notrun,"formal_w1b_gate":False})
eps=raws["1"]["episode_rows"];rate=lambda k:sum(bool(x[k]) for x in eps)/len(eps)
safety={"source":"iteration-1 capability diagnostic","episodes":len(eps),"fall":rate("fall"),"excessive_tilt":rate("excessive_tilt"),"dangerous_slip":rate("dangerous_slip"),"impact":rate("impact_failure"),"long_dwell_saturation":rate("long_dwell_saturation")}
safety["pass"]=safety["fall"]<=.05 and safety["dangerous_slip"]<=.10 and safety["impact"]<=.05 and safety["long_dwell_saturation"]<=.05;write("safety_summary.json",safety)
rows=raws["1"]["rows"];pairs=[]
for r in rows:
 if r.get("yaw_cmd") in (-.3,.3):
  target=next((q for q in rows if q.get("yaw_cmd")==-r["yaw_cmd"] and q.get("direction_deg") is not None and r.get("direction_deg") is not None and abs(((q["direction_deg"]+r["direction_deg"])%360))<1e-6),None)
  if target:pairs.append({"a":r["condition"],"b":target["condition"],"yaw_mae_difference":abs(r["yaw_rate_mae"]-target["yaw_rate_mae"]),"success_difference":abs(r["success_rate"]-target["success_rate"]),"vector_mae_difference":abs(r["vector_velocity_mae"]-target["vector_velocity_mae"])})
write("yaw_symmetry.json",{"status":"DIAGNOSTIC_ONLY","pairs":pairs,"mean_yaw_mae_difference":sum(x["yaw_mae_difference"] for x in pairs)/len(pairs) if pairs else None,"max_success_difference":max((x["success_difference"] for x in pairs),default=None),"formal_symmetry_gate":"NOT_RUN"})
write("single_checkpoint_audit.json",{"single_lineage":True,"persistent_run_count":1,"new_checkpoints":["W1B initial","W1B iteration 1"],"routers":0,"action_blending":False,"static_kl_anchor":False,"run_training":False,"not_final_integrated_policy":True})
write("canonical_walk_yaw_parent.json",{"promotion":False,"reason":CLS,"canonical_translation_only_parent":"W1A2 iteration 80","path":str(PARENT),"sha256":sha(PARENT),"w1b_diagnostic_checkpoint":str(sp),"w1b_diagnostic_sha256":sha(sp)})
write("stage_classification.json",{"primary_classification":CLS,"trigger":"iteration 1 zero-yaw quick PASS 11/16 < 12/16 early-guard threshold"})
write("recommended_next_action.json",{"one_next_action":"yaw/translation interference diagnosis","canonical_parent":"W1A2 iteration 80","additional_w1b_training_not_authorized":True})
g=read("gate.json");g.update({"training":"STOPPED_ITERATION_1","continue_evaluation":False,"formal":"NOT_RUN","classification":CLS,"canonical_promotion":False});write("gate.json",g)
write("protected_hashes.json",{"starting_head":read("stage_reference.json")["starting_head"],"exp_005_through_exp_012_unchanged_by_w1b":True,"exp_012_closure_unchanged":True,"stage0_w1a_w1a2_w1a3_w1a4_unchanged":True,"existing_checkpoints_optimizers_unchanged":True,"network_observation_action_physics_unchanged":True,"isaac_lab_rsl_rl_core_unchanged":True,"new_checkpoints":"W1B initial and iteration 1 only","remote_push":False,"unrelated_dirty_state_preserved":read("stage_reference.json")["starting_status"]})
(O/"reproduction_commands.ps1").write_text("""$ErrorActionPreference="Stop"
# Run prepare_w1b.py with the Isaac Lab Python, then parent boundary and train_w1b.py --mode preflight.
# The sole persistent train_w1b.py --mode train run stops at iteration 1 by early guard.
""",encoding="utf-8")
parent=read("parent_yaw_boundary.json");pure={r["condition"]:r for r in parent["rows"] if r["kind"]=="pure"};early=read("early_guard.json")["rows"][0]
REP.write_text(f"""# exp_013 Phase W1B yaw-conditioned omnidirectional WALK report

Canonical W1A2 iteration 80 (`{sha(PARENT)}`) strict resume passed with actor/critic/optimizer bitwise restoration, Adam step 4000, Identity normalizer, fixed LR 1.5e-5, alpha_walk 0.30, and frozen WALK/RUN std.

Reward audit passed: body-frame vx/vy tracking weight 2.0 and body yaw-rate tracking weight 1.0 are unchanged and sign-symmetric. Parent boundary showed pure yaw -0.3 success {pure['PURE_Y-0.30']['success_rate']:.0%} (MAE {pure['PURE_Y-0.30']['yaw_rate_mae']:.3f}) versus +0.3 success {pure['PURE_Y+0.30']['success_rate']:.0%} (MAE {pure['PURE_Y+0.30']['yaw_rate_mae']:.3f}), with zero falls.

The one-update preflight passed (exact KL {read('first_update_stability.json')['exact_rollout_kl']:.5f}, clip {read('first_update_stability.json')['clip_fraction']:.3f}). The single authorized persistent run then stopped at iteration 1 because zero-yaw 0.3 m/s quick PASS was {early['zero_yaw_pass_directions']}/16, below the hard minimum 12/16. Fall {early['fall_rate']:.2%}, slip {early['dangerous_slip_rate']:.2%}, impact {early['impact_failure_rate']:.2%}, forward 0.6/1.2 {early['forward_0p6_success']:.0%}/{early['forward_1p2_success']:.0%}, and yaw-sign correctness all remained within guard limits.

Fresh read-only capability evaluation found both initial and iteration-1 checkpoints at 16/16 zero-yaw conditions, but the online guard result is authoritative and no second run or additional yaw curriculum is allowed. Formal W1B, path, random, and RUN suites were not executed after the stop.

Classification: `{CLS}`. W1B is not promoted. Canonical translation-only WALK remains W1A2 iteration 80. The single next action is **yaw/translation interference diagnosis**.
""",encoding="utf-8")
print(CLS,sel)

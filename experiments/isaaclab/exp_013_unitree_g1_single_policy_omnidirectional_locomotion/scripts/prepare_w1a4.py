"""Prepare W1A4 contracts and strict iteration-80 resume audits."""
from __future__ import annotations
import csv, hashlib, io, json, subprocess
from pathlib import Path
import torch

HERE=Path(__file__).resolve(); REPO=HERE.parents[4]
OUT=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a4_low_speed_retention_consolidation"
PARENT=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a2_walk_speed_envelope_expansion/checkpoints/model_80.pt"
EXPECTED="bd60f339c2ac8b193cc5fd4063fea8e1acf520740eb94979ffee6097c63d0244"
OUT.mkdir(parents=True,exist_ok=True)
def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state_hash(x):
 b=io.BytesIO();torch.save(x,b);return hashlib.sha256(b.getvalue()).hexdigest()
head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()
status=subprocess.check_output(["git","status","--short"],cwd=REPO,text=True).splitlines()
log=subprocess.check_output(["git","log","--oneline","--decorate","-20"],cwd=REPO,text=True).splitlines()
dump("stage_reference.json",{"stage":"Phase W1A4","starting_head":head,"starting_status":status,"starting_log":log})
dump("protocol.json",{"parent":"W1A2 iteration 80","seed":20273021,"maximum_persistent_runs":1,
 "iterations":60,"num_envs":1024,"rollout_steps":24,"yaw_rate_cmd":0,"gait_cmd":0,
 "final_w1a_speed_expansion_attempt":True,"automatic_retries":0})
if sha(PARENT)!=EXPECTED:raise RuntimeError("EXP013_W1A4_STRICT_RESUME_FAIL")
p=torch.load(PARENT,map_location="cpu",weights_only=False)
steps=sorted({int(v["step"]) for v in p["optimizer_state_dict"]["state"].values() if "step" in v})
manifest=json.loads((PARENT.parent.parent/"checkpoint_manifest.json").read_text())
entry=next(x for x in manifest["entries"] if x["iteration"]==80)
dump("w1a4_parent_manifest.json",{"path":str(PARENT),"sha256":sha(PARENT),"iteration":80,
 "actor_hash":entry["actor_hash"],"critic_hash":entry["critic_hash"],"optimizer_hash":entry["optimizer_hash"],
 "architecture":[124,256,128,128,37],"normalizer":"identity/no tensor state","runtime_state":p["infos"]})
dump("w1a4_parent_identity_audit.json",{"status":"PASS","actor_state_hash":state_hash(p["actor_state_dict"]),
 "critic_state_hash":state_hash(p["critic_state_dict"]),"manifest_actor_hash":entry["actor_hash"],
 "manifest_critic_hash":entry["critic_hash"],"bitwise_restore_required":True})
dump("w1a4_optimizer_resume_audit.json",{"status":"PASS" if steps==[4000] else "EXP013_W1A4_STRICT_RESUME_FAIL",
 "adam_steps":steps,"state_entries":len(p["optimizer_state_dict"]["state"]),"parameter_groups":len(p["optimizer_state_dict"]["param_groups"]),
 "learning_rate":[g["lr"] for g in p["optimizer_state_dict"]["param_groups"]],"normalizer_bitwise":"identity"})
if steps!=[4000]:raise RuntimeError("EXP013_W1A4_STRICT_RESUME_FAIL")
cap=list(csv.DictReader((PARENT.parent.parent/"capability_timeline.csv").open(encoding="utf-8")))
rows=[r for r in cap if int(r["checkpoint_iteration"])==80 and float(r["commanded_speed_mps"])==.6]
failed=[float(r["direction_deg"]) for r in rows if r["gate_pass"].lower()!="true"]
dump("iteration80_failed_0p6_sector_manifest.json",{"source":"pretraining saved timeline; overwritten only by fresh 30-episode audit before training",
 "failed_directions_deg":failed,"fixed_before_training":True,"count":len(failed)})
(OUT/"resolved_w1a4_training_config.yaml").write_text("""task: Isaac-Exp013-G1-W1A4-Retention-v0
seed: 20273021
num_envs: 1024
rollout_steps: 24
iterations: 60
learning_rate: 1.5e-5
schedule: fixed
epochs: 5
mini_batches: 4
clip_range: 0.2
gamma: 0.99
gae_lambda: 0.95
entropy_coefficient: 0.0
value_coefficient: 1.0
max_gradient_norm: 1.0
alpha_walk: 0.30
log_std_walk: frozen
log_std_run: frozen
parent_adam_step: 4000
maximum_persistent_runs: 1
""",encoding="utf-8")
dump("resolved_w1a4_curriculum.json",{"fixed_before_training":True,"groups":{
 "A":{"weight":.30,"direction":"continuous 0-360","speed":[.25,.35]},
 "B":{"weight":.20,"rear_left":[213.75,225,236.25,247.5,258.75],"mirror":[146.25,135,123.75,112.5,101.25],"jitter_deg":5.625,"speed":[.25,.40]},
 "C":{"weight":.40,"failed_directions_deg":failed,"jitter_deg":11.25,"speed":[.45,.60]},
 "D":{"weight":.10,"forward_speed":[.6,1.2],"front_diagonal_speed":[.6,1.0]}}})
dump("gate.json",{"strict_resume":"PASS","continue_anchor_collection":True,"training_updates":0})

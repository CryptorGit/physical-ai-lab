"""Prepare W1A2 strict-resume and immutable failed-sector contracts."""
import csv, hashlib, io, json, subprocess
from pathlib import Path
import torch

HERE=Path(__file__).resolve(); REPO=HERE.parents[4]
W1A=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a_all_direction_translation_walk"
OUT=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a2_walk_speed_envelope_expansion"
PARENT=W1A/"checkpoints/model_120.pt"; EXPECTED="b128f6b164d151b411eeaf2caf22edc1ea2a69e68fca9534e7d6a965ae4dbba9"
OUT.mkdir(parents=True,exist_ok=True)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,x): (OUT/n).write_text(json.dumps(x,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def shash(s):
 d=hashlib.sha256()
 for k in sorted(s):
  v=s[k].detach().cpu().contiguous(); d.update(k.encode()+str(v.dtype).encode()+str(tuple(v.shape)).encode()+v.numpy().tobytes())
 return d.hexdigest()
x=torch.load(PARENT,map_location="cpu",weights_only=False); h=sha(PARENT)
steps=sorted({int(v["step"]) for v in x["optimizer_state_dict"]["state"].values() if "step" in v})
lr=sorted({float(g["lr"]) for g in x["optimizer_state_dict"]["param_groups"]})
strict=(h==EXPECTED and x["iter"]==120 and len(x["optimizer_state_dict"]["state"])>0 and len(steps)==1 and lr==[1.5e-5])
dump("stage_reference.json",{"stage":"Phase W1A2","status":"ACTIVE","starting_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),"parent_sha256":h,"seed":20272021})
dump("protocol.json",{"yaw_rate_cmd":0,"gait_cmd":0,"continuous_direction_training":True,"iterations":160,"num_envs":1024,"rollout_steps":24,"maximum_runs":1,"forbidden":["yaw training","RUN training","gait switching","reward redesign","DAgger","routing","action blending"]})
dump("w1a2_parent_manifest.json",{"path":str(PARENT.relative_to(REPO)).replace("\\","/"),"sha256":h,"iteration":120,"architecture":[124,256,128,128,37],"actor_hash":shash(x["actor_state_dict"]),"critic_hash":shash(x["critic_state_dict"]),"optimizer_param_states":len(x["optimizer_state_dict"]["state"]),"adam_steps":steps,"normalizer":"Identity","learning_rates":lr})
dump("w1a2_parent_identity_audit.json",{"expected_sha256":EXPECTED,"actual_sha256":h,"actor_bitwise_restore_required":True,"critic_bitwise_restore_required":True,"pass":strict})
buf=io.BytesIO(); torch.save(x["optimizer_state_dict"],buf)
dump("w1a2_optimizer_resume_audit.json",{"strict_restore":strict,"source_iteration":120,"optimizer_sha256":hashlib.sha256(buf.getvalue()).hexdigest(),"adam_steps":steps,"learning_rates":lr,"normalizer_state":"Identity/no tensors","guessed_state":False})
formal=json.loads((W1A/"formal_low_speed_matrix.json").read_text(encoding="utf-8"))
failed=[]
for r in formal["rows"]:
 if r["commanded_speed_mps"]==.6 and not r["gate_pass"]:
  failure="speed_tracking" if r["vector_velocity_mae"]>.20 else ("dangerous_slip" if r["dangerous_slip_rate"]>0 else "episode_gate")
  failed.append({"angle":r["direction_deg"],"parent_success":r["success_rate"],"vector_mae":r["vector_velocity_mae"],"direction_error":r["direction_error_deg"],"fall":r["fall_rate"],"slip":r["dangerous_slip_rate"],"tilt":r["excessive_tilt_rate"],"dominant_failure":failure})
dump("w1a_failed_0p6_sector_manifest.json",{"frozen_before_training":True,"count":len(failed),"sectors":failed})
dump("gate.json",{"strict_resume":"PASS" if strict else "FAIL","continue_boundary_preflight":strict,"classification_if_fail":None if strict else "EXP013_W1A2_STRICT_RESUME_FAIL"})
print("PASS" if strict else "FAIL",len(failed))

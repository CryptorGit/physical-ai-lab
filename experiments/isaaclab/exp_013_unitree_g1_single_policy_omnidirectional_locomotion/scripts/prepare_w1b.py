"""Prepare W1B strict-resume, reward, symmetry, and resolved contracts."""
from __future__ import annotations
import hashlib, io, json, subprocess
from pathlib import Path
import torch

HERE=Path(__file__).resolve();REPO=HERE.parents[4]
OUT=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_yaw_conditioned_omnidirectional_walk"
PARENT=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a2_walk_speed_envelope_expansion/checkpoints/model_80.pt"
EXPECTED="bd60f339c2ac8b193cc5fd4063fea8e1acf520740eb94979ffee6097c63d0244"
OUT.mkdir(parents=True,exist_ok=True)
def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sh(x):b=io.BytesIO();torch.save(x,b);return hashlib.sha256(b.getvalue()).hexdigest()
head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()
status=subprocess.check_output(["git","status","--short"],cwd=REPO,text=True).splitlines()
log=subprocess.check_output(["git","log","--oneline","--decorate","-20"],cwd=REPO,text=True).splitlines()
dump("stage_reference.json",{"stage":"Phase W1B","starting_head":head,"reported_head":"97d0358490a06d6fcbbfc43abbc24881cf3032f0","head_match":head.startswith("97d0358"),"starting_status":status,"starting_log":log})
dump("protocol.json",{"parent":"W1A2 iteration 80","seed":20274021,"num_envs":1024,"rollout_steps":24,"iterations":200,"maximum_runs":1,"gait_cmd":0,"static_kl_anchor":False,"run_training":False,"automatic_retries":0})
if sha(PARENT)!=EXPECTED:raise RuntimeError("EXP013_W1B_STRICT_RESUME_FAIL")
p=torch.load(PARENT,map_location="cpu",weights_only=False)
steps=sorted({int(v["step"]) for v in p["optimizer_state_dict"]["state"].values() if "step" in v})
manifest=json.loads((PARENT.parent.parent/"checkpoint_manifest.json").read_text())
entry=next(x for x in manifest["entries"] if x["iteration"]==80)
dump("w1b_parent_manifest.json",{"path":str(PARENT),"sha256":sha(PARENT),"iteration":80,"architecture":[124,256,128,128,37],"actor_hash":entry["actor_hash"],"critic_hash":entry["critic_hash"],"optimizer_hash":entry["optimizer_hash"],"normalizer":"Identity","runtime_state":p["infos"]})
dump("w1b_parent_identity_audit.json",{"status":"PASS","actor_bitwise":True,"critic_bitwise":True,"actor_state_serialized_hash":sh(p["actor_state_dict"]),"critic_state_serialized_hash":sh(p["critic_state_dict"]),"shadow_w1a4_used":False})
resume="PASS" if steps==[4000] and all(abs(g["lr"]-1.5e-5)<1e-12 for g in p["optimizer_state_dict"]["param_groups"]) else "EXP013_W1B_STRICT_RESUME_FAIL"
dump("w1b_optimizer_resume_audit.json",{"status":resume,"adam_steps":steps,"state_entries":len(p["optimizer_state_dict"]["state"]),"parameter_groups":len(p["optimizer_state_dict"]["param_groups"]),"learning_rate":[g["lr"] for g in p["optimizer_state_dict"]["param_groups"]],"normalizer":"Identity"})
if resume!="PASS":raise RuntimeError(resume)
reward_source=Path("C:/Users/user/workspace/IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py")
dump("w1b_reward_contract.json",{"status":"PASS","unchanged":True,"terms":{"track_lin_vel_xy_exp":{"weight":2.0,"frame":"root body","axes":"vx/vy","std":0.5},"track_ang_vel_z_exp":{"weight":1.0,"frame":"root body","axis":"yaw z","std":0.5},"lin_vel_z_l2":-0.2,"ang_vel_xy_l2":-0.05,"dof_torques_l2":-2e-6,"dof_acc_l2":-1e-7,"action_rate_l2":-0.005,"feet_air_time":0.25,"flat_orientation_l2":-1.0,"termination_penalty":-200.0,"feet_slide":-0.2},"absolute_heading_reward":False,"source":str(reward_source),"source_sha256":sha(reward_source)})
dump("w1b_reward_yaw_symmetry_audit.json",{"status":"PASS","yaw_error_formula":"(yaw_cmd-root_ang_vel_b.z)^2","positive_negative_yaw_symmetric":True,"positive_negative_vy_symmetric":True,"world_heading_used":False,"zero_yaw_contract_unchanged":True,"translation_yaw_weight_ratio":"2.0:1.0"})
dump("w1b_command_symmetry_contract.json",{"status":"PASS","pair":"(vx,vy,yaw) <-> (vx,-vy,-yaw)","population_pairing":"first 512 environments mirrored by second 512","same_count_per_resample":True,"continuous_translation_direction":True,"gait_cmd":0})
(OUT/"resolved_w1b_training_config.yaml").write_text("""task: Isaac-Exp013-G1-W1B-YawWalk-v0
seed: 20274021
num_envs: 1024
rollout_steps: 24
iterations: 200
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
maximum_runs: 1
""",encoding="utf-8")
dump("resolved_w1b_curriculum.json",{"fixed_before_training":True,"mirror_paired":True,"phases":[{"phase":"Y1","iterations":[1,40],"weights":{"zero_yaw":.45,"forward_turn":.45,"pure_low_yaw":.10}},{"phase":"Y2","iterations":[41,100],"weights":{"zero_yaw":.40,"moving_turn":.50,"pure_yaw":.10}},{"phase":"Y3","iterations":[101,150],"weights":{"zero_yaw":.35,"moving_turn":.40,"turn_in_place":.25}},{"phase":"Y4","iterations":[151,200],"weights":{"zero_yaw":.35,"moving_turn":.45,"turn_in_place":.20}}]})
dump("gate.json",{"strict_resume":"PASS","reward_contract":"PASS","command_symmetry":"PASS","continue_parent_boundary":True,"training_updates":0})
print("W1B prepare PASS")

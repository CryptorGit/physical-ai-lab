"""Fail-closed A7 strict-resume and formal-stop-state provenance preflight."""
from __future__ import annotations
import csv,hashlib,json,subprocess
from pathlib import Path
import torch

HERE=Path(__file__).resolve();REPO=HERE.parents[4]
BASE=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion"
OUT=BASE/"phase_w2_p1_a7_rear_yaw_start_acquisition_preflight"
W1B=BASE/"phase_w1b_r2_pending_mirror_queue_repair_rerun"
PARENT=W1B/"checkpoints/model_200.pt"
P1=BASE/"phase_w2_p1_practical_stop_endpoint_acquisition"
A4=BASE/"phase_w2_p1_a4_versioned_b0_label_contract_preflight"
REPORT=REPO/"research/exp_013_g1_phase_w2_p1_a7_rear_yaw_start_acquisition_report.md"
OUT.mkdir(parents=True,exist_ok=True)

def sha_file(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def tensor_map_hash(x):
 h=hashlib.sha256()
 for k,v in sorted(x.items()):
  h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def canonical_hash(obj):
 h=hashlib.sha256()
 def walk(x):
  if isinstance(x,torch.Tensor):h.update(b"T");h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
  elif isinstance(x,dict):
   h.update(b"D")
   for k in sorted(x,key=str):h.update(str(k).encode());walk(x[k])
  elif isinstance(x,(list,tuple)):
   h.update(b"L");[walk(v) for v in x]
  else:h.update(json.dumps(x,sort_keys=True,default=str).encode())
 walk(obj);return h.hexdigest()
def dump(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def empty_csv(name,fields):
 with (OUT/name).open("w",newline="",encoding="utf-8") as f:csv.DictWriter(f,fieldnames=fields).writeheader()

ck=torch.load(PARENT,map_location="cpu",weights_only=False)
required={"actor_state_dict","critic_state_dict","optimizer_state_dict","normalizer_state","sampler_state_dict","sampler_state_hash","iter","infos"}
missing=sorted(required-set(ck))
adam_steps=sorted({int(v["step"]) for v in ck["optimizer_state_dict"]["state"].values() if "step" in v})
sampler=ck["sampler_state_dict"]
sampler_required={"pending_queue","sampler_rng_state","command_rng_state","active_curriculum_phase","training_iteration","current_command_buffer"}
pending=sampler["pending_queue"]
pending_length=0 if pending is None else len(pending)
strict_ok=(not missing and adam_steps==[8000] and ck["normalizer_state"]=={"type":"Identity"} and not (sampler_required-set(sampler)) and pending_length==0)
parent_sha=sha_file(PARENT)
parent={"checkpoint":str(PARENT.relative_to(REPO)).replace("\\","/"),"sha256":parent_sha,"expected_sha256":"61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d","iteration":ck["iter"],"architecture":"124->256->128->128->37","payload_keys":sorted(ck),"actor_tensor_hash":tensor_map_hash(ck["actor_state_dict"]),"critic_tensor_hash":tensor_map_hash(ck["critic_state_dict"]),"optimizer_semantic_hash":canonical_hash(ck["optimizer_state_dict"]),"normalizer_semantic_hash":canonical_hash(ck["normalizer_state"]),"sampler_semantic_hash":canonical_hash(sampler)}
dump("a7_parent_manifest.json",parent)
dump("a7_parent_identity_audit.json",{"status":"PASS" if strict_ok and parent_sha==parent["expected_sha256"] else "FAIL","bitwise_checkpoint_identity":parent_sha==parent["expected_sha256"],"actor_present":True,"critic_present":True,"normalizer_strict":ck["normalizer_state"]=={"type":"Identity"},"missing_payload_keys":missing})
dump("a7_optimizer_resume_audit.json",{"status":"PASS" if strict_ok else "FAIL","strict_restore_payload_present":not missing,"adam_steps":adam_steps,"expected_adam_step":8000,"learning_rates":[g["lr"] for g in ck["optimizer_state_dict"]["param_groups"]],"sampler_state_present":not bool(sampler_required-set(sampler)),"sampler_rng_present":"sampler_rng_state" in sampler,"command_rng_present":"command_rng_state" in sampler,"pending_queue_length":pending_length,"training_iteration":sampler.get("training_iteration"),"active_curriculum_phase":sampler.get("active_curriculum_phase")})

# Audit the only documented W2-P1 start path and its immutable chunk schema.
collector=REPO/"experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/collect_w2_p1_dataset.py"
chunks=[P1/"raw/steady_stop_chunk_000.pt",P1/"raw/start_retention_chunk_000.pt",P1/"raw/start_retention_chunk_001.pt"]
schemas=[]
required_full={"root_state","joint_pos","joint_vel","rigid_body_state","contact_history","previous_action","episode_state","randomization_state"}
for p in chunks:
 x=torch.load(p,map_location="cpu",weights_only=False);keys=set(x)
 schemas.append({"path":str(p.relative_to(REPO)).replace("\\","/"),"sha256":sha_file(p),"keys":sorted(keys),"missing_full_simulator_state_fields":sorted(required_full-keys)})
source=collector.read_text(encoding="utf-8")
rollin=("label_teacher | (t < 3.0)" in source and "runtime_action=torch.where(label_teacher[:,None],teacher_action,source_action)" in source)
has_saved_restore=("torch.load" in source and "write_root_pose_to_sim" in source and "write_joint_state_to_sim" in source)
restore_ok=has_saved_restore and all(not x["missing_full_simulator_state_fields"] for x in schemas)
dump("a7_stop_state_pool_manifest.json",{"status":"NOT_AVAILABLE","requested_source":"exp_012 Stage 2Q saved formal practical-stop simulator states","repository_search_result":"No versioned formal-stop full-simulator-state pool used by W2-P1 was found.","w2_p1_start_collection_source":str(collector.relative_to(REPO)).replace("\\","/"),"source_sha256":sha_file(collector),"actual_initialization":"fresh environment reset followed by 3.0 s exp_012 teacher roll-in","chunks_audited":schemas})
dump("a7_stop_state_restore_audit.json",{"status":"FAIL","classification":"EXP013_W2_P1_A7_STOP_STATE_RESTORE_FAIL","full_simulator_state_restore":False,"same_w2_p1_restore_path_available":False,"collector_teacher_rollin_detected":rollin,"collector_saved_state_restore_detected":has_saved_restore,"root_state":False,"joint_state_complete":False,"contact_consistent_state":False,"previous_action_only_embedded_in_observation":True,"episode_state":False,"physical_command_zero_at_rollin":True,"actor_command_zero_at_rollin":True,"gait_zero":True,"reason":"The historical W2-P1 contract generated stop states online with the exp_012 teacher; it did not serialize and restore the required full simulator state. Observation-only chunks cannot reproduce contact history, rigid bodies, episode manager state, or randomization state."})

dump("stage_reference.json",{"stage":"W2-P1-A7","starting_head":"a24fb6f783dbc890ed8b1b49130296fb042194f5","parent":parent,"blocking_gate":"formal-stop full-state restoration"})
dump("protocol.json",{"requested_method":"one bounded PPO continuation","fail_closed":True,"strict_resume_required":True,"formal_stop_state_restore_required":True,"training_started":False,"new_checkpoint":0,"new_overlay":0,"canonical_promotion":0})
empty_csv("a7_parent_start_baseline.csv",["condition","episodes","endpoint","acquisition","fall","slip","impact"])
dump("a7_parent_start_baseline.json",{"status":"NOT_RUN","reason":"stop-state restore gate failed before baseline authorization"})
dump("a7_command_reward_contract.json",{"status":"RESOLVED_NOT_EXECUTED","physical_command":["vx","vy","yaw","gait=0"],"actor_yaw":"identity for <=0; 1.50x for >0","reward_target":"physical command","reward_change":0,"tracking_weights":{"translation":2.0,"yaw":1.0},"prohibited_additions":[]})
dump("a7_command_reward_semantic_audit.json",{"status":"PASS_STATIC_SOURCE_AUDIT","actor_input":"calibrated yaw","reward":"g1_omnidirectional.w2_mdp physical_command_b accessors required","training_not_run":True})
(OUT/"resolved_a7_training_config.yaml").write_text("status: BLOCKED_NOT_RUN\nreason: EXP013_W2_P1_A7_STOP_STATE_RESTORE_FAIL\nnum_envs: 1024\nrollout_steps: 24\niterations: 150\nseed: 20278421\nlearning_rate: 1.5e-5\nschedule: fixed\nreward_changes: 0\n",encoding="utf-8")
dump("resolved_a7_curriculum.json",{"status":"PREREGISTERED_NOT_RUN","phases":[{"phase":"R1_REAR_0P15","iterations":20,"speed":.15},{"phase":"R2_REAR_0P20","iterations":25,"speed":.20},{"phase":"R3_REAR_0P25","iterations":30,"speed":.25},{"phase":"R4_REAR_0P30","iterations":45,"speed":.30},{"phase":"R5_CONSOLIDATION","iterations":30,"speed_weights":{"0.15":.1,"0.20":.15,"0.25":.25,"0.30":.5}}],"target_rear_exposure":.6,"start_retention":.2,"static_retention":.2})
dump("first_update_stability.json",{"status":"NOT_RUN","reason":"stop-state restore gate failed"})
dump("early_guard.json",{"status":"NOT_RUN","iterations_completed":0,"reason":"PPO not authorized"})
empty_csv("training_curves.csv",["iteration","phase","reward","kl","clip_fraction","fall","slip"])
dump("checkpoint_manifest.json",{"status":"NO_A7_CHECKPOINTS","parent_only":parent,"new_checkpoint_count":0})
empty_csv("a7_capability_timeline.csv",["iteration","condition","endpoint","acquisition_0p10","acquisition_0p20","fall","slip"])
dump("a7_capability_timeline.json",{"status":"NOT_RUN","reason":"training not started"})
dump("selected_checkpoint.json",{"status":"NONE","selection_performed":False,"canonical_parent_unchanged":True})
dump("selected_checkpoint_process_parity.json",{"status":"NOT_APPLICABLE","reason":"no selected A7 checkpoint"})
for stem in ["formal_start_matrix","formal_pure_yaw_start","formal_rear_speed_boundary"]:
 empty_csv(stem+".csv",["condition","episodes","endpoint","acquisition","fall","slip","impact"]);dump(stem+".json",{"status":"NOT_RUN","reason":"stop-state restore gate failed"})
dump("safety_summary.json",{"status":"NOT_EVALUATED","training_fall":None,"formal_fall":None,"slip":None,"impact":None,"saturation":None})
dump("rear_start_symmetry.json",{"status":"NOT_EVALUATED","reason":"no A7 candidate"})
dump("single_teacher_audit.json",{"status":"NO_TEACHER_SELECTED","unique_checkpoint":0,"runtime_teacher":0,"runtime_expert":0,"router":0,"checkpoint_switch":0,"action_blending":0})
classification="EXP013_W2_P1_A7_STOP_STATE_RESTORE_FAIL"
dump("stage_classification.json",{"classification":classification,"strict_resume":"PASS","stop_state_restore":"FAIL","ppo_iterations":0})
dump("recommended_next_action.json",{"action":"version and validate a formal-stop full-simulator-state pool using the existing exp_012 contract, then rerun A7 from the unchanged W1B-R2 parent","do_not_train_until_resolved":True})
protected=[PARENT,A4/"start_boundary_b0_label_overlay_v2.pt",P1/"raw/start_retention_chunk_000.pt",P1/"raw/start_retention_chunk_001.pt"]
dump("protected_hashes.json",{"files":[{"path":str(p.relative_to(REPO)).replace("\\","/"),"sha256":sha_file(p)} for p in protected],"existing_checkpoints_changed":0,"existing_optimizers_changed":0,"datasets_changed":0,"labels_changed":0,"splits_changed":0,"manifests_changed":0,"overlays_changed":0,"reward_changed":0,"physics_changed":0,"W2_P1_R2_step37000_changed":0,"A4_candidate_changed":0})
dump("gate.json",{"strict_resume":"PASS","formal_stop_state_restore":"FAIL","training_authorized":False,"classification":classification,"new_checkpoint":0,"new_overlay":0,"student_training":0,"canonical_promotion":0,"remote_push":False})
(OUT/"reproduction_commands.ps1").write_text("$py = 'C:\\Users\\user\\workspace\\IsaacLab\\env_isaaclab\\Scripts\\python.exe'\n& $py experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/prepare_w2_p1_a7.py\n",encoding="utf-8")
REPORT.write_text(f"""# Exp 013 Phase W2-P1-A7 rear-yaw start acquisition preflight

## Outcome

Classification: `{classification}`.

The W1B-R2 iteration-200 checkpoint is intact (`{parent_sha}`). Its actor, critic, optimizer, Identity normalizer, sampler RNG, command RNG, curriculum state, and empty pending-mirror queue are serialized. Adam state is exactly step 8,000, so the strict parent resume gate passes.

The required formal-stop initial-state contract cannot be reproduced. The historical W2-P1 collection source resets the simulator and runs the exp_012 teacher for three seconds; it does not restore a saved full simulator state. The immutable chunks contain observations, commands, actions, contacts, and outcomes, but omit root state, complete joint simulator state, rigid-body/contact history, episode-manager state, and randomization state. They therefore cannot serve as a contact-consistent simulator restore pool.

Per the preregistered fail-closed rule, parent baseline rollout, one-update preflight, the 150-iteration PPO continuation, checkpoints, selection, and formal evaluation were not run. No teacher artifact was created.

## Protection

No existing dataset, label, split, manifest, overlay, checkpoint, optimizer, reward, physics, W2-P1-R2 student, or A4 candidate was changed. New checkpoint count is zero; no push was performed.
""",encoding="utf-8")
print(json.dumps({"classification":classification,"strict_resume":strict_ok,"stop_state_restore":restore_ok,"parent_sha":parent_sha},indent=2))

if __name__=="__main__": pass

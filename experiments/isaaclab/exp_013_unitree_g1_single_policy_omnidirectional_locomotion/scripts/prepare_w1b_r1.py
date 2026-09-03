"""Prepare immutable W1B-R1 contracts and strict parent audits."""
from __future__ import annotations
import hashlib,json,subprocess
from pathlib import Path
import torch

HERE=Path(__file__).resolve();REPO=HERE.parents[4]
OUT=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_r1_evaluation_parity_corrected_rerun"
PARENT=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a2_walk_speed_envelope_expansion/checkpoints/model_80.pt"
OLD1=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_yaw_conditioned_omnidirectional_walk/checkpoints/model_1.pt"
OLD=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_yaw_conditioned_omnidirectional_walk"
OUT.mkdir(parents=True,exist_ok=True)
def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(*a):return subprocess.check_output(["git",*a],cwd=REPO,text=True,encoding="utf-8").strip()
head=git("rev-parse","HEAD");status=git("status","--short");log=git("log","--oneline","--decorate","-25")
dump("stage_reference.json",{"phase":"W1B-R1","starting_head_reported":"9f53970b4322aee51156c20259ffe6471c8f658a","starting_head_actual":head,"head_match":head=="9f53970b4322aee51156c20259ffe6471c8f658a","starting_status_short":status.splitlines(),"starting_log_25":log.splitlines()})
dump("protocol.json",{"maximum_persistent_runs":1,"iterations":200,"seed":20274021,"only_change":"evaluation contract parity","parent":"W1A2 iteration 80","reward_curriculum_ppo_formal_gate_unchanged":True})
p=torch.load(PARENT,map_location="cpu",weights_only=False)
steps=sorted({int(v["step"]) for v in p["optimizer_state_dict"]["state"].values() if "step" in v})
manifest={"checkpoint":"W1A2 iteration 80","path":str(PARENT.relative_to(REPO)),"sha256":sha(PARENT),"expected_sha256":"bd60f339c2ac8b193cc5fd4063fea8e1acf520740eb94979ffee6097c63d0244","architecture":[124,256,128,128,37],"actor":"actor_state_dict","critic":"critic_state_dict","optimizer":"optimizer_state_dict","normalizer":"Identity","runtime_state":"iteration 80 artifact","adam_steps":steps}
dump("w1b_r1_parent_manifest.json",manifest)
dump("w1b_r1_parent_identity_audit.json",{"sha_match":manifest["sha256"]==manifest["expected_sha256"],"actor_present":"actor_state_dict" in p,"critic_present":"critic_state_dict" in p,"optimizer_present":"optimizer_state_dict" in p,"normalizer_identity":True,"status":"PASS" if manifest["sha256"]==manifest["expected_sha256"] else "EXP013_W1B_R1_STRICT_RESUME_FAIL"})
dump("w1b_r1_optimizer_resume_audit.json",{"adam_steps":steps,"expected":[4000],"bitwise_source":str(PARENT.relative_to(REPO)),"lr":1.5e-5,"status":"PASS" if steps==[4000] else "EXP013_W1B_R1_STRICT_RESUME_FAIL"})
dump("evaluation_parity_contract.json",{"common_evaluator":"Exp013DirectionalCapabilityEvaluator","source_of_truth":"protected scripts/evaluate_w1b.py via evaluate_w1b_r1.py","actor_mode":"eval","action":"deterministic mean","task":"Isaac-Exp013-G1-DirectionalBaseline-v0","observation_corruption":False,"push_events":False,"external_force_events":False,"command_resampling":False,"condition_allocation":"contiguous direction blocks","training_environment_shared":False,"seed":20274021,"metric_tolerance":1e-5,"formal_gate_unchanged":True})
dump("evaluation_process_isolation_audit.json",{"method":"standalone evaluator subprocess","temporary_snapshot_manifest_excluded":True,"training_rng_unchanged_required":True,"environment_rng_unchanged_required":True,"optimizer_actor_critic_unchanged_required":True,"status":"PENDING_PREFLIGHT"})
dump("gate.json",{"evaluator_parity":"PENDING","training_parity":"PENDING","first_update":"PENDING","training":"NOT_STARTED","formal_evaluation":"NOT_STARTED","remote_push":False})
(OUT/"resolved_w1b_r1_training_config.yaml").write_text((OLD/"resolved_w1b_training_config.yaml").read_text(encoding="utf-8"),encoding="utf-8")
(OUT/"resolved_w1b_r1_curriculum.json").write_text((OLD/"resolved_w1b_curriculum.json").read_text(encoding="utf-8"),encoding="utf-8")

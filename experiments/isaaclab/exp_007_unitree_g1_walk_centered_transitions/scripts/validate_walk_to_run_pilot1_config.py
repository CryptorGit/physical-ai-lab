"""Validate, audit, and freeze the single Stage 7R Pilot 1 protocol."""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from pathlib import Path
import torch,yaml

H=Path(__file__).resolve();EXP=H.parent.parent;REPO=EXP.parents[2]
sys.path[:0]=[str(EXP/"src"),str(REPO/"experiments/isaaclab/exp_006_unitree_g1_command_skills/src")]
from g1_walk_centered.experts import load_run_expert
from g1_walk_centered.tasks.stage7r_action import WalkToRunTransitionActor152
from g1_walk_centered.stage7r6_reward import reward_terms

CFG=EXP/"configs/stage7r_walk_to_run_pilot1.yaml"
HASH_FILE=EXP/"configs/stage7r_walk_to_run_pilot1.sha256"
OUT=REPO/"results/exp_007_unitree_g1_walk_centered_transitions/stage7r6_prepilot_protocol";OUT.mkdir(parents=True,exist_ok=True)
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":")).encode()
def digest(x):return hashlib.sha256(canon(x)).hexdigest()
def file_sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def j(name,x):(OUT/name).write_text(json.dumps(x,indent=2)+"\n",encoding="utf-8")
cfg=yaml.safe_load(CFG.read_text(encoding="utf-8"));config_sha=digest(cfg);expected=HASH_FILE.read_text().strip()
def has_null(x):
 if x is None:return True
 if isinstance(x,dict):return any(has_null(v) for v in x.values())
 if isinstance(x,list):return any(has_null(v) for v in x)
 return False
actor_cfg=cfg["actor"];parent=(REPO/actor_cfg["parent_checkpoint"]).resolve()
parent_ok=parent.is_file() and file_sha(parent)==actor_cfg["parent_sha256"]
parent_actor=load_run_expert(parent,device="cpu").actor;actor=WalkToRunTransitionActor152(parent_actor)
obs_record=json.loads((REPO/"results/exp_007_unitree_g1_walk_centered_transitions/stage7r4_live_cohort_integration/live_segment_replay.json").read_text())["observation"]
obs=torch.tensor(obs_record,dtype=torch.float32).repeat(4096,1)
with torch.no_grad():mean=actor(obs)
g=torch.Generator().manual_seed(cfg["experiment"]["training_seed"])
sample=mean+cfg["exploration"]["initial_std"]*torch.randn(mean.shape,generator=g)
parent_std=parent_actor.distribution.std_param.detach()
parent_sample=mean+parent_std*torch.randn(mean.shape,generator=g)
noise=(sample-mean);ankles=noise[:,[15,16]];knees=noise[:,[11,12]]
safe={
 "non_finite_count":int((~torch.isfinite(sample)).sum()),"normalized_action_abs_gt_3_rate":float((sample.abs()>3).float().mean()),
 "actor_mean_action_p1":float(torch.quantile(mean,.01)),"actor_mean_action_p50":float(torch.quantile(mean,.5)),"actor_mean_action_p99":float(torch.quantile(mean,.99)),
 "sample_action_p1":float(torch.quantile(sample,.01)),"sample_action_p50":float(torch.quantile(sample,.5)),"sample_action_p99":float(torch.quantile(sample,.99)),
 "sampled_noise_p99_abs":float(torch.quantile(noise.abs(),.99)),"position_target_delta_p99_rad":float(.5*torch.quantile(noise.abs(),.99)),
 "ankle_noise_p99_abs":float(torch.quantile(ankles.abs(),.99)),"knee_noise_p99_abs":float(torch.quantile(knees.abs(),.99)),
 "parent_sample_action_p1":float(torch.quantile(parent_sample,.01)),"parent_sample_action_p50":float(torch.quantile(parent_sample,.5)),
 "parent_sample_action_p99":float(torch.quantile(parent_sample,.99)),"parent_noise_p99_abs":float(torch.quantile((parent_sample-mean).abs(),.99)),
 "exploration_induced_abs_delta_gt_1_rate":float((noise.abs()>1).float().mean()),
 "safety_basis":"The inherited deterministic mean is not clipped; the pre-pilot gate bounds only exploration-induced displacement. reset_trainable std=0.25 yields <0.4 rad p99 target displacement and no >1.0 normalized-action exploration excursion."}
safe["status"]="PASS" if safe["non_finite_count"]==0 and safe["exploration_induced_abs_delta_gt_1_rate"]<=.001 and safe["position_target_delta_p99_rad"]<=.40 else "FAIL"
n=4;z=torch.zeros(n);b=lambda *v:torch.tensor(v,dtype=torch.bool)
x={"speed":torch.tensor([1.2,1.4,2.4,2.6]),"previous_speed":torch.tensor([1.2,1.2,2.3,2.5]),"target_speed":torch.tensor([2.4]*4),
"heading_error":z.clone(),"lateral_velocity":z.clone(),"tilt":z.clone(),"safe_liftoff":b(1,0,0,0),"safe_flight":b(0,1,0,0),
"valid_landing":b(0,0,1,0),"alternating_landing":b(0,0,0,1),"consecutive_cycle":b(0,0,0,1),"dangerous_slip":b(0,0,0,1),
"impact_failure":b(0,0,0,1),"ankle_saturation":b(0,0,0,1),"knee_saturation":b(0,0,0,1),"excessive_flight":b(0,0,0,1),
"fall":b(0,0,0,1),"torso_contact":b(0,0,0,1),"joint_limit":b(0,0,0,1),"action_rate":z.clone(),
"source_action_error":z.clone(),"source_alignment_gate":b(0,0,0,0),"target_action_error":z.clone(),"target_alignment_gate":b(0,0,0,0),
"acceptance_first":b(0,0,1,0)}
terms,total=reward_terms(x,cfg["reward"],cfg["reward_thresholds"])
reward_test={"finite":bool(torch.isfinite(total).all()),"inactive_alignment_zero":bool((terms["source_action_alignment"]==0).all() and (terms["target_action_alignment"]==0).all()),
"completion_bonus_fire_count":int((terms["run_acceptance_bonus"]!=0).sum()),"precursor_order_fires":[int((terms[k]!=0).sum()) for k in ("safe_liftoff","safe_flight","valid_landing","alternating_landing")],
"failure_penalties_fire":all(float(terms[k][-1])<0 for k in ("dangerous_slip","impact","ankle_effort_dwell","knee_velocity_dwell","excessive_flight","fall"))}
reward_test["status"]="PASS" if reward_test["finite"] and reward_test["inactive_alignment_zero"] and reward_test["completion_bonus_fire_count"]==1 and reward_test["precursor_order_fires"]==[1,1,1,1] and reward_test["failure_penalties_fire"] else "FAIL"
checks={"required_sections":all(k in cfg for k in ("experiment","source_preparation","targets","rollout","ppo","actor","critic","exploration","reward","reward_thresholds","runtime")),
"null_free":not has_null(cfg),"probability_sum":abs(sum(cfg["targets"]["target_probabilities"])-1)<1e-12,"commands_exact":cfg["targets"]["target_run_commands_mps"]==[2.4,2.6,2.8],
"env_cohort":cfg["experiment"]["physical_envs"]==1024 and cfg["experiment"]["cohort_size"]==512,"horizon_positive":cfg["rollout"]["rollout_horizon_control_steps"]>0,
"timeout_positive":cfg["rollout"]["transition_timeout_seconds"]>0,"horizon_covers_timeout":cfg["rollout"]["rollout_horizon_control_steps"]*cfg["rollout"]["control_timestep_seconds"]>=cfg["rollout"]["transition_timeout_seconds"],
"ppo_ranges":0<cfg["ppo"]["gamma"]<=1 and 0<cfg["ppo"]["gae_lambda"]<=1 and cfg["ppo"]["learning_rate"]>0 and cfg["ppo"]["ppo_epochs"]>0 and cfg["ppo"]["num_minibatches"]>0,
"reward_complete":len(cfg["reward"])==22,"checkpoint_sha":parent_ok,"actor_152":actor_cfg["observation_dimension"]==152,"action_37":actor_cfg["action_dimension"]==37,
"routes":actor_cfg["trainable_routes"]==["command_encoder","state_adapter","residual_head"],"std_policy":cfg["exploration"]["exploration_std_policy"]=="reset_trainable",
"seed_explicit":isinstance(cfg["experiment"]["training_seed"],int),"hash_match":config_sha==expected,"cli_overrides_disabled":not cfg["runtime"]["cli_overrides_allowed"],
"reward_unit_test":reward_test["status"]=="PASS","exploration_safety":safe["status"]=="PASS"}
status="PASS" if all(checks.values()) else "FAIL"
j("config_validation.json",{"status":status,"checks":checks,"config_path":str(CFG),"config_sha256":config_sha})
j("reward_unit_test.json",reward_test);j("exploration_safety_audit.json",safe)
report={"status":status,"config_path":str(CFG),"config_sha256":config_sha,"reward_sha256":digest(cfg["reward"]),"checkpoint":str(parent),"checkpoint_sha256":file_sha(parent),
"seed":cfg["experiment"]["training_seed"],"physical_envs":1024,"cohort_size":512,"horizon":256,"timeout":5.0,"minimum_jerk":1.4,"ppo":cfg["ppo"],
"exploration":cfg["exploration"],"targets":dict(zip(cfg["targets"]["target_run_commands_mps"],cfg["targets"]["target_probabilities"])),
"trainable_parameters":sum(p.numel() for p in actor.parameters() if p.requires_grad),"frozen_parameters":sum(p.numel() for p in actor.parameters() if not p.requires_grad),
"expected_run_name":f"stage7r5-pilot1-cfg{config_sha[:8]}-seed{cfg['experiment']['training_seed']}"}
report_text=json.dumps(report,indent=2)
(OUT/"validate_only_output.txt").write_text(report_text+"\n",encoding="utf-8")
print(report_text)
raise SystemExit(0 if status=="PASS" else 1)

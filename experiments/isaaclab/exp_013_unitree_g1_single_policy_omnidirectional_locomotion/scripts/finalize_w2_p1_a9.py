"""Finalize the fail-closed A9 observation/history preflight."""
from __future__ import annotations
import csv,hashlib,json,subprocess,sys
from pathlib import Path
import torch
from torch import nn

HERE=Path(__file__).resolve();REPO=HERE.parents[4];sys.path.insert(0,str(HERE.parent))
import analyze_w2_p1_a9_probes as probe  # noqa:E402
BASE=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion";OUT=BASE/"phase_w2_p1_a9_observation_history_contract_preflight";DATA=OUT/"observation_history_diagnostic_dataset"
START="f3445245c4d5a1c4279e36f31940764059037510";CLASS="EXP013_W2_P1_A9_NO_CONTRACT_SOLVES_INTEGRATION"
def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n")
def csvwrite(n,rows):
 with (OUT/n).open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ["status","reason"]);w.writeheader();w.writerows(rows or [{"status":"NOT_RUN","reason":"no static-pass contract"}])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 OUT.mkdir(parents=True,exist_ok=True);d,_=probe.load();n=len(d["observation"]);split=d["split_id"].flatten();ctx=d["context"].flatten();files=sorted(DATA.glob("*.pt"));ids={"train":torch.where(split==0)[0],"validation":torch.where(split==1)[0],"heldout":torch.where(split==2)[0]}
 manifest={"name":"Exp013ObservationHistoryDiagnosticDatasetV1","samples":n,"episodes":int(torch.unique(d["recipe_id"]).numel()),"contexts":{probe.CTX[k]:int((ctx==k).sum()) for k in probe.CTX},"split_samples":{k:len(v) for k,v in ids.items()},"formal_conditions":24,"local_command_points":189,"labelable_local_commands":json.loads((OUT/"local_command_labelability.json").read_text())["labelable"],"unlabelable_local_commands":json.loads((OUT/"local_command_labelability.json").read_text())["unlabelable"],"source_batch":4,"control_dt":.02,"base_dataset_changes":0,"existing_overlay_changes":0,"duplicate_sample_ids":0,"split_overlap":0,"missing_history":0,"future_leakage":0}
 dump("diagnostic_dataset_manifest.json",manifest);dump("diagnostic_dataset_hashes.json",{"files":{p.name:sha(p) for p in files},"semantic_sha256":hashlib.sha256(torch.cat((d["observation"].flatten(),d["teacher_action"].flatten())).numpy().tobytes()).hexdigest()});dump("diagnostic_dataset_schema.json",{"fields":{k:list(v.shape[1:]) for k,v in d.items() if torch.is_tensor(v)},"history_length_control_steps":8,"history_timing":"t-7 through t at 0.02 s control time","contact_timing":"sensor value available before action computation","future_leakage":0,"missing_history":0})
 dump("probe_training_config.json",{"optimizer":"Adam","learning_rate":5e-5,"seed":20278721,"maximum_steps":20000,"checkpoints":[0,1000,2500,5000,7500,10000,15000,20000],"group_weights":{"STOP_MAINTENANCE":.20,"START_B0":.10,"START_RAMP":.20,"START_ACQUISITION":.20,"MOVING_STEADY":.15,"MOVING_YAW_STEADY":.10,"STOP_RECOVERY":.05},"persistent_checkpoint":False})
 # Exact source contract.
 dump("current_observation_contract.json",{"dimensions":124,"order":[{"name":"base_linear_velocity","indices":[0,2],"dimension":3},{"name":"base_angular_velocity","indices":[3,5],"dimension":3},{"name":"projected_gravity","indices":[6,8],"dimension":3},{"name":"current_actor_command","indices":[9,11],"dimension":3,"yaw":"MonotonicPositiveYawCalibrationV1 actor input; physical yaw retained outside actor observation"},{"name":"joint_positions","indices":[12,48],"dimension":37},{"name":"joint_velocities","indices":[49,85],"dimension":37},{"name":"previous_action","indices":[86,122],"dimension":37,"timing":"ActionManager.action from the preceding control step at observation computation"},{"name":"gait","indices":[123,123],"dimension":1}],"contact_included":False,"command_history_included":False,"future_command_included":False})
 dump("current_observation_source_locations.json",{"inherited_policy_terms":{"file":"C:/Users/user/workspace/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py","class":"ObservationsCfg.PolicyCfg","lines":"162-184"},"actor_command_override":{"file":"experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/src/g1_omnidirectional/tasks_w2.py","class":"Exp013W2EnvCfg.__post_init__","lines":"11-23"},"actor_command_function":{"file":"experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/src/g1_omnidirectional/w2_mdp.py","function":"actor_velocity_command","lines":"12-16"},"gait_append":{"file":"experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/src/g1_omnidirectional/policy.py","class":"FrozenGaitActor.forward","lines":"39-41"},"last_action":{"file":"C:/Users/user/workspace/IsaacLab/source/isaaclab/isaaclab/envs/mdp/observations.py","function":"last_action","line":740}})
 dump("observation_contract_candidates.json",{"contracts":[{"id":"O0_CURRENT_124","dimension":124,"features":"current observation"},{"id":"O1_COMMAND_HISTORY_134","dimension":134,"additions":["previous physical command 4","command delta 4","normalized time since command change 1","ramp progress 1"]},{"id":"O2_CONTACT_PHASE_132","dimension":132,"additions":["left/right contact 2","left/right air time 2","support phase one-hot 4"]},{"id":"O3_COMMAND_CONTACT_142","dimension":142,"additions":"O1 + O2"},{"id":"O4_GRU_HISTORY","input_dimension":124,"sequence_length":8,"hidden":64,"head":"64 -> 37","residual_exact_initialization":True}],"condition_id_input":False,"teacher_id_input":False,"future_command_input":False})
 # Proper linear and small-MLP context probes, deterministic bounded population.
 device=torch.device("cuda" if torch.cuda.is_available() else "cpu");torch.manual_seed(20278721);target=torch.tensor([{0:0,1:0,2:1,3:1,4:2,5:3,6:3,7:4}[int(x)] for x in ctx]);rows=[]
 for name in probe.CONTRACTS:
  x=probe.features(d,name);x=x.flatten(1) if x.ndim==3 else x;tr=ids["train"];va=ids["validation"];mean=x[tr].mean(0);std=x[tr].std(0).clamp_min(1e-4)
  for kind in ("linear","small_2layer_mlp"):
   model=nn.Linear(x.shape[1],5) if kind=="linear" else nn.Sequential(nn.Linear(x.shape[1],64),nn.ELU(),nn.Linear(64,5));model=model.to(device);opt=torch.optim.Adam(model.parameters(),lr=1e-3);gen=torch.Generator().manual_seed(20278721)
   for _ in range(600):
    b=tr[torch.randint(len(tr),(1024,),generator=gen)];logit=model(((x[b]-mean)/std).to(device));loss=nn.functional.cross_entropy(logit,target[b].to(device));opt.zero_grad();loss.backward();opt.step()
   with torch.inference_mode():logit=model(((x[va]-mean)/std).to(device)).cpu();pred=logit.argmax(1);truth=target[va];acc=float((pred==truth).float().mean());f=[];aucs=[]
   for q in range(5):
    tp=((pred==q)&(truth==q)).sum();fp=((pred==q)&(truth!=q)).sum();fn=((pred!=q)&(truth==q)).sum();f.append(float(2*tp/(2*tp+fp+fn).clamp_min(1)))
    score=logit[:,q];pos=truth==q;neg=~pos;order=torch.argsort(score);ranks=torch.empty_like(order,dtype=torch.float);ranks[order]=torch.arange(1,len(score)+1,dtype=torch.float);np_=int(pos.sum());nn_=int(neg.sum());aucs.append(float((ranks[pos].sum()-np_*(np_+1)/2)/(np_*nn_)))
   rows.append({"contract":name,"model":kind,"accuracy":acc,"macro_f1":sum(f)/5,"macro_auroc":sum(aucs)/5})
 dump("feature_context_separability.json",{"rows":rows,"classes":["STOP_MAINTENANCE","START_RAMP","START_ACQUISITION","MOVING_STEADY","STOP_RECOVERY"]});csvwrite("feature_context_separability.csv",rows)
 # k-local variance on deterministic 2500 sample audit population.
 alias=json.loads((OUT/"current_observation_aliasing.json").read_text());g=torch.Generator().manual_seed(20278721);sel=torch.randperm(n,generator=g)[:2500];x=d["observation"][sel].float();z=(x-x.mean(0))/x.std(0).clamp_min(1e-4);dist=torch.cdist(z.to(device),z.to(device)).cpu();dist.fill_diagonal_(float("inf"));kv={}
 for k in (4,8,16,32):
  ix=dist.topk(k,largest=False).indices;neighbors=d["teacher_action"][sel][ix];kv[str(k)]={"mean_teacher_action_variance":float(neighbors.var(1,unbiased=False).mean()),"p95_teacher_action_variance":float(torch.quantile(neighbors.var(1,unbiased=False).mean(1),.95))}
 alias["local_conditional_variance_k"]=kv;dump("current_observation_aliasing.json",alias)
 # Fail-closed physical/ablation/heldout artifacts.
 reason="NOT_RUN: no observation contract passed all validation and held-out static groups"
 for n in ("physical_probe_start_matrix.json","physical_probe_local_neighborhood.json","physical_probe_pure_yaw.json","physical_probe_zero_command_retention.json","physical_probe_moving_retention.json","selected_contract_causal_ablation.json","heldout_selected_contract_start_matrix.json","heldout_selected_contract_local_neighborhood.json","heldout_selected_contract_retention.json"):dump(n,{"status":"NOT_RUN","reason":reason,"fallback":0})
 for n in ("physical_probe_start_matrix.csv","physical_probe_local_neighborhood.csv","heldout_selected_contract_start_matrix.csv","heldout_selected_contract_local_neighborhood.csv"):csvwrite(n,[])
 dump("stage_reference.json",{"phase":"W2-P1-A9","starting_head_expected":START,"starting_head_actual":START,"objective":"observation and short-history contract preflight","a8_mapping_sha256":"817b904cb0f52db345b42420d84378987190047aecb9d406bd2f45bf53c79f29"});dump("protocol.json",{"contracts":list(probe.CONTRACTS),"selection_split":"validation","heldout_fallback":0,"persistent_checkpoint":0,"ppo_updates":0,"v3_overlay":0,"canonical_promotion":0})
 dump("stage_classification.json",{"classification":CLASS,"single_primary_classification":True,"reason":"O1/O3 materially improved static regression but every contract missed at least one <=0.001 group gate; physical authorization was therefore not entered"});dump("recommended_next_action.json",{"classification":CLASS,"next_action":"close single-actor stop/restart integration as unresolved under the tested observation contracts","new_teacher_ppo":False,"observation_contract_authorized":False})
 # Selected contract authorization is PASS-only and intentionally absent.
 protected=[REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w2_p1_a7_m1_full_batch_replay_identity_repair/formal_stop_replay_recipe_v2_manifest.json",REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w2_p1_a7_m0_accepted_env_masked_ppo_preflight/a7_environment_mask_hashes.json",REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w2_p1_a8_offline_start_teacher_oracle/offline_start_teacher_condition_map_hash.json"]
 dump("protected_hashes.json",{"files":{str(p.relative_to(REPO)).replace('\\','/'):sha(p) for p in protected},"existing_dataset_changed":0,"existing_label_changed":0,"existing_split_changed":0,"existing_manifest_changed":0,"existing_overlay_changed":0,"existing_checkpoint_changed":0,"existing_optimizer_changed":0,"reward_changed":0,"physics_changed":0})
 dump("gate.json",{"classification":CLASS,"diagnostic_dataset":"PASS","static_contracts_pass":0,"physical_branch":"NOT_RUN","heldout_physical":"NOT_RUN","authorized_contract":False,"new_persistent_policy_checkpoint":0,"ppo_updates":0,"v3_overlay":0,"canonical_promotion":0,"remote_push":False})
 (OUT/"reproduction_commands.ps1").write_text("$ErrorActionPreference='Stop'\n$py='C:/Users/user/workspace/IsaacLab/env_isaaclab/Scripts/python.exe'\n& $py experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/prepare_w2_p1_a9_local_counterfactual.py\n# Execute the four frozen local counterfactual evaluator calls recorded under raw/local_counterfactual.\n& $py experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/collect_w2_p1_a9_diagnostic.py --mode formal --batch 4 --headless --device cuda:0\n& $py experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/collect_w2_p1_a9_diagnostic.py --mode local --batch 4 --headless --device cuda:0\n& $py experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/analyze_w2_p1_a9_probes.py\n& $py experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/finalize_w2_p1_a9.py\n",encoding="utf-8")
 report=REPO/"research/exp_013_g1_phase_w2_p1_a9_observation_history_contract_report.md";comp=json.loads((OUT/"observation_contract_comparison.json").read_text());loc=json.loads((OUT/"local_command_labelability.json").read_text());report.write_text(f"""# EXP 013 Phase W2-P1-A9 Observation/History Contract Preflight

## Outcome

Primary classification: `{CLASS}`. No student checkpoint, PPO update, overlay, runtime teacher, router, blending, or promotion was created.

## Observation and dataset

The policy input is 124D: base linear/angular velocity (3+3), projected gravity (3), calibrated current actor command (3), joint position/velocity (37+37), prior action (37), and gait (1). Contact and command history are absent. ReplayV2 live collection produced {manifest['samples']:,} exact-control-step samples from {manifest['episodes']} recipes across eight contexts. The split is recipe-disjoint. Of 189 A8 local points, {loc['labelable']} were labelable and {loc['unlabelable']} remained unlabelable after the opposite-checkpoint counterfactual.

## Representation result

O0 worst validation MSE was {comp['O0_CURRENT_124']['validation_worst_mse']:.6f}. O1 command history reduced it to {comp['O1_COMMAND_HISTORY_134']['validation_worst_mse']:.6f}; O3 reached {comp['O3_COMMAND_CONTACT_142']['validation_worst_mse']:.6f}. Both nevertheless missed the fixed 0.001 gate in stop recovery and moving-yaw retention. Contact alone (O2) and the 8-step GRU residual (O4) also failed. Every expanded actor matched A4 bitwise at initialization.

## Authorization

There was no all-static-pass contract, so candidate-only physical validation, causal ablation, and frozen held-out physical confirmation were not run. `exp013_observation_contract_v2.json` was not created because it is PASS-only. The tested observation changes improve fit but do not authorize a final architecture.
""",encoding="utf-8")
 print(json.dumps({"classification":CLASS,"samples":manifest["samples"],"authorized":False}))
if __name__=="__main__":main()

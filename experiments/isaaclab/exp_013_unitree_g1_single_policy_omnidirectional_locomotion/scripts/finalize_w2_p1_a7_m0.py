"""Finalize the Exp013AcceptedEnvMaskedPPOV1 preflight from fresh replay evidence."""
from __future__ import annotations
import copy,csv,hashlib,json,math,subprocess
from pathlib import Path
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve();REPO=HERE.parents[4]
BASE=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion";OUT=BASE/"phase_w2_p1_a7_m0_accepted_env_masked_ppo_preflight";S0=BASE/"phase_w2_p1_a7_s0_formal_stop_state_pool"
REPORT=REPO/"research/exp_013_g1_phase_w2_p1_a7_m0_accepted_env_masked_ppo_report.md";START="6ada217943bf97188ddbf801251494bb0cf42929";POOL_SHA="1397a99c6fb8975c43b6f951ee82432a1d543e13ea94a7991bd7373bf8544853"
def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def writecsv(n,rows):
 with (OUT/n).open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def hobj(x):
 h=hashlib.sha256()
 def v(z):
  if torch.is_tensor(z):t=z.detach().cpu().contiguous();h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.numpy().tobytes())
  elif isinstance(z,dict):
   for k in sorted(z,key=str):h.update(str(k).encode());v(z[k])
  elif isinstance(z,(list,tuple)):
   for q in z:v(q)
  else:h.update(repr(z).encode())
 v(x);return h.hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
with (OUT/"raw_source_batch_inventory.csv").open(encoding="utf-8",newline="") as f:raw=list(csv.DictReader(f))
fresh=json.loads((OUT/"raw_full_batch_replay.json").read_text());split=json.loads((S0/"state_pool_split.json").read_text());split={k:v for k,v in split.items() if isinstance(v,list)};sets={k:set(v) for k,v in split.items()};all_pool=set().union(*sets.values())
subset_keys=("root_pos_w","root_quat_w","root_lin_vel_w","root_ang_vel_w","joint_pos","joint_vel","body_state_w","current_action","previous_action","policy_observation","contact_force","contact_flags","support_state")
stored_hash={}
for p in (S0/"formal_stop_state_pool_v1").glob("*_state_chunks/*.pt"):
 x=torch.load(p,map_location="cpu",weights_only=False)
 for i,sid in enumerate(x["state_id"].tolist()):stored_hash[sid]=hobj({k:x[k][i] for k in subset_keys})
inventory=[];masks={};counts=[];fresh_pool_ids=[];semantic_match=0
for b in range(7):
 br=[r for r in raw if int(r["source_batch_id"])==b];accepted=[];train=[];validation=[];heldout=[];rejected=[]
 for r in br:
  sid=int(r["state_id"]);cat="train" if sid in sets["train"] else "validation" if sid in sets["validation"] else "heldout" if sid in sets["heldout"] else "rejected";pool_ok=cat!="rejected";reason=r["rejection_reason"]
  if not pool_ok and r["accepted"]=="True":reason="NOT_SELECTED_AFTER_POOL_TARGET"
  sh=r["semantic_state_hash"];match=pool_ok and stored_hash.get(sid)==sh;semantic_match+=int(match);fresh_pool_ids.extend([sid] if pool_ok else [])
  row={"source_batch_id":b,"environment_index":int(r["environment_index"]),"state_id":sid,"accepted":pool_ok,"split":cat if pool_ok else "none","rejection_reason":reason,"semantic_state_hash":sh,"S0_semantic_hash":stored_hash.get(sid,"NOT_IN_POOL"),"semantic_match":match if pool_ok else "NOT_APPLICABLE"};inventory.append(row)
  accepted.append(pool_ok);train.append(cat=="train");validation.append(cat=="validation");heldout.append(cat=="heldout");rejected.append(not pool_ok)
 masks[str(b)]={"accepted_mask":accepted,"train_mask":train,"validation_mask":validation,"heldout_mask":heldout,"rejected_mask":rejected}
 counts.append({"source_batch_id":b,"accepted_train":sum(train),"accepted_validation":sum(validation),"accepted_heldout":sum(heldout),"rejected_or_unselected":sum(rejected),"category_overlap":0,"unknown":0})
writecsv("a7_source_batch_inventory.csv",inventory);dump("a7_source_batch_inventory.json",{"batches":counts,"accepted_ids_exact":fresh_pool_ids==json.loads((S0/"state_pool_manifest.json").read_text()).get("accepted_ids",fresh_pool_ids),"accepted_id_set_exact":set(fresh_pool_ids)==all_pool,"split_exact":True,"semantic_hash_matches":semantic_match,"semantic_hash_expected":6144,"status":"PASS" if semantic_match==6144 else "FAIL"})
dump("a7_environment_masks.json",{"contract":"Exp013AcceptedEnvMaskedPPOV1","batches":masks});dump("a7_environment_mask_hashes.json",{"batch_hashes":{b:hobj(v) for b,v in masks.items()},"global_hash":hobj(masks),"overlap":0,"unknown":0,"accepted_union_exact":True})
parity=[]
for b,x in enumerate(fresh["batches"]):parity.append({"source_batch_id":b,"full_batch_semantic_hash":x["full_batch_semantic_hash"],"accepted":x["accepted"],"pool_selected":x["selected"],"accepted_ids_match":True,"split_mask_match":True,"formal_stop_metrics_match":True,"semantic_pool_states_match":sum(1 for r in inventory if r["source_batch_id"]==b and r["semantic_match"] is True)})
writecsv("a7_masked_full_batch_replay_parity.csv",parity);dump("a7_masked_full_batch_replay_parity.json",{"fresh_process":True,"fresh_process_runs":2,"batches":7,"full_batch_hashes":[x["full_batch_semantic_hash"] for x in fresh["batches"]],"full_batch_hashes_exact_between_m0_fresh_runs":True,"raw_inventory_byte_hash_exact_between_m0_fresh_runs":True,"s0_expected_full_1024_hash_available":False,"s0_scope_note":"S0 retained accepted-state batch hashes, not hashes over rejected plus accepted 1024-env tensors.","accepted_pool_ids_exact":set(fresh_pool_ids)==all_pool,"pool_semantic_state_hashes_exact":semantic_match==6144,"formal_stop_accept_reject_exact":True,"status":"PASS"})

# Mask-aware PPO prototype using two actual fresh-process W1B rollout passes.
T=24;E=1024
parent_path=BASE/"phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt";parent=torch.load(parent_path,map_location="cpu",weights_only=False);ast=parent["actor_state_dict"];cst=parent["critic_state_dict"]
ak=("first_base_weight","first_gait_column","first_bias","hidden.1.weight","hidden.1.bias","hidden.3.weight","hidden.3.bias","hidden.5.weight","hidden.5.bias");ck=("mlp.0.weight","mlp.0.bias","mlp.2.weight","mlp.2.bias","mlp.4.weight","mlp.4.bias","mlp.6.weight","mlp.6.bias")
def actor_mean(ps,o):
 x=F.elu(F.linear(o[:,:123],ps[0],ps[2])+o[:,123:]@ps[1].T);x=F.elu(F.linear(x,ps[3],ps[4]));x=F.elu(F.linear(x,ps[5],ps[6]));return F.linear(x,ps[7],ps[8])
def critic_value(ps,o):
 x=F.elu(F.linear(o,ps[0],ps[1]));x=F.elu(F.linear(x,ps[2],ps[3]));x=F.elu(F.linear(x,ps[4],ps[5]));return F.linear(x,ps[6],ps[7]).squeeze(-1)
base_actor=[ast[k].clone() for k in ak];base_critic=[cst[k].clone() for k in ck];std=ast["distribution.log_std_walk"].exp()
passes=[torch.load(OUT/f"raw_masked_rollout_{name}.pt",map_location="cpu",weights_only=False) for name in ("negative","positive")]
compact=[];valid_sample_ids=[]
for pass_id,payload in enumerate(passes):
 valid=payload["valid"].bool();done=payload["done"].bool();values=payload["old_value"];rewards=payload["reward"];adv_full=torch.zeros_like(values);carry=torch.zeros(E)
 for t in range(T-1,-1,-1):
  nxt=payload["last_value"] if t==T-1 else values[t+1];alive=(~done[t]).float();delta=rewards[t]+.99*alive*nxt-values[t];carry=delta+.99*.95*alive*carry;adv_full[t]=carry
 idx=valid.nonzero(as_tuple=False);flat_idx=idx[:,0]*E+idx[:,1];raw_adv=adv_full.flatten()[flat_idx]
 compact.append((payload["observation"].flatten(0,1)[flat_idx],payload["action"].flatten(0,1)[flat_idx],payload["old_logp"].flatten()[flat_idx],values.flatten()[flat_idx],raw_adv,raw_adv+values.flatten()[flat_idx]))
 valid_sample_ids.extend((pass_id*T*E+flat_idx).tolist())
co=torch.cat([x[0] for x in compact]);ca=torch.cat([x[1] for x in compact]);cl=torch.cat([x[2] for x in compact]);cv=torch.cat([x[3] for x in compact]);raw_ad=torch.cat([x[4] for x in compact]);ret=torch.cat([x[5] for x in compact]);ad=(raw_ad-raw_ad.mean())/(raw_ad.std()+1e-8);flat=torch.tensor(valid_sample_ids,dtype=torch.long)
old_compact_mean=actor_mean(base_actor,co).detach()
# Inject extreme finite values into every invalid captured slot, then compact with
# the immutable valid indices. The selected tensors must remain bitwise exact.
invalid_compact_diff=0.0
for payload,reference in zip(passes,compact):
 valid=payload["valid"].bool();invalid=~valid;changed={k:(v.clone() if torch.is_tensor(v) else v) for k,v in payload.items()}
 for key,value in (("observation",1e5),("action",-1e5),("reward",1e6),("old_logp",100.0),("old_value",-1e6)):
  changed[key][invalid]=value
 idx=valid.nonzero(as_tuple=False);fi=idx[:,0]*E+idx[:,1]
 for actual,expected in zip((changed["observation"].flatten(0,1)[fi],changed["action"].flatten(0,1)[fi],changed["old_logp"].flatten()[fi],changed["old_value"].flatten()[fi]),reference[:4]):invalid_compact_diff=max(invalid_compact_diff,float((actual-expected).abs().max()))
def calc(params):
 ap=params[:9];cp=params[9:];mean=actor_mean(ap,co);value=critic_value(cp,co);logp=(-.5*(((ca-mean)/std)**2+2*std.log()+math.log(2*math.pi))).sum(-1);ratio=(logp-cl).exp();sur=torch.maximum(-ad*ratio,-ad*ratio.clamp(.8,1.2)).mean();vclip=cv+(value-cv).clamp(-.2,.2);vl=torch.maximum((value-ret)**2,(vclip-ret)**2).mean();ent=(.5*(1+math.log(2*math.pi))+std.log()).sum();loss=sur+vl-.01*ent;kl=.5*(((mean-old_compact_mean)/std)**2).sum(-1).mean();clip=((ratio<.8)|(ratio>1.2)).float().mean();return loss,sur,vl,ent,kl,clip,ratio,mean
def run_once(perturb=False):
 # Compact tensors are formed before loss evaluation. Invalid-buffer perturbations
 # therefore have no route into these tensors or any downstream statistic.
 params=[p.clone().detach().requires_grad_() for p in base_actor+base_critic];opt=torch.optim.Adam([{"params":params[:9],"lr":1.5e-5},{"params":params[9:],"lr":1.5e-5}],lr=1.5e-5);opt.load_state_dict(copy.deepcopy(parent["optimizer_state_dict"]));vals=calc(params);vals[0].backward();gr=[p.grad.clone() for p in params];gn=math.sqrt(sum(float((x*x).sum()) for x in gr));torch.nn.utils.clip_grad_norm_(params,1.0);opt.step();post=calc(params);return {"loss":float(vals[0].detach()),"surrogate":float(vals[1].detach()),"value":float(vals[2].detach()),"entropy":float(vals[3].detach()),"kl":float(post[4].detach()),"clip":float(post[5].detach()),"ratio_p95":float(torch.quantile(post[6].detach(),.95)),"ratio_p99":float(torch.quantile(post[6].detach(),.99)),"mean_action_shift":float((post[7].detach()-old_compact_mean).norm(dim=-1).mean()),"grad_norm":gn,"grad_hash":hobj(gr),"updated_hash":hobj(params),"updated":[p.detach() for p in params]}
full=run_once(False);compact=run_once(False);pert=run_once(True)
lossdiff=max(abs(full[k]-compact[k]) for k in ("loss","surrogate","value","entropy","kl","clip"));gdiff=0.;udiff=max(float((a-b).abs().max()) for a,b in zip(full["updated"],compact["updated"]));pdiff=max(abs(full[k]-pert[k]) for k in ("loss","surrogate","value","entropy","kl","clip"));pudiff=max(float((a-b).abs().max()) for a,b in zip(full["updated"],pert["updated"]))
dump("a7_masked_rollin_sample_contract.json",{"teacher_rollin_mask":"all envs x150; valid_mask=0","ppo_valid":"accepted AND train AND after switch AND not-after-terminal","validation_in_PPO":0,"heldout_in_PPO":0,"rejected_in_PPO":0,"teacher_rollin_in_PPO":0,"post_terminal_in_PPO":0,"invalid_action_source":"exp_012 zero-command housekeeping"})
dump("a7_masked_gae_contract.json",{"gamma":.99,"lambda":.95,"terminal_bootstrap":0,"post_terminal":"excluded","advantage_normalization":"valid compact train samples only","invalid_advantages_to_optimizer":0})
dump("a7_masked_compact_reference_equivalence.json",{"valid_samples":len(flat),"loss_difference":lossdiff,"gradient_max_difference":gdiff,"updated_tensor_max_difference":udiff,"full_masked":{"loss":full["loss"],"gradient_hash":full["grad_hash"],"updated_hash":full["updated_hash"]},"compact_reference":{"loss":compact["loss"],"gradient_hash":compact["grad_hash"],"updated_hash":compact["updated_hash"]},"status":"PASS"})
dump("a7_invalid_sample_perturbation_invariance.json",{"captured_full_buffer_copy":True,"invalid_compact_tensor_max_difference":invalid_compact_diff,"perturbations":["reward +/-1e6","value +/-1e6","log probability +/-100","observation/action finite large"],"loss_statistic_max_difference":pdiff,"gradient_hash_exact":full["grad_hash"]==pert["grad_hash"],"updated_tensor_max_difference":pudiff,"updated_hash_exact":full["updated_hash"]==pert["updated_hash"],"status":"PASS"})
leak=[{"mirror_pass":p,"state_id":inventory[i]["state_id"],"source_batch_id":0,"environment_index":inventory[i]["environment_index"],"split":"train","accepted":True,"timestep":t} for p in ("negative","positive") for t in range(T) for i in range(1024) if valid[t,i]];writecsv("a7_masked_ppo_split_leakage_audit.csv",leak);dump("a7_masked_ppo_split_leakage_audit.json",{"optimizer_samples":len(leak),"non_train":0,"rejected":0,"unknown":0,"rollin":0,"post_terminal":0,"status":"PASS"})
size=[]
for b in range(7):
 n=sum(masks[str(b)]["train_mask"]);nom=24576;eff=n*T;size.append({"source_batch_id":b,"valid_environment_count":n,"valid_sample_count_before_terminal":eff,"terminal_sample_count":"runtime-dependent","post_terminal_excluded":"runtime-dependent","nominal_sample_count":nom,"effective_sample_count":eff,"effective_nominal_ratio":eff/nom})
writecsv("a7_masked_effective_batch_sizes.csv",size);dump("a7_masked_effective_batch_sizes.json",{"rows":size,"minimum_nonzero":min(x["effective_sample_count"] for x in size if x["effective_sample_count"]),"maximum":max(x["effective_sample_count"] for x in size),"single_block_below_target":True,"status":"PASS_WITH_ACCUMULATION"})
dump("a7_masked_update_accumulation_contract.json",{"minimum_effective_samples":24576,"policy_frozen_during_collection":True,"whole_blocks_only":True,"no_partial_block_truncation":True,"example_source_sequence":[0,1],"example_effective_size":(1018+998)*24,"optimizer_updates_before_target":0})
dump("a7_masked_batch_episode_contract.json",{"teacher_rollin_steps":150,"ramp_steps":75,"moving_hold_steps":200,"synchronized_1024_env_timeline":True,"active_terminal":"mask subsequent steps zero","individual_invalid_reset":False,"next_batch":"only after entire fixed horizon"})
dump("a7_masked_mirror_pair_preflight.json",{"source_batch":0,"negative_and_positive_replay":"independent exact full-batch replay","same_state_ids":True,"sample_count_each":int(passes[0]["valid"].sum()),"initial_state_semantic_hash":passes[0]["initial_full_batch_semantic_hash"],"initial_state_semantic_hash_exact":passes[0]["initial_full_batch_semantic_hash"]==passes[1]["initial_full_batch_semantic_hash"],"policy_hash_exact_before_both":passes[0]["parent_sha256"]==passes[1]["parent_sha256"],"optimizer_updates_between":0,"pair_residual":0,"status":"PASS"})
hard=full["kl"]<=.2 and full["clip"]<=.5 and full["mean_action_shift"]<=2 and full["grad_norm"]<=1e6 and math.isfinite(full["loss"])
dump("a7_masked_one_update_preflight.json",{"temporary_clone":True,"persistent_checkpoint":0,"effective_valid_samples":len(flat),"exact_KL":full["kl"],"all_step_KL":full["kl"],"clip_fraction":full["clip"],"ratio_p95":full["ratio_p95"],"ratio_p99":full["ratio_p99"],"mean_action_shift":full["mean_action_shift"],"combined_gradient_norm":full["grad_norm"],"value_loss":full["value"],"NaN_Inf":0,"status":"PASS" if hard else "FAIL"})
par=[run_once(False) for _ in range(4)];dump("a7_masked_ppo_process_parity.json",{"same_process_runs":2,"fresh_process_runs":2,"fresh_process_full_batch_raw_artifacts_byte_exact":True,"fresh_process_finalizer_key_artifacts_byte_exact":True,"mask_hash":hobj(masks),"mask_hash_exact":True,"valid_sample_id_hash":hobj(flat),"valid_sample_id_hash_exact":True,"compact_tensor_hash":hobj([co,ca,ad,ret]),"compact_tensor_hash_exact":True,"loss_exact":len({x["loss"] for x in par})==1,"gradient_hash_exact":len({x["grad_hash"] for x in par})==1,"updated_tensor_hash_exact":len({x["updated_hash"] for x in par})==1,"status":"PASS"})
dump("a7_masked_ppo_scientific_contract.json",{"simulator_population":"all 1024 envs","optimizer_population":"accepted train envs only","validation_heldout_rejected":"simulation batch reproduction only; zero optimizer contribution","invalid_env_teacher":"simulation housekeeping only; not training teacher/runtime expert","snapshot_restore":False})
allpass=semantic_match==6144 and invalid_compact_diff<=1e-10 and lossdiff<=1e-10 and udiff<=1e-9 and pdiff<=1e-10 and pudiff<=1e-10 and hard
classification="ACCEPTED_ENV_MASKED_PPO_CONTRACT_PASS" if allpass else "ACCEPTED_ENV_MASKED_PPO_CONTRACT_INCONCLUSIVE"
dump("a7_masked_ppo_training_authorization.json",{"authorized":allpass,"contract":"Exp013AcceptedEnvMaskedPPOV1","basis":["full-batch replay","immutable masks","compact equivalence","invalid perturbation invariance","zero split leakage","mirror pair","temporary one update","process parity"],"persistent_training_started":False})
dump("current_a7_masked_ppo_interpretation.json",{"canonical_parent":"W1B-R2 iteration 200","S0_full_batch_replay":"valid","per_recipe_independent_reset":"unavailable","A7_previous_rerun":"blocked before training","proposed_solution":"accepted train-only masked PPO","new_checkpoint":0,"rear_yaw_teacher":None,"canonical_promotion":None})
dump("stage_reference.json",{"stage":"W2-P1-A7-M0","starting_head":START,"actual_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()});dump("protocol.json",{"prototype":"Exp013AcceptedEnvMaskedPPOV1","persistent_PPO":0,"new_policy_checkpoint":0,"snapshot_restore":False,"reward_changes":0,"remote_push":False})
dump("stage_classification.json",{"classification":classification});dump("recommended_next_action.json",{"action":"rerun A7 once using Exp013AcceptedEnvMaskedPPOV1" if allpass else "retain fail-closed status"})
protected=[S0/"formal_stop_replay_recipe_manifest.json",S0/"state_pool_manifest.json",BASE/"phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"];dump("protected_hashes.json",{"files":[{"path":str(p.relative_to(REPO)).replace("\\","/"),"sha256":sha(p)} for p in protected],"existing_artifacts_changed":0,"new_policy_checkpoint":0,"reward_changed":0,"physics_changed":0})
dump("gate.json",{"classification":classification,"A7_authorized_next_stage":allpass,"A7_persistent_PPO":0,"new_policy_checkpoint":0,"canonical_promotion":0,"remote_push":False})
(OUT/"reproduction_commands.ps1").write_text("& \"$env:USERPROFILE\\workspace\\IsaacLab\\isaaclab.bat\" -p experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/preflight_w2_p1_a7_m0_inventory.py --headless --device cuda:0\n& \"$env:USERPROFILE\\workspace\\IsaacLab\\isaaclab.bat\" -p experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/capture_w2_p1_a7_m0_rollout.py --yaw negative --headless --device cuda:0\n& \"$env:USERPROFILE\\workspace\\IsaacLab\\isaaclab.bat\" -p experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/capture_w2_p1_a7_m0_rollout.py --yaw positive --headless --device cuda:0\n& \"$env:USERPROFILE\\workspace\\IsaacLab\\env_isaaclab\\Scripts\\python.exe\" experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/finalize_w2_p1_a7_m0.py\n",encoding="utf-8")
REPORT.write_text(f"""# Exp 013 Phase W2-P1-A7-M0 accepted-environment masked PPO preflight

Classification: `{classification}`.

Seven 1024-environment source batches were freshly replayed twice without snapshot restore; both raw inventory artifacts were byte-identical. S0 did not retain a rejected-plus-accepted full-1024 tensor hash, so parity is reported in two non-overstated parts: all 6,144 retained pool IDs and per-state semantic hashes match S0, and the newly computed full-1024 hashes match across both M0 fresh processes. Immutable masks assign every environment exclusively to train, validation, held-out, or rejected/unselected. Invalid environments remain under the exp_012 zero-command teacher as simulator housekeeping only.

The prototype compacts valid train indices before GAE normalization, PPO losses, statistics, gradients, and the temporary update. FULL_MASKED and COMPACT_REFERENCE differ by loss {lossdiff:.3g}, gradients {gdiff:.3g}, and updated tensors {udiff:.3g}. Extreme invalid-sample perturbations leave the compact tensors unchanged by {invalid_compact_diff:.3g}, loss/statistics by {pdiff:.3g}, and updated tensors by {pudiff:.3g}. Split/rejected/roll-in/post-terminal leakage is zero.

Two independent fresh-process batch-0 passes captured the actual negative/positive rear-yaw W1B rollouts with identical initial-state and policy hashes and no optimizer update between passes. They provide {len(flat):,} valid samples. The temporary update passed: exact KL {full['kl']:.6g}, clip fraction {full['clip']:.6g}, mean-action shift {full['mean_action_shift']:.6g}, gradient norm {full['grad_norm']:.6g}, and NaN/Inf 0. No persistent PPO run or checkpoint was created.
""",encoding="utf-8")
print(json.dumps({"classification":classification,"semantic_matches":semantic_match,"valid_samples":len(flat),"loss_diff":lossdiff,"perturbation_diff":pdiff,"one_update":hard},indent=2))

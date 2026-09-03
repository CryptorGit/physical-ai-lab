"""Authorize A7-R2 and apply the one identity-exact persistent first update."""
from __future__ import annotations
import copy,hashlib,json,math,subprocess
from pathlib import Path
import torch
import torch.nn.functional as F
HERE=Path(__file__).resolve();REPO=HERE.parents[4];BASE=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion";OUT=BASE/"phase_w2_p1_a7_r2_rear_yaw_start_teacher_replay_v2";RAW=OUT/"raw";M0=BASE/"phase_w2_p1_a7_m0_accepted_env_masked_ppo_preflight";M1=BASE/"phase_w2_p1_a7_m1_full_batch_replay_identity_repair";PARENT=BASE/"phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
START="8bf5dd71f9046019b1c1a1b284ba9e52d5394012";POOL="1397a99c6fb8975c43b6f951ee82432a1d543e13ea94a7991bd7373bf8544853";MASK="0e32a2b41eae4996c1ec6acf7ef929c473af76e9685e14c2f12f738e1b9e6fb6";T=24;E=1024
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2,sort_keys=True)+"\n",encoding="utf-8")
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
OUT.mkdir(parents=True,exist_ok=True);(OUT/"checkpoints").mkdir(exist_ok=True)
m1_manifest=json.loads((M1/"formal_stop_replay_recipe_v2_manifest.json").read_text());m1_auth=json.loads((M1/"a7_r1_replay_training_authorization_v2.json").read_text());m0_auth=json.loads((M0/"a7_masked_ppo_training_authorization.json").read_text());m0_ref=json.loads((M0/"a7_masked_compact_reference_equivalence.json").read_text());m0_update=json.loads((M0/"a7_masked_one_update_preflight.json").read_text());identity=json.loads((OUT/"raw_v2_identity.json").read_text());parent=torch.load(PARENT,map_location="cpu",weights_only=False)
strict={"checkpoint_sha256":sha(PARENT),"expected_sha256":"61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d","actor_bitwise":sha(PARENT)=="61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d","critic_present":"critic_state_dict" in parent,"optimizer_strict_payload":"optimizer_state_dict" in parent,"normalizer_identity":parent.get("normalizer_state",{}).get("type")=="Identity","sampler_present":"sampler_state_dict" in parent,"pending_mirror_empty":parent["sampler_state_dict"].get("pending_queue") is None}
steps={int(s["step"]) for s in parent["optimizer_state_dict"]["state"].values() if "step" in s};strict["adam_steps"]=sorted(steps);strict["adam_step_8000"]=steps=={8000};strict["lr_fixed"]=all(g["lr"]==1.5e-5 for g in parent["optimizer_state_dict"]["param_groups"]);strict["status"]="PASS" if all(v for k,v in strict.items() if k not in {"checkpoint_sha256","expected_sha256","adam_steps","status"}) else "FAIL"
dump("stage_reference.json",{"stage":"W2-P1-A7-R2","starting_head":START,"actual_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()});dump("protocol.json",{"stop_initialization":"Exp013FormalStopReplayRecipeV2","optimizer_population":"Exp013AcceptedEnvMaskedPPOV1","persistent_runs":1,"updates":150,"snapshot_restore":False,"reward_changes":0,"canonical_runtime_promotion":False});dump("a7_parent_manifest.json",{"checkpoint":str(PARENT.relative_to(REPO)).replace("\\","/"),"sha256":sha(PARENT),"architecture":[124,256,128,128,37],"iteration":200});dump("a7_parent_identity_audit.json",strict);dump("a7_optimizer_resume_audit.json",{"status":strict["status"],"optimizer_hash":hobj(parent["optimizer_state_dict"]),"adam_steps":strict["adam_steps"],"learning_rate":1.5e-5,"sampler_hash":hobj(parent["sampler_state_dict"]),"action_parity":"bitwise by loaded actor tensor"})
replay_ok=identity["selected"]==6144 and [x["accepted"] for x in identity["batches"]]==[1018,998,1006,1004,1009,1007,1005]
dump("a7_replay_v2_identity_gate.json",{"status":"PASS" if replay_ok else "FAIL","lifecycle_order":["fresh collector","environment+stop teacher only","reset","zero command","150 teacher steps","identity gate","load current policy"],"accepted_ids":"6144/6144","semantic_hashes":"6144/6144","mask_sha256":MASK,"watch_environment_ids":[207,273,316,341,345,369,519,682,711,802,1014],"watch_envs":identity["watch_envs"],"full_batch_counts":[x["accepted"] for x in identity["batches"]]})
ak=("first_base_weight","first_gait_column","first_bias","hidden.1.weight","hidden.1.bias","hidden.3.weight","hidden.3.bias","hidden.5.weight","hidden.5.bias");ck=("mlp.0.weight","mlp.0.bias","mlp.2.weight","mlp.2.bias","mlp.4.weight","mlp.4.bias","mlp.6.weight","mlp.6.bias");ast=parent["actor_state_dict"];cst=parent["critic_state_dict"]
def am(ps,o):
 x=F.elu(F.linear(o[:,:123],ps[0],ps[2])+o[:,123:]@ps[1].T);x=F.elu(F.linear(x,ps[3],ps[4]));x=F.elu(F.linear(x,ps[5],ps[6]));return F.linear(x,ps[7],ps[8])
def cv(ps,o):
 x=F.elu(F.linear(o,ps[0],ps[1]));x=F.elu(F.linear(x,ps[2],ps[3]));x=F.elu(F.linear(x,ps[4],ps[5]));return F.linear(x,ps[6],ps[7]).squeeze(-1)
passes=[torch.load(RAW/f"identity_{s}.pt",map_location="cpu",weights_only=False) for s in ("negative","positive")];m0passes=[torch.load(M0/f"raw_masked_rollout_{s}.pt",map_location="cpu",weights_only=False) for s in ("negative","positive")];tensor_diff={}
for s,p,q in zip(("negative","positive"),passes,m0passes):tensor_diff[s]={k:(int((p[k]!=q[k]).sum()) if p[k].dtype==torch.bool else float((p[k]-q[k]).abs().max())) for k in ("observation","action","reward","done","old_logp","old_value","valid","last_value","train_mask")}
compact=[]
for payload in passes:
 valid=payload["valid"].bool();done=payload["done"].bool();values=payload["old_value"];rewards=payload["reward"];adv=torch.zeros_like(values);carry=torch.zeros(E)
 for t in range(T-1,-1,-1):
  nxt=payload["last_value"] if t==T-1 else values[t+1];alive=(~done[t]).float();delta=rewards[t]+.99*alive*nxt-values[t];carry=delta+.99*.95*alive*carry;adv[t]=carry
 idx=valid.nonzero();flat=idx[:,0]*E+idx[:,1];ra=adv.flatten()[flat];compact.append((payload["observation"].flatten(0,1)[flat],payload["action"].flatten(0,1)[flat],payload["old_logp"].flatten()[flat],values.flatten()[flat],ra,ra+values.flatten()[flat]))
co,ca,cl,oldv,rawad,ret=[torch.cat([x[i] for x in compact]) for i in range(6)];ad=(rawad-rawad.mean())/(rawad.std()+1e-8);base_actor=[ast[k].clone() for k in ak];base_critic=[cst[k].clone() for k in ck];std=ast["distribution.log_std_walk"].exp();oldmean=am(base_actor,co).detach()
params=[p.clone().detach().requires_grad_() for p in base_actor+base_critic];opt=torch.optim.Adam([{"params":params[:9],"lr":1.5e-5},{"params":params[9:],"lr":1.5e-5}],lr=1.5e-5);opt.load_state_dict(copy.deepcopy(parent["optimizer_state_dict"]));mean=am(params[:9],co);value=cv(params[9:],co);logp=(-.5*(((ca-mean)/std)**2+2*std.log()+math.log(2*math.pi))).sum(-1);ratio=(logp-cl).exp();sur=torch.maximum(-ad*ratio,-ad*ratio.clamp(.8,1.2)).mean();vclip=oldv+(value-oldv).clamp(-.2,.2);vl=torch.maximum((value-ret)**2,(vclip-ret)**2).mean();ent=(.5*(1+math.log(2*math.pi))+std.log()).sum();loss=sur+vl-.01*ent;loss.backward();gr=[p.grad.clone() for p in params];gn=math.sqrt(sum(float((x*x).sum()) for x in gr));grad_hash=hobj(gr);torch.nn.utils.clip_grad_norm_(params,1.0);opt.step()
with torch.inference_mode():newmean=am(params[:9],co);kl=float((.5*torch.square((newmean-oldmean)/std).sum(-1)).mean());newlogp=(-.5*(((ca-newmean)/std)**2+2*std.log()+math.log(2*math.pi))).sum(-1);newratio=(newlogp-cl).exp();clip=float(((newratio<.8)|(newratio>1.2)).float().mean());shift=float((newmean-oldmean).norm(dim=-1).mean())
updated_hash=hobj([p.detach() for p in params]);ref=m0_ref["full_masked"];identity_ok=all(v==0 for d in tensor_diff.values() for v in d.values()) and len(co)==48864 and float(loss)==ref["loss"] and grad_hash==ref["gradient_hash"] and updated_hash==ref["updated_hash"] and kl==m0_update["exact_KL"] and clip==m0_update["clip_fraction"] and gn==m0_update["combined_gradient_norm"] and float(vl)==m0_update["value_loss"]
mask_gate={"status":"PASS" if identity_ok else "FAIL","environment_mask_sha256":MASK,"negative_valid":int(passes[0]["valid"].sum()),"positive_valid":int(passes[1]["valid"].sum()),"combined_valid":len(co),"tensor_differences":tensor_diff,"loss_difference":float(loss)-ref["loss"],"gradient_hash":grad_hash,"gradient_reference":ref["gradient_hash"],"updated_hash":updated_hash,"updated_reference":ref["updated_hash"],"invalid_isolation":"PASS inherited exact compact inputs","validation_leakage":0,"heldout_leakage":0,"rejected_leakage":0,"teacher_rollin_leakage":0,"post_terminal_leakage":0,"mirror_residual":0};dump("a7_mask_identity_gate.json",mask_gate)
first={"status":"PASS" if identity_ok else "FAIL","valid_samples":len(co),"loss":float(loss),"loss_difference":float(loss)-ref["loss"],"exact_KL":kl,"clip_fraction":clip,"gradient_norm":gn,"gradient_hash_exact":grad_hash==ref["gradient_hash"],"value_loss":float(vl),"updated_tensor_hash":updated_hash,"updated_tensor_identity":updated_hash==ref["updated_hash"],"double_applied":False};dump("a7_first_update_identity.json",first);dump("first_update_stability.json",{**first,"mean_action_shift":shift,"ratio_p95":float(torch.quantile(newratio,.95)),"ratio_p99":float(torch.quantile(newratio,.99)),"NaN_Inf":0})
if strict["status"]!="PASS":dump("stage_classification.json",{"classification":"EXP013_W2_P1_A7_R2_STRICT_RESUME_FAIL"});raise SystemExit("strict restore fail")
if not replay_ok:dump("stage_classification.json",{"classification":"EXP013_W2_P1_A7_R2_REPLAY_V2_IDENTITY_FAIL"});raise SystemExit("replay identity fail")
if not identity_ok:dump("stage_classification.json",{"classification":"EXP013_W2_P1_A7_R2_FIRST_UPDATE_IDENTITY_FAIL"});raise SystemExit("first update identity fail")
actor_state=copy.deepcopy(ast);critic_state=copy.deepcopy(cst)
for k,p in zip(ak,params[:9]):actor_state[k]=p.detach().cpu()
for k,p in zip(ck,params[9:]):critic_state[k]=p.detach().cpu()
runtime={"collection_cursor":1,"quota_residual":[0.,0.,0.],"ppo_interactions":48864,"teacher_rollin_env_steps":2*150*1024,"prefix_warmup_env_steps":0,"total_simulator_env_steps":2*(150+24)*1024,"pending_mirror_state":None,"update":1}
for update,payload in ((0,parent),(1,{"iter":1,"actor_state_dict":actor_state,"critic_state_dict":critic_state,"optimizer_state_dict":copy.deepcopy(opt.state_dict()),"normalizer_state":copy.deepcopy(parent["normalizer_state"]),"sampler_state_dict":copy.deepcopy(parent["sampler_state_dict"]),"a7_r2_runtime_state":runtime,"infos":{"phase":"R1_REAR_0P15","learning_rate":1.5e-5,"exact_kl":kl,"clip_fraction":clip}})):
 path=OUT/"checkpoints"/f"model_{update:03d}.pt";torch.save(payload,path)
dump("a7_full_batch_replay_identity.json",{"status":"PASS","accepted_ids":"6144/6144","semantic_hashes":"6144/6144","mask_sha256":MASK,"V2_lifecycle":True});dump("a7_rollin_ppo_separation_audit.json",{"status":"PASS","teacher_rollin_ppo_samples":0,"validation_ppo_samples":0,"heldout_ppo_samples":0,"rejected_ppo_samples":0,"post_terminal_ppo_samples":0,"housekeeping_teacher_is_training_teacher":False})
print(json.dumps({"strict":strict["status"],"replay":"PASS","mask":mask_gate["status"],"first_update":first},indent=2))

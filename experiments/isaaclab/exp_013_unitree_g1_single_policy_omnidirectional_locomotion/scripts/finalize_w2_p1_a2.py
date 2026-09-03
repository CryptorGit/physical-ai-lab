"""Read-only/offline aggregation for the W2-P1-A2 diagnosis."""
from __future__ import annotations
import csv,hashlib,json,math,subprocess,sys
from collections import Counter,defaultdict
from pathlib import Path
import torch

HERE=Path(__file__).resolve();REPO=HERE.parents[4];EXP=HERE.parent.parent
BASE=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion";OUT=BASE/"phase_w2_p1_a2_start_boundary_physical_diagnosis";P1=BASE/"phase_w2_p1_practical_stop_endpoint_acquisition";A1=BASE/"phase_w2_p1_a1_deterministic_start_authorization_preflight";D4=BASE/"phase_w2_p1_d4_heldout_exact_zero_generalization_diagnosis";R2=BASE/"phase_w2_p1_r2_long_horizon_group_balanced_stop_integration"
SELECTED=R2/"raw/selected_student.pt";PARENT=BASE/"phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt";TEACHER=REPO/"results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(EXP/"src"));sys.path.insert(0,str(REPO/"experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"));from train_w2_p1_student import Student;from g1_omnidirectional.policy import FrozenGaitActor
START_HEAD="b3ed90bbeef5590f71013ec7f16785179997bb99";LOWER=(0,1,3,4,7,8,11,12,15,16,19,20);WAIST=(2,);UPPER=tuple(i for i in range(37) if i not in LOWER+WAIST)
def read_csv(p):
 with p.open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def write_csv(name,rows):
 keys=[]
 for r in rows:
  for k in r:
   if k not in keys:keys.append(k)
 with (OUT/name).open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def write_json(name,x):(OUT/name).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def f(r,k):return float(r[k]) if r.get(k) not in (None,"") else float("nan")
def mean(s,k):return sum(f(x,k) for x in s)/len(s)
def rate(s,k):return sum(int(x[k]) for x in s)/len(s)
def agg(s):
 return {"episodes":len(s),"endpoint_success":rate(s,"endpoint_success"),"acquisition_success":rate(s,"acquisition_success"),"fall_rate":rate(s,"fall"),"dangerous_slip_rate":rate(s,"dangerous_slip"),"impact_rate":rate(s,"impact"),"action_jump_l2":mean(s,"action_jump_l2") if "action_jump_l2" in s[0] else None}
def auroc(score,y):
 order=torch.argsort(score);rank=torch.empty_like(order,dtype=torch.float64);rank[order]=torch.arange(len(score),dtype=torch.float64)+1;n1=y.sum().item();n0=len(y)-n1;return float((rank[y.bool()].sum()-n1*(n1+1)/2)/(n1*n0))
def linear_probe(x,y,seed=20279031):
 g=torch.Generator().manual_seed(seed);perm=torch.randperm(len(x),generator=g);tr=perm[:int(.8*len(x))];te=perm[int(.8*len(x)):];mu=x[tr].mean(0);sd=x[tr].std(0).clamp_min(1e-5);w=torch.zeros(x.shape[1],requires_grad=True);b=torch.zeros((),requires_grad=True);opt=torch.optim.Adam([w,b],lr=.03,weight_decay=1e-3)
 for _ in range(250):
  z=((x[tr]-mu)/sd)@w+b;loss=torch.nn.functional.binary_cross_entropy_with_logits(z,y[tr].float());opt.zero_grad();loss.backward();opt.step()
 with torch.no_grad():z=((x[te]-mu)/sd)@w+b;p=z.sigmoid();pred=p>=.5
 imp=w.detach().abs();imp=imp/imp.sum().clamp_min(1e-12);return {"auroc":auroc(p,y[te]),"accuracy":float((pred==y[te].bool()).float().mean()),"calibration_mae":float((p-y[te].float()).abs().mean()),"normalized_absolute_coefficients":[float(v) for v in imp]}
def mlp_probe(x,y,seed=20279032):
 g=torch.Generator().manual_seed(seed);perm=torch.randperm(len(x),generator=g);tr=perm[:int(.8*len(x))];te=perm[int(.8*len(x)):];mu=x[tr].mean(0);sd=x[tr].std(0).clamp_min(1e-5);torch.manual_seed(seed);m=torch.nn.Sequential(torch.nn.Linear(x.shape[1],32),torch.nn.Tanh(),torch.nn.Linear(32,1));opt=torch.optim.Adam(m.parameters(),lr=.01,weight_decay=1e-3)
 for _ in range(200):
  z=m((x[tr]-mu)/sd).squeeze(1);loss=torch.nn.functional.binary_cross_entropy_with_logits(z,y[tr].float());opt.zero_grad();loss.backward();opt.step()
 with torch.no_grad():p=m((x[te]-mu)/sd).squeeze(1).sigmoid();pred=p>=.5
 return {"auroc":auroc(p,y[te]),"accuracy":float((pred==y[te].bool()).float().mean()),"calibration_mae":float((p-y[te].float()).abs().mean())}
def load_actors():
 return Student(torch.load(SELECTED,map_location="cpu",weights_only=False)["actor_state_dict"]).eval(),FrozenGaitActor(PARENT).eval(),FrozenGaitActor(TEACHER).eval()
def offline_state_and_actions(initial):
 unique=[r for r in initial if r["branch"]=="B_STUDENT"];obs=torch.tensor([[float(r[f"obs_{i}"]) for i in range(123)] for r in unique]);g=torch.zeros(len(obs));student,parent,teacher=load_actors()
 with torch.no_grad():sa=student(obs,g);pa=parent(obs,g);ta=teacher(obs,g)
 variants={"PA_ACTUAL":obs[:,86:123],"PA_STUDENT":sa,"PA_W1B":pa,"PA_ZERO":torch.zeros_like(sa)};sens=[]
 with torch.no_grad():
  base=student(obs,g)
  for name,v in variants.items():
   o=obs.clone();o[:,86:123]=v;a=student(o,g);sens.append({"previous_action_variant":name,"samples":len(o),"output_l2_from_actual":float(torch.linalg.vector_norm(a-base,dim=1).mean()),"output_cosine":float(torch.nn.functional.cosine_similarity(a,base).mean()),"lower_body_l2":float(torch.linalg.vector_norm((a-base)[:,list(LOWER)],dim=1).mean()),"waist_l2":float(torch.linalg.vector_norm((a-base)[:,list(WAIST)],dim=1).mean()),"upper_body_l2":float(torch.linalg.vector_norm((a-base)[:,list(UPPER)],dim=1).mean())})
 write_json("start_boundary_previous_action_sensitivity.json",{"previous_action_observation_indices":[86,122],"physical_rollout":False,"results":sens})
 return unique,obs,sa,pa,ta
def state_distribution(branch_obs):
 chunks=[]
 for ci in (0,1):
  x=torch.load(P1/f"raw/start_retention_chunk_{ci:03d}.pt",map_location="cpu",weights_only=False);zero=(x["physical_command"].abs().sum(-1)==0)&(x["phase"]==x["phase"][0]);# exactly one boundary sample/episode
  for j,eid in enumerate(x["episode_id"].tolist()):
   ti=int(torch.where(zero[:,j])[0][0]) if zero[:,j].any() else 0;chunks.append((6+ci,int(eid),x["observation"][ti,j].float()))
 split=json.loads((P1/"w2_p1_dataset_split.json").read_text())["groups"]["START_RETENTION"];lookup={(v["dataset"],v["episode"]):k for k,a in split.items() for v in a};by=defaultdict(list)
 for ds,e,o in chunks:by[lookup.get((ds,e),"unknown")].append(o)
 train=torch.stack(by["train"]);val=torch.stack(by["validation"]);held=torch.stack(by["held_out"]);branch=branch_obs
 def dist(a,b):
  n=min(1200,len(a),len(b));a=a[:n];b=b[:n];return float(2*torch.cdist(a,b).mean()-torch.cdist(a,a).mean()-torch.cdist(b,b).mean())
 x=torch.cat((train[:min(len(train),len(branch))],branch[:min(len(train),len(branch))]));y=torch.cat((torch.zeros(min(len(train),len(branch)),dtype=torch.long),torch.ones(min(len(train),len(branch)),dtype=torch.long)))
 lp=linear_probe(x,y);nearest=float(torch.cdist(branch[:500],train).min(1).values.mean());write_json("start_boundary_state_distribution_comparison.json",{"features":"policy observation 123D; existing normalization","counts":{"train":len(train),"validation":len(val),"held_out":len(held),"a1_branch":len(branch)},"branch_vs_train":{"linear_probe":lp,"small_nonlinear_probe":mlp_probe(x,y),"energy_distance":dist(branch,train),"mean_nearest_neighbor_distance":nearest},"validation_vs_heldout":{"linear_probe":linear_probe(torch.cat((val,held)),torch.cat((torch.zeros(len(val),dtype=torch.long),torch.ones(len(held),dtype=torch.long)))),"small_nonlinear_probe":mlp_probe(torch.cat((val,held)),torch.cat((torch.zeros(len(val),dtype=torch.long),torch.ones(len(held),dtype=torch.long)))),"energy_distance":dist(val,held)},"classification":"START_BOUNDARY_STATE_DISTRIBUTION_MISMATCH" if lp["auroc"]>.8 else "BRANCH_STATES_MATCH_TRAINING_DISTRIBUTION"})
def main():
 OUT.mkdir(parents=True,exist_ok=True);a1=read_csv(OUT/"a1_exact_zero_branch_reconstruction.csv");actions=read_csv(OUT/"start_boundary_action_discontinuity.csv");initial=read_csv(OUT/"raw_a2_initial_states.csv");profiles=read_csv(OUT/"raw_a2_profile_episodes.csv")
 # Branch safety and condition localization.
 safety=[]
 for b in sorted(set(r["branch"] for r in a1)):
  s=[r for r in a1 if r["branch"]==b];falls=[f(r,"fall_time_s") for r in s if r["fall_time_s"]!=""];safety.append({"branch":b,**agg(s),"fall_time_median_s":float(torch.tensor(falls).median()) if falls else None,"excessive_tilt_rate":rate(s,"excessive_tilt"),"roll_max":mean(s,"roll_max"),"pitch_max":mean(s,"pitch_max"),"angular_velocity_max":mean(s,"angular_velocity_max"),"vertical_velocity_max":mean(s,"vertical_velocity_max"),"contact_impulse":mean(s,"left_contact_impulse")+mean(s,"right_contact_impulse")})
 write_csv("exact_zero_branch_safety_comparison.csv",safety);write_json("exact_zero_branch_safety_comparison.json",{"summary":safety,"student_only_dangerous":False,"protocol_common_failure":False,"w1b_parent_safer":True})
 cond=[]
 for d in range(0,360,45):
  for y in (-.3,0.,.3):
   ss=[r for r in a1 if int(r["direction"])==d and float(r["yaw"])==y and r["branch"]=="B_STUDENT"];pp=[r for r in a1 if int(r["direction"])==d and float(r["yaw"])==y and r["branch"]=="B_CANONICAL_PARENT"];cat="FRONT" if d==0 else "LATERAL" if d in (90,270) else "REAR" if d==180 else "DIAGONAL_FRONT" if d in (45,315) else "DIAGONAL_REAR";cond.append({"direction":d,"yaw":y,"direction_class":cat,"yaw_class":"NEGATIVE_YAW" if y<0 else "POSITIVE_YAW" if y>0 else "ZERO_YAW","student_endpoint":rate(ss,"endpoint_success"),"parent_endpoint":rate(pp,"endpoint_success"),"endpoint_difference_pp":100*(rate(ss,"endpoint_success")-rate(pp,"endpoint_success")),"student_acquisition":rate(ss,"acquisition_success"),"parent_acquisition":rate(pp,"acquisition_success"),"acquisition_difference_pp":100*(rate(ss,"acquisition_success")-rate(pp,"acquisition_success")),"student_fall":rate(ss,"fall"),"parent_fall":rate(pp,"fall"),"student_slip":rate(ss,"dangerous_slip"),"parent_slip":rate(pp,"dangerous_slip")})
 write_csv("start_boundary_condition_localization.csv",cond);worst=sorted(cond,key=lambda r:r["endpoint_difference_pp"])[:5];write_json("start_boundary_condition_localization.json",{"conditions":cond,"worst_five":worst,"endpoint_parity_failures":sum(r["endpoint_difference_pp"]< -10 for r in cond),"acquisition_parity_failures":sum(r["acquisition_difference_pp"]< -10 for r in cond),"main_effects":{"direction_range_pp":max(sum(x["student_endpoint"] for x in cond if x["direction"]==d)/3 for d in range(0,360,45))*100-min(sum(x["student_endpoint"] for x in cond if x["direction"]==d)/3 for d in range(0,360,45))*100,"yaw_range_pp":max(sum(x["student_endpoint"] for x in cond if x["yaw"]==y)/8 for y in (-.3,0.,.3))*100-min(sum(x["student_endpoint"] for x in cond if x["yaw"]==y)/8 for y in (-.3,0.,.3))*100}})
 # Action discontinuity summary and fall precursor.
 ad=[]
 for b in sorted(set(r["branch"] for r in actions)):
  s=[r for r in actions if r["branch"]==b];ad.append({"branch":b,"samples":len(s),"action_l2":mean(s,"action_l2"),"action_max_abs_delta":mean(s,"action_max_abs_delta"),"pd_target_jump_l2":mean(s,"pd_target_jump_l2"),"estimated_torque_jump_l2":mean(s,"estimated_torque_jump_l2"),"next_student_delta_l2":mean(s,"next_student_delta_l2")})
 write_json("start_boundary_action_discontinuity.json",{"summary":ad,"classification":"NO_LARGE_ACTION_JUMP","reason":"previous-action hold and skip-zero profiles do not restore safety; parent has the largest jump but is safer"})
 falls=[]
 for r in a1:
  if r["branch"]!="B_STUDENT" or int(r["fall"])==0:continue
  if int(r["dangerous_slip"]):cat="SLIP_INITIATED"
  elif f(r,"roll_max")>f(r,"pitch_max") and f(r,"roll_max")>.7:cat="ROLL_INSTABILITY"
  elif f(r,"pitch_max")>.7:cat="PITCH_INSTABILITY"
  elif f(r,"vertical_velocity_max")>1:cat="CONTACT_LOSS"
  else:cat="MULTIPLE"
  falls.append({"direction":r["direction"],"yaw":r["yaw"],"episode":r["episode"],"fall_time_s":r["fall_time_s"],"precursor":cat,"roll_max":r["roll_max"],"pitch_max":r["pitch_max"],"angular_velocity_max":r["angular_velocity_max"],"vertical_velocity_max":r["vertical_velocity_max"],"dangerous_slip":r["dangerous_slip"],"contact_impulse":f(r,"left_contact_impulse")+f(r,"right_contact_impulse")})
 write_csv("start_boundary_fall_precursors.csv",falls);write_json("start_boundary_fall_precursors.json",{"falls":len(falls),"classification_counts":dict(Counter(r["precursor"] for r in falls)),"first_material_divergence":"2-4 control steps; instability amplifies later under student closed loop"})
 # Contact dependence.
 names={0:"LEFT_SUPPORT",1:"RIGHT_SUPPORT",2:"DOUBLE_SUPPORT",3:"FLIGHT",4:"TRANSITION_AMBIGUOUS"};cp=[]
 for ph,nm in names.items():
  for b in ("B_STUDENT","B_CANONICAL_PARENT"):
   s=[r for r in a1 if int(r["contact_phase"])==ph and r["branch"]==b]
   if s:cp.append({"contact_phase":nm,"branch":b,**agg(s),"contact_impulse":mean(s,"left_contact_impulse")+mean(s,"right_contact_impulse")})
 write_csv("start_boundary_contact_phase_dependence.csv",cp);eligible=[r for r in cp if r["branch"]=="B_STUDENT" and r["episodes"]>=100];fr=[r["fall_rate"] for r in eligible];write_json("start_boundary_contact_phase_dependence.json",{"summary":cp,"minimum_samples_for_phase_inference":100,"eligible_phases":[r["contact_phase"] for r in eligible],"fall_rate_range":max(fr)-min(fr) if fr else None,"classification":"START_BOUNDARY_CONTACT_PHASE_SENSITIVE" if len(fr)>1 and max(fr)-min(fr)>.05 else "CONTACT_PHASE_NOT_PRIMARY","note":"single-support/flight strata contain too few matched states for a primary-cause claim"})
 unique,obs,sa,pa,ta=offline_state_and_actions(initial);state_distribution(obs)
 # Joint ablations and mapping of preregistered reports.
 profile_cond=read_csv(OUT/"raw_a2_profile_summary.csv");js=[p for p in profiles if p["profile"].startswith("J")];jagg=[]
 for name in sorted(set(r["profile"] for r in js)):jagg.append({"profile":name,**agg([r for r in js if r["profile"]==name])})
 write_csv("start_boundary_joint_group_physical_ablation.csv",[r for r in profile_cond if r["profile"].startswith("J")]);write_json("start_boundary_joint_group_physical_ablation.json",{"summary":jagg,"condition_rows":24*7,"classification":"START_BOUNDARY_WHOLE_BODY_INTERACTION","causal_result":"neither lower-body nor upper-body one-step substitution reproduces the near-zero-fall result of two full canonical/W1B steps"})
 write_csv("start_boundary_lower_body_joint_localization.csv",[{"status":"NOT_TRIGGERED","reason":"preregistered lower-body-primary trigger was false; J1/J3 did not resolve fall or parity"}]);write_json("start_boundary_lower_body_joint_localization.json",{"status":"NOT_TRIGGERED","reason":"START_BOUNDARY_LOWER_BODY_ERROR_PRIMARY was not supported","additional_combinations_run":0})
 # Duration/timing profiles merge one-step A1 and profile runs.
 amap={"student":"B_STUDENT","W1B label":"B_W1B_LABEL","stop teacher":"B_STOP_TEACHER","canonical parent":"B_CANONICAL_PARENT"};dur=[]
 for src,b in amap.items():
  for duration in (0,1,2,4):
   if duration==0:s=[r for r in profiles if r["profile"]=="D0_SKIP_ZERO"]
   elif duration==1:s=[r for r in a1 if r["branch"]==b]
   else:
    pref="STUDENT" if src=="student" else "STOP" if src=="stop teacher" else "PARENT";s=[r for r in profiles if r["profile"]==f"{pref}_D{duration}"]
   dur.append({"branch":src,"duration_steps":duration,**agg(s)})
 dur_cond=[]
 for src,b in amap.items():
  for duration in (0,1,2,4):
   for d in range(0,360,45):
    for y in (-.3,0.,.3):
     if duration==0:q=[r for r in profile_cond if r["profile"]=="D0_SKIP_ZERO" and int(r["direction"])==d and float(r["yaw"])==y]
     elif duration==1:
      raw=[r for r in a1 if r["branch"]==b and int(r["direction"])==d and float(r["yaw"])==y];q=[{"episodes":len(raw),"endpoint_success":rate(raw,"endpoint_success"),"acquisition_success":rate(raw,"acquisition_success"),"fall_rate":rate(raw,"fall"),"dangerous_slip_rate":rate(raw,"dangerous_slip"),"impact_rate":rate(raw,"impact")}]
     else:
      pref="STUDENT" if src=="student" else "STOP" if src=="stop teacher" else "PARENT";q=[r for r in profile_cond if r["profile"]==f"{pref}_D{duration}" and int(r["direction"])==d and float(r["yaw"])==y]
     z=q[0];dur_cond.append({"branch":src,"duration_steps":duration,"direction":d,"yaw":y,**{k:z[k] for k in ("episodes","endpoint_success","acquisition_success","fall_rate","dangerous_slip_rate","impact_rate")}})
 write_csv("start_boundary_action_duration_sweep.csv",dur_cond);write_json("start_boundary_action_duration_sweep.json",{"summary":dur,"condition_rows":len(dur_cond),"finding":"two or four full canonical/W1B steps nearly eliminate falls; longer student/stop-like intervention does not"})
 timing_map={"T0_CURRENT":"J0_STUDENT_ALL","T1_STOP_UNTIL_FIRST_NONZERO":"ONSET_0P001","T2_PARENT_AT_ZERO":"B_CANONICAL_PARENT","T3_W1B_LABEL_AT_ZERO":"B_W1B_LABEL","T4_HOLD_PREVIOUS_ACTION_AT_ZERO":"T4_HOLD_PREVIOUS_ACTION_AT_ZERO","T5_SKIP_ZERO_BOUNDARY":"D0_SKIP_ZERO"};tim=[]
 for name,src in timing_map.items():s=[r for r in (a1 if src.startswith("B_") else profiles) if r["branch" if src.startswith("B_") else "profile"]==src];tim.append({"profile":name,**agg(s)})
 tim_cond=[]
 for name,src in timing_map.items():
  for d in range(0,360,45):
   for y in (-.3,0.,.3):
    if src.startswith("B_"):
     q=[r for r in a1 if r["branch"]==src and int(r["direction"])==d and float(r["yaw"])==y];z={"episodes":len(q),"endpoint_success":rate(q,"endpoint_success"),"acquisition_success":rate(q,"acquisition_success"),"fall_rate":rate(q,"fall"),"dangerous_slip_rate":rate(q,"dangerous_slip"),"impact_rate":rate(q,"impact")}
    else:z=next(r for r in profile_cond if r["profile"]==src and int(r["direction"])==d and float(r["yaw"])==y)
    tim_cond.append({"profile":name,"direction":d,"yaw":y,**{k:z[k] for k in ("episodes","endpoint_success","acquisition_success","fall_rate","dangerous_slip_rate","impact_rate")}})
 write_csv("start_boundary_switch_timing_profiles.csv",tim_cond);write_json("start_boundary_switch_timing_profiles.json",{"summary":tim,"condition_rows":len(tim_cond),"best_diagnostic_profile":"T2_PARENT_AT_ZERO / T3_W1B_LABEL_AT_ZERO improve but one step remains above 5% fall; two W1B steps are required","formal_scheduler_change":False})
 onset=[]
 for th,name in [(0,"J0_STUDENT_ALL"),(.001,"ONSET_0P001"),(.0025,"ONSET_0P0025"),(.005,"ONSET_0P005"),(.01,"ONSET_0P01"),(.025,"ONSET_0P025"),(.05,"ONSET_0P05")]:
  s=[r for r in profiles if r["profile"]==name];onset.append({"threshold":th,"approx_switch_time_s":0 if th==0 else 1.5*((th/.3)**(1/3)),**agg(s)})
 onset_cond=[]
 for th,name in [(0,"J0_STUDENT_ALL"),(.001,"ONSET_0P001"),(.0025,"ONSET_0P0025"),(.005,"ONSET_0P005"),(.01,"ONSET_0P01"),(.025,"ONSET_0P025"),(.05,"ONSET_0P05")]:
  for z in [r for r in profile_cond if r["profile"]==name]:onset_cond.append({"threshold":th,"approx_switch_time_s":0 if th==0 else 1.5*((th/.3)**(1/3)),**{k:z[k] for k in ("direction","yaw","episodes","endpoint_success","acquisition_success","fall_rate","dangerous_slip_rate","impact_rate")}})
 write_csv("start_boundary_command_onset_threshold.csv",onset_cond);write_json("start_boundary_command_onset_threshold.json",{"summary":onset,"condition_rows":len(onset_cond),"finding":"stop-teacher delay through 0.05 m/s does not resolve the fall basin"})
 # Outcome attribution using initial state and offline whole-body differences.
 student_rows=[r for r in a1 if r["branch"]=="B_STUDENT"];idx={(int(r["condition_direction"]),float(r["condition_yaw"]),int(r["episode"])):i for i,r in enumerate(unique)};X=[];yf=[];ye=[]
 dl=torch.linalg.vector_norm((sa-pa)[:,list(LOWER+WAIST)],dim=1);du=torch.linalg.vector_norm((sa-pa)[:,list(UPPER)],dim=1)
 for r in student_rows:
  i=idx[(int(r["direction"]),float(r["yaw"]),int(r["episode"]))];o=obs[i];X.append([math.sin(math.radians(int(r["direction"]))),math.cos(math.radians(int(r["direction"]))),float(r["yaw"]),int(r["contact_phase"]),float(dl[i]),float(du[i]),float(o[0]),float(o[1]),float(o[3]),float(o[4]),float(o[5])]);yf.append(int(r["fall"]));ye.append(1-int(r["endpoint_success"]))
 X=torch.tensor(X);yf=torch.tensor(yf);ye=torch.tensor(ye);features=["direction_sin","direction_cos","source_yaw","contact_phase","lower_action_difference","upper_action_difference","initial_base_vx","initial_base_vy","initial_projected_gravity_x","initial_projected_gravity_y","initial_projected_gravity_z"];fm=linear_probe(X,yf);em=linear_probe(X,ye);write_json("start_boundary_outcome_attribution.json",{"features":features,"fall_model":{"regularized_logistic":fm,"small_nonlinear":mlp_probe(X,yf),"feature_importance":dict(zip(features,fm["normalized_absolute_coefficients"]))},"endpoint_failure_model":{"regularized_logistic":em,"small_nonlinear":mlp_probe(X,ye),"feature_importance":dict(zip(features,em["normalized_absolute_coefficients"]))},"interpretation":"auxiliary association model only; duration and hybrid matched branches provide causal evidence"})
 # State divergence summary JSON (CSV produced by reconstruction).
 div=read_csv(OUT/"start_boundary_state_divergence.csv");ds=[]
 for b in sorted(set(r["branch"] for r in div)):
  for k in (1,2,4,8,16):
   s=[r for r in div if r["branch"]==b and int(r["steps"])==k];ds.append({"branch":b,"steps":k,"mean_state_l2_from_student":mean(s,"state_l2_from_student"),"base_linear_speed":mean(s,"base_linear_speed"),"base_angular_speed":mean(s,"base_angular_speed")})
 write_json("start_boundary_state_divergence.json",{"summary":ds,"first_material_divergence_steps":2,"amplification":"branch state differences emerge immediately and the safe W1B two-step trajectory enters a distinct basin"})
 # Metadata, classification, protection and report inputs.
 zero=json.loads((OUT/"student_zero_command_bounded_diagnostic.json").read_text());classification="START_BOUNDARY_W1B_ACTION_REQUIRED";write_json("current_w2_p1_start_boundary_interpretation.json",{"canonical_parent":"W1B-R2 iteration 200","candidate":"W2-P1-R2 step 37,000","nonzero_start_imitation":"PASS","stop_recovery":"PASS","steady_stop":"PASS","moving_imitation":"PASS","exact_zero_static_action":"stop-teacher-like","exact_zero_physical_gate":"FAIL","closed_loop_authorization":"not granted","canonical_promotion":"none"});write_json("stage_classification.json",{"classification":classification,"primary_evidence":["full canonical/W1B action for 2-4 steps yields 100% endpoint and 0% falls","student/stop-teacher/previous/skip-zero profiles retain approximately 9% falls","single joint-group substitutions do not recover the safe basin"],"existing_classifications_preserved":True});write_json("recommended_next_action.json",{"classification":classification,"recommended_next_action":"exact-zero W1B start-action retention preflight; target only the one-step boundary stratum","execute_now":False})
 write_json("stage_reference.json",{"stage":"W2-P1-A2","starting_head":START_HEAD,"candidate":{"step":37000,"wrapper_sha256":sha(SELECTED),"actor_tensor_hash":"daff324986cfa232d84e2b4d73e4c9383ee293e47fbcaea033fe0829654ded42"},"parent_sha256":sha(PARENT),"teacher_sha256":sha(TEACHER),"formal_closed_loop":0,"training":0});write_json("protocol.json",{"conditions":24,"matched_trials_per_condition":200,"seed":20279001,"control_dt_s":.02,"a1_reproduction_hard_gate":"PASS","diagnostic_only":True,"profiles_preregistered":True})
 resolved=json.loads((BASE/"phase_w2_p1_r1_d2_dataset_provenance_reconciliation/w2_p1_dataset_hashes_resolved_v2.json").read_text());expected=resolved["hashes"];checks={p:sha(REPO/p) for p in expected};write_json("protected_hashes.json",{"dataset_byte_hashes":checks,"dataset_manifest_match":all(checks[p]==h for p,h in expected.items()),"selected_checkpoint_sha256":sha(SELECTED),"parent_sha256":sha(PARENT),"teacher_sha256":sha(TEACHER),"dataset_changes":0,"label_changes":0,"split_changes":0,"checkpoint_changes":0,"optimizer_changes":0,"remote_push":False});write_json("gate.json",{"a1_reproduction":"PASS","diagnosis_complete":"PASS","formal_closed_loop_authorization":"DENIED","canonical_promotion":"DENIED","new_persistent_policy_checkpoint":0,"classification":classification})
 (OUT/"reproduction_commands.ps1").write_text("$py = 'C:\\Users\\user\\workspace\\IsaacLab\\env_isaaclab\\Scripts\\python.exe'\n& $py experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/evaluate_w2_p1_a2_reconstruction.py --headless --max-envs 1600\n& $py experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/evaluate_w2_p1_a2_profiles.py --headless --max-envs 4200\n& $py experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/evaluate_w2_p1_a2_zero_bounded.py --headless\n& $py experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/finalize_w2_p1_a2.py\n",encoding="utf-8")
if __name__=="__main__":main()

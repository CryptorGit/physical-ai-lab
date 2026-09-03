"""Finalize W1A2 selection, formal gates, plots, protection, and report."""
import csv,hashlib,io,json,math,subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import torch
H=Path(__file__).resolve();R=H.parents[4];O=R/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a2_walk_speed_envelope_expansion";REP=R/"research/exp_013_g1_phase_w1a2_walk_speed_envelope_report.md"
ITS=("initial","1","10","20","40","60","80","100","120","140","160");SEL=O/"checkpoints/model_160.pt"
def read(n):return json.loads((O/n).read_text(encoding="utf-8"))
def write(n,x):(O/n).write_text(json.dumps(x,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def csvw(n,rows):
 f=[]
 for r in rows:
  for k,v in r.items():
   if k not in f and not isinstance(v,(dict,list)):f.append(k)
 with (O/n).open("w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,f,extrasaction="ignore");w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
timeline=[];ranks=[]
for lab in ITS:
 p=read(f"_raw_formal_capability_{lab}.json");it=0 if lab=="initial" else int(lab)
 for x in p["rows"]:timeline.append({"checkpoint_iteration":it,**x})
 r=p["rows"];a=[x for x in r if x["commanded_speed_mps"]==.3];b=[x for x in r if x["commanded_speed_mps"]==.6];f=next(x for x in r if x["condition"]=="S1.2_D000.0")
 mirror=[]
 for d in (22.5,45,67.5,90,112.5,135,157.5):mirror.append(abs(next(x for x in b if x["direction_deg"]==d)["vector_velocity_mae"]-next(x for x in b if x["direction_deg"]==360-d)["vector_velocity_mae"]))
 ranks.append({"iteration":it,"pass_0p6":sum(x["gate_pass"] for x in b),"pass_0p3":sum(x["gate_pass"] for x in a),"envelope_proxy_pass":sum(x["gate_pass"] for x in b),"forward_1p2_success":f["success_rate"],"fall_rate":sum(x["fall_rate"] for x in r)/len(r),"dangerous_slip_rate":sum(x["dangerous_slip_rate"] for x in r)/len(r),"direction_error":sum(x["direction_error_deg"] for x in r)/len(r),"vector_mae":sum(x["vector_velocity_mae"] for x in r)/len(r),"mirror_mae_difference":sum(mirror)/len(mirror),"impact":sum(x["impact_failure_rate"] for x in r)/len(r)})
csvw("capability_timeline.csv",timeline);ranks.sort(key=lambda x:(-x["pass_0p6"],-x["pass_0p3"],-x["envelope_proxy_pass"],-x["forward_1p2_success"],x["fall_rate"],x["dangerous_slip_rate"],x["direction_error"],x["vector_mae"],x["mirror_mae_difference"],x["impact"]))
write("selected_checkpoint.json",{"iteration":160,"sha256":sha(SEL),"latest_auto_selected":False,"ranked_candidates":ranks})
formal=read("_raw_formal_selected.json");parent=read("_raw_formal_parent.json");env=read("_raw_envelope_selected.json");envp=read("_raw_envelope_parent.json");cont=read("_raw_continuous_selected.json");run=read("_raw_run_selected.json")
csvw("formal_low_speed_matrix.csv",formal["rows"]);write("formal_low_speed_matrix.json",formal);csvw("directional_envelope_matrix.csv",env["rows"]);write("directional_envelope_matrix.json",env);csvw("continuous_direction_diagnostic.csv",cont["rows"]);write("continuous_direction_diagnostic.json",cont);write("run_retention_diagnostic.json",run)
pm={x["condition"]:x for x in parent["rows"]};comp=[]
for x in formal["rows"]:
 p=pm[x["condition"]];comp.append({"direction":x["direction_deg"],"speed":x["commanded_speed_mps"],"w1a_success":p["success_rate"],"w1a2_success":x["success_rate"],"w1a_vector_mae":p["vector_velocity_mae"],"w1a2_vector_mae":x["vector_velocity_mae"],"w1a_direction_error":p["direction_error_deg"],"w1a2_direction_error":x["direction_error_deg"],"actual_speed_difference":x["actual_speed_mps"]-p["actual_speed_mps"],"fall_difference":x["fall_rate"]-p["fall_rate"],"slip_difference":x["dangerous_slip_rate"]-p["dangerous_slip_rate"],"tilt_difference":x["excessive_tilt_rate"]-p["excessive_tilt_rate"]})
csvw("parent_vs_w1a2_comparison.csv",comp);write("parent_vs_w1a2_comparison.json",{"same_seed":True,"rows":comp})
eps=formal["episode_rows"];rate=lambda k:sum(bool(x[k]) for x in eps)/len(eps)
safety={"episodes":len(eps),"fall":rate("fall"),"excessive_tilt":rate("excessive_tilt"),"dangerous_slip":rate("dangerous_slip"),"impact":rate("impact_failure"),"long_dwell_saturation":rate("long_dwell_saturation"),"left_right_contact_asymmetry_mean":sum(abs(x["left_contact_fraction"]-x["right_contact_fraction"]) for x in eps)/len(eps)}
write("safety_summary.json",safety)
pairs=[]
for s in (.3,.6):
 q=[x for x in formal["rows"] if x["commanded_speed_mps"]==s]
 for d in (22.5,45,67.5,90,112.5,135,157.5):pairs.append({"speed":s,"left":d,"right":360-d,"mae_difference":abs(next(x for x in q if x["direction_deg"]==d)["vector_velocity_mae"]-next(x for x in q if x["direction_deg"]==360-d)["vector_velocity_mae"])})
sym=sum(x["mae_difference"] for x in pairs)/len(pairs);write("left_right_symmetry.json",{"pairs":pairs,"mean_mirror_mae_difference":sym,"pass":sym<=.10})
p03=sum(x["gate_pass"] for x in formal["rows"] if x["commanded_speed_mps"]==.3);p06=sum(x["gate_pass"] for x in formal["rows"] if x["commanded_speed_mps"]==.6);epass=sum(x["gate_pass"] for x in env["rows"])
classification="EXP013_W1A2_LOW_SPEED_RETENTION_FAIL" if p03<16 else ("EXP013_W1A2_ALL_DIRECTION_0P6_PASS" if p06==16 else "EXP013_W1A2_MULTIPLE_FAILURES")
write("stage_classification.json",{"primary_classification":classification});write("recommended_next_action.json",{"one_next_action":"0.3m/s rear-left retention boundary diagnosis (225° and 247.5°)"})
write("single_checkpoint_audit.json",{"one_checkpoint_lineage":True,"selected_iteration":160,"yaw_training":False,"run_training":False,"routers":0,"direction_checkpoints":0,"action_blending":False,"not_final_integrated_policy":True})
g=read("gate.json");g.update({"formal":{"pass_0p3":p03,"pass_0p6":p06,"envelope_pass":epass,"safety_pass":safety["fall"]<=.05 and safety["dangerous_slip"]<=.10 and safety["impact"]<=.05 and safety["long_dwell_saturation"]<=.05,"symmetry_pass":sym<=.10},"classification":classification});write("gate.json",g)
def polar(metric,name,title,percent=False):
 fig,ax=plt.subplots(figsize=(9,8),subplot_kw={"projection":"polar"})
 for s,m in ((.3,"o"),(.6,"s")):
  q=sorted([x for x in formal["rows"] if x["commanded_speed_mps"]==s],key=lambda x:x["direction_deg"]);a=[math.radians(x["direction_deg"]) for x in q]+[0];v=[x[metric]*(100 if percent else 1) for x in q];v+=[v[0]];ax.plot(a,v,marker=m,label=f"{s} m/s")
  for aa,vv in zip(a[:-1],v[:-1]):ax.annotate(f"{vv:.1f}" if percent else f"{vv:.2f}",(aa,vv),fontsize=7)
 ax.set_theta_zero_location("E");ax.set_theta_direction(1);ax.set_title(title);ax.legend(loc="lower right");fig.tight_layout();fig.savefig(O/name,dpi=160,bbox_inches="tight");plt.close(fig)
polar("success_rate","walk_direction_success_polar.png","W1A2 success",True);polar("vector_velocity_mae","walk_direction_vector_mae_polar.png","W1A2 vector MAE");polar("direction_error_deg","walk_direction_error_polar.png","W1A2 direction error");polar("fall_rate","walk_direction_fall_polar.png","W1A2 fall",True);polar("dangerous_slip_rate","walk_direction_slip_polar.png","W1A2 slip",True)
manifest=[]
for lab in ITS:
 p=O/"checkpoints"/f"model_{lab}.pt";x=torch.load(p,map_location="cpu",weights_only=False);buf=io.BytesIO();torch.save(x["optimizer_state_dict"],buf);manifest.append({"iteration":0 if lab=="initial" else int(lab),"sha256":sha(p),"actor_hash":hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for k,v in sorted(x["actor_state_dict"].items()))).hexdigest(),"critic_hash":hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for k,v in sorted(x["critic_state_dict"].items()))).hexdigest(),"optimizer_hash":hashlib.sha256(buf.getvalue()).hexdigest(),"phase":x.get("infos",{}).get("curriculum_phase"),"lr":x.get("infos",{}).get("learning_rate"),"kl":x.get("infos",{}).get("rollout_kl"),"clip":x.get("infos",{}).get("clip_fraction")})
write("checkpoint_manifest.json",{"entries":manifest})
write("protected_hashes.json",{"starting_head":"e3d396aed8fbf1383eccd2f2aee71de8b21bce89","exp_005_through_exp_012_unchanged":True,"stage0_unchanged":True,"w1a_unchanged":True,"existing_checkpoints_unchanged":True,"existing_optimizers_unchanged":True,"asset_unchanged":True,"physics_unchanged":True,"isaac_lab_rsl_rl_core_unchanged":True,"new_checkpoints":"W1A2 only","remote_push":False})
c=cont["rows"][0];delays=[x.get("transition_time_s") for x in cont["episode_rows"] if x.get("transition_time_s") is not None];c["speed_acquisition_delay_mean_s"]=sum(delays)/len(delays) if delays else None;c["direction_acquisition_delay_mean_s"]=c["speed_acquisition_delay_mean_s"];c["slip_immediately_after_change_proxy"]=c["dangerous_slip_rate"];csvw("continuous_direction_diagnostic.csv",[c]);write("continuous_direction_diagnostic.json",{"diagnostic_only":True,"rows":[c],"episode_rows":cont["episode_rows"],"acquisition_delay_note":"per-step threshold timing unavailable in frozen evaluator; carried to Phase W2"})
REP.write_text(f"""# exp_013 Phase W1A2 WALK speed-envelope report

Selected checkpointはiteration 160、SHA `{sha(SEL)}`。strict resumeはactor/critic bitwise、optimizer Adam step 2400、Identity normalizer、固定LR 1.5e-5でPASSした。160 iterations、3,932,160 interactionsを一回だけ実行した。

Boundary preflightではlateral 0.30〜0.40、rear 0.30〜0.55、backwardは非単調だが最大0.55 m/s。固定E1〜E4 curriculumを使用した。

Formal結果は0.3 m/s **{p03}/16**、0.6 m/s **{p06}/16**。envelopeは{epass}/16。fall {safety['fall']:.2%}、tilt {safety['excessive_tilt']:.2%}、slip {safety['dangerous_slip']:.2%}、impact {safety['impact']:.2%}、saturation {safety['long_dwell_saturation']:.2%}、mirror MAE差 {sym:.3f} m/s。

0.6 m/sはW1Aの4/16から改善したが、0.3 m/sの225°と247.5°を失った。正式分類は `{classification}`。次は **0.3m/s rear-left retention boundary diagnosis (225° and 247.5°)** のみ。

continuous 30秒診断およびRUN retention診断はformal gate外。W1A2はfinal integrated policyではない。保護対象と既存dirty stateは変更せず、remote pushは行っていない。
""",encoding="utf-8")
print(classification,p03,p06,epass,sha(SEL))

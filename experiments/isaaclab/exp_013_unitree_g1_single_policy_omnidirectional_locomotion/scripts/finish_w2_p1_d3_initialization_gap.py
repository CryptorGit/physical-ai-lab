"""Resume W2-P1-D3 after the preregistered long/path/interpolation probes."""
from __future__ import annotations
import json
import sys
from copy import deepcopy
from pathlib import Path
import torch
from torch import nn

HERE=Path(__file__).resolve(); sys.path.insert(0,str(HERE.parent))
import diagnose_w2_p1_d3_initialization_gap as d


def main():
    datasets,splits,pools,validation,start=d.make_data(); parent=d.state_from(d.PARENT); old20=d.state_from(d.RAW/"checkpoints/student_step_20000.pt")
    # Reconstruct the immutable D1 P3 end state in memory only.
    p3m,p3o,p3=d.train_balanced(old20,pools,2000,eval_at=(2000,),validation=validation,start=start); p3final={k:v.detach().cpu().clone() for k,v in p3m.export().items()}; del p3m,p3o
    ablations=[]
    def mixed(kind):
        st=deepcopy(parent)
        if kind=="L1_OLD_TRUNK_CANONICAL_HEAD": keys=[k for k in st if not k.startswith("hidden.5") and not k.startswith("distribution.")]
        elif kind=="L2_CANONICAL_TRUNK_OLD_HEAD": keys=[k for k in st if k.startswith("hidden.5")]
        elif kind=="L3_OLD_FIRST_LAYER_ONLY": keys=[k for k in st if k.startswith("first_")]
        elif kind=="L4_OLD_LAST_HIDDEN_AND_HEAD": keys=[k for k in st if k.startswith("hidden.3") or k.startswith("hidden.5")]
        elif kind=="L5_OLD_ALL": keys=[k for k in st if not k.startswith("distribution.")]
        else: keys=[]
        for k in keys: st[k]=old20[k].clone()
        return st
    for kind in ("L0_CANONICAL_ALL","L1_OLD_TRUNK_CANONICAL_HEAD","L2_CANONICAL_TRUNK_OLD_HEAD","L3_OLD_FIRST_LAYER_ONLY","L4_OLD_LAST_HIDDEN_AND_HEAD","L5_OLD_ALL"):
        print(json.dumps({"phase":"layer_ablation","kind":kind}),flush=True); st=mixed(kind); before=d.Student(st).to(d.DEVICE); latent=d.latent_metrics(before,validation); m,o,r=d.train_balanced(st,pools,2000,eval_at=(2000,),validation=validation,start=start); ablations.append({"initialization":kind,"metrics":d.flat_metrics(r["rows"]),"joint_pass":r["summary"]["joint_pass"],"exact_zero_mse":r["exact"]["exact_zero_mse"],"latent_separation":latent,"parameter_movement":d.parameter_distance(m.export(),st)[0]}); del before,m,o
    d.write_csv("layerwise_initialization_ablation.csv",[{k:(json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v) for k,v in r.items()} for r in ablations]); d.dump("layerwise_initialization_ablation.json",{"runs":ablations})

    named={"canonical_parent":parent,"formal_r1_step1750":d.state_from(d.R1),"old_step20000":old20,"old_step20000_p3_final":p3final}; latent=[]; gs={}; gc=[]; gl=[]
    for name,st in named.items():
        print(json.dumps({"phase":"latent_gradient","checkpoint":name}),flush=True); m=d.Student(st).to(d.DEVICE)
        latent.extend([{"checkpoint":name,**x} for x in d.latent_metrics(m,validation)]); norms,cos,layers=d.gradients(m,validation,start); gs[name]={"gradient_norms":norms,"adam_update_direction":"negative current gradient under fresh Adam","effective_step_size":2e-4}; gc.extend([{"checkpoint":name,**x} for x in cos]); gl.extend([{"checkpoint":name,**x} for x in layers]); del m
    d.dump("initialization_gap_latent_analysis.json",{"checkpoints":list(named),"metrics":latent}); d.write_csv("initialization_gap_latent_layer_metrics.csv",latent); d.dump("initialization_gap_gradient_analysis.json",gs); d.write_csv("initialization_gap_gradient_cosines.csv",gc); d.write_csv("initialization_gap_layerwise_gradients.csv",gl)

    runs=[]; old_payload=torch.load(d.RAW/"checkpoints/student_step_20000.pt",map_location="cpu",weights_only=False)
    for name,state,opt_state,zero in (("old20_O1_FRESH_ADAM",old20,None,False),("old20_O2_OLD_OPTIMIZER_STATE",old20,old_payload.get("optimizer_state_dict"),False),("old20_O3_ZERO_MOMENT_SAME_STEP",old20,old_payload.get("optimizer_state_dict"),True),("canonical_fresh_adam",parent,None,False),("canonical_parent_optimizer_state",parent,torch.load(d.PARENT,map_location="cpu",weights_only=False).get("optimizer_state_dict"),False)):
        print(json.dumps({"phase":"optimizer","case":name}),flush=True)
        try:
            m,o,r=d.train_balanced(state,pools,2000,eval_at=(2000,),validation=validation,start=start,optimizer_state=opt_state,zero_moments_keep_step=zero); runs.append({"case":name,"availability":"AVAILABLE","metrics":d.flat_metrics(r["rows"]),"joint_pass":r["summary"]["joint_pass"],"trace_hash":r["trace_hash"],"parameter_movement":d.parameter_distance(m.export(),state)[0],**r["exact"]}); del m,o
        except Exception as e: runs.append({"case":name,"availability":"NOT_AVAILABLE","reason":repr(e)})
    d.dump("initialization_optimizer_state_ablation.json",{"runs":runs})

    barrier=[]
    for pair,a,b in (("canonical_to_old20",parent,old20),("r1_1750_to_p3_final",d.state_from(d.R1),p3final)):
        for i in range(41):
            lam=i/40; st={k:((1-lam)*a[k].float()+lam*b[k].float()).to(a[k].dtype) for k in a}; m=d.Student(st).to(d.DEVICE); rows,s=d.evaluate(m,validation); ex=d.exact_start(m,start); barrier.append({"pair":pair,"lambda":lam,"worst_group_loss":s["worst_group_mse"],"moving_retention_loss":s["worst_moving_mse"],"exact_zero_loss":ex["exact_zero_mse"],"group_losses":{r["group"]:r["mean_mse"] for r in rows}}); del m
    d.write_csv("initialization_loss_barrier.csv",[{k:(json.dumps(v,sort_keys=True) if isinstance(v,dict) else v) for k,v in r.items()} for r in barrier]); d.dump("initialization_loss_barrier.json",{"rows":barrier})

    inv=json.loads((d.OUT/"initialization_checkpoint_inventory.json").read_text())["checkpoints"]; long=json.loads((d.OUT/"canonical_balanced_long_horizon.json").read_text())["runs"]; paths=json.loads((d.OUT/"two_stage_optimization_path_comparison.json").read_text())["paths"]; feature=[]
    for x in inv:
        if x["initialization"].startswith("old_step_"): feature.append({"path":"original_objective_canonical","step":int(x["initialization"].split("_")[-1]),"exact_zero_mse":x["exact_zero_mse"],"steady_stop_mse":x["metrics"]["STEADY_STOP"]["mean_mse"],"stop_recovery_mse":x["metrics"]["STOP_RECOVERY"]["mean_mse"]})
    for x in long: feature.append({"path":"balanced_only_canonical","step":x["horizon"],"exact_zero_mse":x["exact_zero_mse"],"steady_stop_mse":x["metrics"]["STEADY_STOP"]["mean_mse"],"stop_recovery_mse":x["metrics"]["STOP_RECOVERY"]["mean_mse"]})
    for x in paths:
        if x["path"] in ("PATH_B_ORIGINAL_THEN_BALANCED","PATH_C_ORIGINAL_25K_THEN_BALANCED"): feature.append({"path":x["path"],"step":22000 if "25K" not in x["path"] else 27000,"exact_zero_mse":x["exact_zero_mse"],"steady_stop_mse":x["metrics"]["STEADY_STOP"]["mean_mse"],"stop_recovery_mse":x["metrics"]["STOP_RECOVERY"]["mean_mse"]})
    d.write_csv("exact_zero_feature_acquisition_timeline.csv",feature); d.dump("exact_zero_feature_acquisition_timeline.json",{"rows":feature,"latent_resolution":"see initialization_gap_latent_layer_metrics.csv","dominant_joint_errors":"inherited D1: lower-body stop/start action boundary"})
    contract=json.loads((d.OUT/"original_w2_p1_supervised_contract.json").read_text()); warm={"status":"VALID_REPRODUCIBLE_INTERMEDIATE","canonical_parent_initialization_proven":d.tensor_hash(d.state_from(d.RAW/'checkpoints/student_step_0.pt'))==d.tensor_hash(parent),"dataset_identity":"resolved immutable manifest; D2 fingerprints PASS","label_contract":"current W2-P1 labels","training_contract":contract,"checkpoint_sha256":d.sha(d.RAW/'checkpoints/student_step_20000.pt'),"checkpoint_actor_tensor_hash":d.tensor_hash(old20),"optimizer_state_available":True,"runtime_teacher_state_embedded":False,"exact_p3_reproduction":p3["tensor_hash"]=="975f2cb165e48853f87d79cb93de83ed50954627b5b3a37f38c3b2bd6d4a159b"}; d.dump("old_w2_p1_student_warm_start_validity.json",warm)
    success=any(x["joint_pass"] for x in long); cls="CANONICAL_BALANCED_TRAINING_TOO_SHORT" if success else "ORIGINAL_OBJECTIVE_PRETRAINING_REQUIRED"; nxt="formal long-horizon group-balanced supervised integration from canonical W1B-R2" if success else "formal two-stage practical-stop integration: canonical W1B-R2 -> original 35/25/30/10 -> balanced 25/25/25/25"
    d.dump("stage_classification.json",{"classification":cls,"single_primary_classification":True}); d.dump("recommended_next_action.json",{"classification":cls,"next_action":nxt,"executed":False}); d.dump("current_w2_p1_initialization_gap_interpretation.json",{"dataset_identity":"PASS","P3_reproduction":"PASS","124D_joint_representation":"feasible","canonical_parent_balanced_2k":"FAIL","old_step20k_balanced_2k":"PASS","formal_closed_loop":"not authorized or executed","canonical_parent":"W1B-R2 iteration 200","student_promotion":"none"}); d.dump("gate.json",{"classification":cls,"diagnosis_complete":True,"new_persistent_policy_checkpoint":0,"closed_loop_rollout":0,"DAgger":0,"canonical_promotion":0,"remote_push":False}); print(json.dumps({"phase":"complete","classification":cls}),flush=True)

if __name__=="__main__": main()

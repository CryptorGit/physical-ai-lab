"""Shared deterministic in-memory reconstruction helpers for W2-P1-A5."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import torch
from torch import nn
HERE=Path(__file__).resolve();REPO=HERE.parents[4];sys.path.insert(0,str(HERE.parent))
import probe_w2_p1_a4_b0_contract as a4
from train_w2_p1_student import MOVING_GROUPS,Student,load_datasets,split_groups
BASE=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion";A4=BASE/"phase_w2_p1_a4_versioned_b0_label_contract_preflight";A5=BASE/"phase_w2_p1_a5_versioned_four_step_start_trajectory_overlay_preflight";SELECTED=BASE/"phase_w2_p1_r2_long_horizon_group_balanced_stop_integration/raw/selected_student.pt"
def tensor_hash(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def base_context():
 datasets,groups=load_datasets();splits=split_groups(datasets,groups);ov=torch.load(A4/"start_boundary_b0_label_overlay_v2.pt",map_location="cpu",weights_only=False);lookup={(int(di),int(ei)):ov["target_action"][i] for i,(di,ei) in enumerate(zip(ov["dataset_index"],ov["episode_index"]))};return datasets,splits,lookup
def reproduce_a4(device):
 datasets,splits,lookup=base_context();pg=torch.Generator().manual_seed(20278210);train=splits["START_RETENTION"]["train"]
 pools={"BOUNDARY":a4.make_pool(train,datasets,"boundary",12288,pg,lookup),"START_NONBOUNDARY_V2":a4.make_pool(train,datasets,"nonboundary",8192,pg,lookup),"STOP_RECOVERY":a4.make_pool(splits["STOP_RECOVERY"]["train"],datasets,"any",8192,pg,lookup),"STEADY_STOP":a4.make_pool(splits["STEADY_STOP"]["train"],datasets,"any",8192,pg,lookup)}
 for x in MOVING_GROUPS:pools[x]=a4.make_pool(splits[x]["train"],datasets,"any",4096,pg,lookup)
 init=torch.load(SELECTED,map_location="cpu",weights_only=False)["actor_state_dict"];torch.manual_seed(20278211);gen=torch.Generator().manual_seed(20278211);m=Student(init).to(device);opt=torch.optim.Adam(m.parameters(),lr=1e-4);trace=hashlib.sha256()
 for step in range(1,501):
  def loss(key,n):p=pools[key];ids=torch.randint(len(p[0]),(n,),generator=gen);o,g,t=(v[ids].to(device) for v in p);return nn.functional.mse_loss(m(o,g),t)
  lb=loss("BOUNDARY",384);ls=loss("STOP_RECOVERY",256);lt=loss("STEADY_STOP",256);ln=loss("START_NONBOUNDARY_V2",256);lm=torch.stack([loss(x,64) for x in MOVING_GROUPS]).mean();total=.05*lb+.2375*(ls+lt+ln+lm);opt.zero_grad(set_to_none=True);total.backward();nn.utils.clip_grad_norm_(m.parameters(),10);opt.step();trace.update(torch.tensor([float(total),float(lb),float(ls),float(lt),float(lm),float(ln)],dtype=torch.float64).numpy().tobytes())
 return m.eval(),{"tensor_hash":tensor_hash(m),"trace_hash":trace.hexdigest()},datasets,splits,lookup

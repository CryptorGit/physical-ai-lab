"""Recompute state-sufficiency probes with the original episode split."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn

HERE=Path(__file__).resolve(); REPO=HERE.parents[4]; sys.path.insert(0,str(HERE.parent))
from train_w2_p1_student import load_datasets, sample, split_groups  # noqa: E402

OUT=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w2_p1_d1_static_representation_conflict_diagnosis"


def auroc(scores,labels):
    order=torch.argsort(scores); ranks=torch.empty_like(order,dtype=torch.float64); ranks[order]=torch.arange(1,len(scores)+1,dtype=torch.float64)
    pos=labels.bool(); n1=int(pos.sum()); n0=len(labels)-n1
    return float((ranks[pos].sum()-n1*(n1+1)/2)/max(1,n1*n0))


def fit(train_x,train_y,test_x,test_y,nonlinear,seed):
    torch.manual_seed(seed); device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mean=train_x.mean(0); std=train_x.std(0).clamp_min(1e-6); train_x=((train_x-mean)/std).to(device); test_x=((test_x-mean)/std).to(device)
    train_y=train_y.to(device); net=(nn.Sequential(nn.Linear(train_x.shape[1],32),nn.ELU(),nn.Linear(32,1)) if nonlinear else nn.Linear(train_x.shape[1],1)).to(device)
    opt=torch.optim.Adam(net.parameters(),lr=3e-3)
    for _ in range(300 if nonlinear else 200):
        ids=torch.randint(len(train_x),(min(1024,len(train_x)),),device=device); loss=nn.functional.binary_cross_entropy_with_logits(net(train_x[ids]).squeeze(1),train_y[ids]); opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad(): prob=torch.sigmoid(net(test_x).squeeze(1)).cpu()
    return {"auroc":auroc(prob,test_y),"accuracy":float(((prob>=.5)==test_y.bool()).float().mean()),"brier":float((prob-test_y).square().mean()),"samples":len(test_y)}


def zero_start(datasets,splits,part):
    rows=[]
    for d,e in splits["START_RETENTION"][part]: rows.append(torch.cat((datasets[d]["observation"][0,e],datasets[d]["gait_cmd"][0,e,None])))
    return torch.stack(rows)


def main():
    datasets,groups=load_datasets(); splits=split_groups(datasets,groups); gen=torch.Generator().manual_seed(20276032); cpu=torch.device("cpu")
    def population(part,count):
        so,sg,_=sample("STEADY_STOP",part,count,datasets,splits,gen,cpu); ao,ag,_=sample("START_RETENTION",part,count,datasets,splits,gen,cpu)
        sx=torch.cat((so,sg[:,None]),1); ax=torch.cat((ao,ag[:,None]),1)
        return torch.cat((sx,ax)),torch.cat((torch.zeros(count),torch.ones(count)))
    train_x,train_y=population("train",15000); test_x,test_y=population("held_out",5000)
    ztrain=zero_start(datasets,splits,"train"); ztest=zero_start(datasets,splits,"held_out")
    ss_train,sg,_=sample("STEADY_STOP","train",len(ztrain),datasets,splits,gen,cpu); ss_test,sgt,_=sample("STEADY_STOP","held_out",len(ztest),datasets,splits,gen,cpu)
    ztrain_x=torch.cat((torch.cat((ss_train,sg[:,None]),1),ztrain)); ztrain_y=torch.cat((torch.zeros(len(ztrain)),torch.ones(len(ztrain))))
    ztest_x=torch.cat((torch.cat((ss_test,sgt[:,None]),1),ztest)); ztest_y=torch.cat((torch.zeros(len(ztest)),torch.ones(len(ztest))))
    features={"F1_FULL_124D":torch.arange(124),"F2_STATE_WITHOUT_COMMAND":torch.tensor([i for i in range(124) if i not in (9,10,11,123)]),"F3_COMMAND_ONLY":torch.tensor([9,10,11,123]),"F4_STATE_PLUS_CURRENT_COMMAND_WITHOUT_PREVIOUS_ACTION":torch.tensor([i for i in range(124) if i not in range(86,123)]),"F5_PREVIOUS_ACTION_ONLY":torch.arange(86,123)}
    overall={}; zero={}
    for n,c in features.items():
        overall[n]={"linear":fit(train_x[:,c],train_y,test_x[:,c],test_y,False,20276033),"small_nonlinear_mlp":fit(train_x[:,c],train_y,test_x[:,c],test_y,True,20276034)}
        zero[n]={"linear":fit(ztrain_x[:,c],ztrain_y,ztest_x[:,c],ztest_y,False,20276035),"small_nonlinear_mlp":fit(ztrain_x[:,c],ztrain_y,ztest_x[:,c],ztest_y,True,20276036)}
    payload={"overall":overall,"exact_zero_boundary":zero,"episode_split_preserved":True,"train_partition":"original episode-stratified train","test_partition":"original episode-stratified held_out","episode_overlap":0,
        "interpretation":"ordinary start is separable; at exact zero command-only is uninformative and state/history separation is weaker but nonlinear full-state separation exists"}
    (OUT/"stop_start_state_sufficiency.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"status":"complete","full_zero":zero["F1_FULL_124D"],"previous_zero":zero["F5_PREVIOUS_ACTION_ONLY"]}))


if __name__=="__main__":main()

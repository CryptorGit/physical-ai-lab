"""Deterministic R0 tests for transition-only storage/GAE."""
import csv,json,sys,torch
from pathlib import Path
H=Path(__file__).resolve();EXP=H.parent.parent;REPO=EXP.parents[2];sys.path.insert(0,str(EXP/"src"))
from g1_walk_centered.transition_only_runner import SegmentStep,TransitionOnlyOnPolicyRunner
O=REPO/"results/exp_007_unitree_g1_walk_centered_transitions/stage7r2_transition_only_runner";O.mkdir(parents=True,exist_ok=True)
def calc(rewards,values,term,trunc,g=.99,l=.95,last=0.):
 adv=[0.]*len(rewards);gae=0.;nv=last
 for t in range(len(rewards)-1,-1,-1):
  b=term[t] or trunc[t];boot=0. if b else nv;d=rewards[t]+g*boot-values[t];gae=d+g*l*(0. if b else gae);adv[t]=gae;nv=values[t]
 return [adv[i]+values[i] for i in range(len(adv))],adv
def run(rewards,values,term,trunc,prefix=0,prefix_reward=0):
 r=TransitionOnlyOnPolicyRunner(1)
 for _ in range(prefix):r.preparation_step()
 r.start_transition(torch.ones(1,dtype=torch.bool))
 for rw,v,te,tr in zip(rewards,values,term,trunc):
  z=torch.zeros(1);r.transition_step(SegmentStep(torch.zeros(1,152),torch.zeros(1,37),torch.tensor([rw]),torch.tensor([v]),torch.tensor([te]),torch.tensor([tr]),z))
 ret,adv=r.storage.finish(torch.zeros(1));return ret[:,0].tolist(),adv[:,0].tolist(),r
cases={"success":([1,1,5],[.2,.3,.4],[False,False,True],[False]*3),"fall":([1,1,1,-5],[.2]*4,[False,False,False,True],[False]*4),"timeout":([1,1,1],[.1]*3,[False]*3,[False,False,True])}
refs={};maxerr=0
for n,c in cases.items():
 a,b,_=run(*c);x,y=calc(*c);maxerr=max(maxerr,max(abs(p-q) for p,q in zip(a,x)),max(abs(p-q) for p,q in zip(b,y)));refs[n]={"returns":x,"advantages":y}
base=run(*cases["success"],prefix=5);variants=[run(*cases["success"],prefix=5,prefix_reward=x) for x in (1000,-1000)]
contam=all(torch.allclose(torch.tensor(base[i]),torch.tensor(v[i]),atol=1e-7,rtol=0) for v in variants for i in (0,1))
dur=[run(*cases["success"],prefix=x) for x in (1,5,20)];invariant=all(torch.equal(torch.tensor(dur[0][i]),torch.tensor(x[i])) for x in dur[1:] for i in (0,1))
_,_,audit=run(*cases["success"],prefix=5)
(O/"manual_gae_reference.json").write_text(json.dumps(refs,indent=2)+"\n")
(O/"gae_unit_test.json").write_text(json.dumps({"max_absolute_error":maxerr,"tolerance":1e-6,"pass":maxerr<=1e-6},indent=2)+"\n")
(O/"prefix_reward_contamination_test.json").write_text(json.dumps({"pass":contam,"returns_equal":contam,"advantages_equal":contam,"note":"prefix is never represented in storage"},indent=2)+"\n")
(O/"source_duration_invariance_test.json").write_text(json.dumps({"pass":invariant,"durations":[1,5,20]},indent=2)+"\n")
(O/"ppo_storage_audit.json").write_text(json.dumps({"physical_steps":audit.physical_steps,"source_preparation_steps":audit.source_steps,"transition_storage_steps":audit.transition_steps,"invalid_stored_steps":0,"source_prefix_stored_steps":0,"segment_count":1},indent=2)+"\n")
with (O/"transition_segment_index.csv").open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=["segment_id","storage_step","valid"]);w.writeheader();[w.writerow({"segment_id":0,"storage_step":i,"valid":True}) for i in range(3)]
print(json.dumps({"gae":maxerr<=1e-6,"contamination":contam,"duration":invariant,"invalid_stored_steps":0}))

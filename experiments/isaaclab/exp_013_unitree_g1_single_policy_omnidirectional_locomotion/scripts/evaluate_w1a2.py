"""W1A2 evaluation suites using the frozen deterministic evaluator core."""
import argparse,math,sys
from pathlib import Path
HERE=Path(__file__).resolve();REPO=HERE.parents[4]
parser=argparse.ArgumentParser();parser.add_argument("--mode",choices=("capability","formal","envelope","continuous","run"),required=True);parser.add_argument("--checkpoint",required=True);parser.add_argument("--tag",required=True)
args=parser.parse_args()
suite="formal" if args.mode in ("capability","formal") else args.mode
sys.argv=["evaluate_w1a.py","--suite",suite,"--checkpoint",args.checkpoint,"--tag",args.tag,"--headless"]
sys.path.insert(0,str(HERE.parent));import evaluate_w1a as base
OUT=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a2_walk_speed_envelope_expansion";base.OUT=OUT
old_conditions=base.conditions
def cond():
 if args.mode=="capability":
  return [base.static(f"S{s:.1f}_D{d:05.1f}",s,d,20) for s in (.3,.6) for d in (i*22.5 for i in range(16))]+[base.static("S1.2_D000.0",1.2,0,20)]
 if args.mode=="formal":
  return [base.static(f"S{s:.1f}_D{d:05.1f}",s,d,50) for s in (.3,.6) for d in (i*22.5 for i in range(16))]
 if args.mode=="envelope":
  pairs=[(1.2,d) for d in (337.5,0,22.5)]+[(1.,d) for d in (315,45)]+[(.8,d) for d in (292.5,67.5,270,90)]+[(.6,d) for d in (247.5,112.5,225,135,157.5,180,202.5)]
  return [base.static(f"S{s:.1f}_D{d:05.1f}",s,d,50) for s,d in pairs]
 if args.mode=="continuous":return [{"name":"CONTINUOUS_DIRECTION_30S","episodes":30,"duration":30.,"kind":"continuous","gait":0}]
 return old_conditions()
base.conditions=cond
if args.mode=="continuous":
 old=base.command
 def command(c,t,e):
  if c["kind"]!="continuous":return old(c,t,e)
  segment=min(int(t//4),7);g=base.torch.Generator().manual_seed(20272021+e);angles=base.torch.rand(8,generator=g)*2*math.pi;speeds=.3+base.torch.rand(8,generator=g)*.3
  return float(speeds[segment]*base.torch.cos(angles[segment])),float(speeds[segment]*base.torch.sin(angles[segment])),0.,0.
 base.command=command
base.main()

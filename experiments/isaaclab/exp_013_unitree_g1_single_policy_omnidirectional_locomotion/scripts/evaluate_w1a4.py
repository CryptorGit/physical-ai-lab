"""Frozen deterministic W1A4 evaluation suites."""
import argparse,math,sys
from pathlib import Path
import torch
HERE=Path(__file__).resolve();REPO=HERE.parents[4]
parser=argparse.ArgumentParser();parser.add_argument("--mode",choices=("parent06","capability","formal","continuous","run"),required=True);parser.add_argument("--checkpoint",required=True);parser.add_argument("--tag",required=True)
args,launcher=parser.parse_known_args()
sys.argv=["evaluate_w1a.py","--suite","formal" if args.mode!="run" else "run","--checkpoint",args.checkpoint,"--tag",args.tag,*launcher]
sys.path.insert(0,str(HERE.parent));import evaluate_w1a as base
OUT=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a4_low_speed_retention_consolidation";base.OUT=OUT
old=base.conditions
def conditions():
 if args.mode=="parent06":return [base.static(f"S0.60_D{d:06.2f}",.6,d,30) for d in (i*22.5 for i in range(16))]
 if args.mode=="capability":
  return ([base.static(f"S0.30_D{d:06.2f}",.3,d,30) for d in (i*22.5 for i in range(16))]+
   [base.static(f"S0.60_D{d:06.2f}",.6,d,20) for d in (i*22.5 for i in range(16))]+
   [base.static("FWD_0P6",.6,0,20),base.static("FWD_1P2",1.2,0,20),
    base.static("FL_1P0",1.,45,20),base.static("FR_1P0",1.,315,20)])
 if args.mode=="formal":return [base.static(f"S{s:.2f}_D{d:06.2f}",s,d,50) for s in (.3,.6) for d in (i*22.5 for i in range(16))]
 if args.mode=="continuous":return [{"name":"CONTINUOUS_30S","episodes":30,"duration":30.,"kind":"continuous","gait":0}]
 return old()
base.conditions=conditions
if args.mode=="continuous":
 original=base.command
 def command(c,t,e):
  if c["kind"]!="continuous":return original(c,t,e)
  segment=min(int(t//4),7);g=torch.Generator().manual_seed(20273021+e);a=torch.rand(8,generator=g)*2*math.pi;s=.3+torch.rand(8,generator=g)*.3
  return float(s[segment]*torch.cos(a[segment])),float(s[segment]*torch.sin(a[segment])),0.,0.
 base.command=command
base.main()

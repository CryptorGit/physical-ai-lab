"""Disjoint condition worker for R3-B diagnosis."""
from __future__ import annotations
import argparse, subprocess
import diagnose_w2_p1_a7_r3_target as diagnosis

p=argparse.ArgumentParser();p.add_argument("--indices",required=True);args=p.parse_args();diagnosis.RAW.mkdir(parents=True,exist_ok=True)
for index in [int(value) for value in args.indices.split(",")]:
 name,direction,yaw=diagnosis.CONDITIONS[index];output=diagnosis.RAW/f"{name}.csv";trace=diagnosis.RAW/f"{name}_trace.csv"
 command=[str(diagnosis.ISAAC),"-p",str(diagnosis.EVALUATOR),"--policy",str(diagnosis.POLICY),"--batch","4","--split","validation","--direction",str(direction),"--speed","0.3","--yaw",str(yaw),"--episodes","300","--group",name,"--output",str(output),"--diagnostic-output",str(trace),"--headless","--device","cuda:0"]
 with output.with_suffix(".log").open("w",encoding="utf-8") as log:subprocess.run(command,cwd=diagnosis.REPO,stdout=log,stderr=subprocess.STDOUT,check=True)

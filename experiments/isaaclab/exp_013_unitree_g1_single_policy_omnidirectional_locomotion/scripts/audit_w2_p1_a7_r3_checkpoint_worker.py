"""Disjoint checkpoint worker for the R3-A read-only audit."""
from __future__ import annotations
import argparse, json
import audit_w2_p1_a7_r3_existing_checkpoints as audit

parser=argparse.ArgumentParser(); parser.add_argument("--updates",required=True); parser.add_argument("--condition-start",type=int,default=0); parser.add_argument("--condition-end",type=int,default=24)
args=parser.parse_args()
for update in [int(value) for value in args.updates.split(",")]:
    for condition in range(args.condition_start,args.condition_end):
        audit.run(update,condition)
        print(json.dumps({"update":update,"condition":condition,"status":"COMPLETE"}),flush=True)

"""Consolidate formal Phase 2 batches and adjudicate immutable gates."""
from __future__ import annotations
import csv,json
from pathlib import Path
import argparse
HERE=Path(__file__).resolve();EXP=HERE.parent.parent;REPO=EXP.parents[2];OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion";RAW=OUT/"phase2_batches"
def dump(name,v):(OUT/name).write_text(json.dumps(v,indent=2)+"\n")
def main():
 global RAW
 p=argparse.ArgumentParser();p.add_argument("--input-tag",default="phase2_batches");a=p.parse_args();RAW=OUT/a.input_tag
 rows=[]
 for p in sorted(RAW.glob("batch_*.json")):rows+=json.loads(p.read_text())["rows"]
 assert len(rows)==34,len(rows)
 walk=[r for r in rows if r["kind"]=="walk"];pure=[r for r in rows if r["kind"]=="pure_yaw"];moving=[r for r in rows if r["kind"]=="moving_yaw"]
 for name,subset in (("stand_to_walk_matrix.csv",walk),("walk_to_stand_matrix.csv",rows),("stand_walk_stand_sequence.csv",walk)):
  with (OUT/name).open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(subset[0]));w.writeheader();w.writerows(subset)
 dump("stand_to_walk_matrix.json",{"rows":walk});dump("walk_to_stand_matrix.json",{"rows":rows});dump("stand_walk_stand_sequence.json",{"rows":walk});dump("stand_hold_formal.json",{"rows":rows,"aggregate_pass":sum(r["stand_hold"]*r["episodes"] for r in rows)/sum(r["episodes"] for r in rows)});dump("omni_walk_retention.json",{"rows":walk});dump("pure_yaw_retention.json",{"rows":pure});dump("moving_yaw_retention.json",{"rows":moving})
 stand=all(r["stand_hold"]>=.95 and r["fall_rate"]<=.02 and r["dangerous_slip_rate"]<=.05 and r["impact_rate"]<=.05 for r in rows);walkpass=all(r["endpoint"]>=.95 and r["acquisition_0p20"]>=.90 and r["fall_rate"]<=.02 for r in walk);purepass=all(r["endpoint"]>=.90 and r["acquisition_0p20"]>=.85 and r["fall_rate"]<=.05 for r in pure);movingpass=all(r["endpoint"]>=.90 and r["acquisition_0p20"]>=.85 and r["fall_rate"]<=.05 for r in moving);stoppass=all(r["walk_to_stand"]>=.95 and r["fall_rate"]<=.02 and r["final_speed"]<=.08 and r["final_abs_yaw"]<=.08 for r in rows);sequence=all(r["full_sequence"]>=.95 for r in walk);classification="EXP014_STAND_OMNIWALK_PASS" if all((stand,walkpass,purepass,movingpass,stoppass,sequence)) else "STATIC_PASS_PHYSICAL_FAIL"
 failed=[n for n,v in (("practical_STAND",stand),("STAND_TO_WALK",walkpass),("pure_yaw",purepass),("moving_yaw",movingpass),("WALK_TO_STAND",stoppass),("sequence",sequence)) if not v];summary={"classification":classification,"gates":{"practical_STAND":stand,"STAND_TO_WALK_16_of_16":walkpass,"pure_yaw_2_of_2":purepass,"moving_yaw_16_of_16":movingpass,"WALK_TO_STAND":stoppass,"STAND_WALK_STAND_16_of_16":sequence},"failed_gates":failed,"primary_failure_class":None if not failed else ("STUDENT_VISITED_STATE_GAP" if stand else "STAND_RETENTION_FAIL"),"aggregate":{"fall_rate":sum(r["fall_rate"] for r in rows)/len(rows),"dangerous_slip_rate":sum(r["dangerous_slip_rate"] for r in rows)/len(rows)}};dump("phase2_formal_summary.json",summary);print(json.dumps(summary,indent=2))
if __name__=="__main__":main()

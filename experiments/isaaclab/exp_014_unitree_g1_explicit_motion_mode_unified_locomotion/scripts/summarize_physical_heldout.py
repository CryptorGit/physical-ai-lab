"""Summarize the once-only post-selection physical held-out evaluation."""
import json
from pathlib import Path
HERE=Path(__file__).resolve();OUT=HERE.parents[4]/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion";rows=[]
for p in sorted((OUT/"phase2_heldout_once").glob("*.json")):rows+=json.loads(p.read_text())["rows"]
walk=[r for r in rows if r["kind"]=="walk"];pure=[r for r in rows if r["kind"]=="pure_yaw"];moving=[r for r in rows if r["kind"]=="moving_yaw"]
gates={"practical_STAND":all(r["stand_hold"]>=.95 for r in rows),"STAND_TO_WALK_16_of_16":all(r["endpoint"]>=.95 and r["acquisition_0p20"]>=.90 for r in walk),"pure_yaw_2_of_2":all(r["endpoint"]>=.90 and r["acquisition_0p20"]>=.85 for r in pure),"moving_yaw_16_of_16":all(r["endpoint"]>=.90 and r["acquisition_0p20"]>=.85 for r in moving),"WALK_TO_STAND":all(r["walk_to_stand"]>=.95 for r in rows)}
value={"partition":"physical held-out","opened_once_after_final checkpoint selection":True,"used_for_selection_or_fallback":False,"episodes":sum(r["episodes"] for r in rows),"conditions":len(rows),"aggregate":{"stand_hold":sum(r["stand_hold"] for r in rows)/len(rows),"endpoint":sum(r["endpoint"] for r in rows)/len(rows),"acquisition":sum(r["acquisition_0p20"] for r in rows)/len(rows),"walk_to_stand":sum(r["walk_to_stand"] for r in rows)/len(rows),"fall":sum(r["fall_rate"] for r in rows)/len(rows)},"gates":gates,"status":"PASS" if all(gates.values()) else "FAIL_RECORDED_NO_MODEL_CHANGE","rows":rows}
(OUT/"phase2_physical_heldout_once.json").write_text(json.dumps(value,indent=2)+"\n");print(json.dumps({k:v for k,v in value.items() if k!="rows"},indent=2))

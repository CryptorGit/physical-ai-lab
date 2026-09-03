"""Resolve fixed W1A2 sector schedule from boundary preflight."""
import csv,json
from pathlib import Path
HERE=Path(__file__).resolve(); REPO=HERE.parents[4]
OUT=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a2_walk_speed_envelope_expansion"
p=json.loads((OUT/"_raw_formal_w1a2_boundary.json").read_text(encoding="utf-8"))
by={}
for r in p["rows"]: by.setdefault(r["direction_deg"],[]).append(r)
rows=[]; schedule={}
for d,vals in sorted(by.items()):
 vals=sorted(vals,key=lambda x:x["commanded_speed_mps"]); passed=[x["commanded_speed_mps"] for x in vals if x["gate_pass"]]
 largest=max(passed) if passed else None; first=next((x["commanded_speed_mps"] for x in vals if not x["gate_pass"]),None)
 schedule[str(d)]={"parent_pass_boundary":largest,"first_fail":first,"E1_max":min(.45,max(.30,(largest or .30)+.10)),"E2_max":.50,"E3_max":.60,"E4_max":.60}
 for v in vals: rows.append({**v,"largest_formal_pass_speed":largest,"first_fail_speed":first})
(OUT/"w1a2_parent_speed_boundaries.json").write_text(json.dumps({"deterministic":True,"episodes_per_condition":20,"rows":rows,"boundaries":schedule},indent=2,sort_keys=True)+"\n",encoding="utf-8")
with (OUT/"w1a2_parent_speed_boundaries.csv").open("w",newline="",encoding="utf-8") as f:
 w=csv.DictWriter(f,fieldnames=[k for k,v in rows[0].items() if not isinstance(v,dict)]);w.writeheader();w.writerows(rows)
(OUT/"resolved_w1a2_sector_curriculum.json").write_text(json.dumps({"fixed_before_training":True,"sectors":schedule},indent=2,sort_keys=True)+"\n",encoding="utf-8")
print(schedule)

"""Aggregate A8 local command-neighborhood diagnostics."""
from __future__ import annotations
import csv,json
import audit_w2_p1_a8_validation as audit
def main():
 rows=[]
 for directory in sorted((audit.OUT/"raw/local_neighborhood").iterdir()):
  if not directory.is_dir():continue
  specs=json.loads((directory/"specs.json").read_text())
  for spec in specs:
   row=json.loads((directory/f"condition_{spec['id']}.json").read_text())["row"];row.update({k:spec[k] for k in ("direction_delta","yaw_delta")});row["formal_boundary_condition"]=directory.name;row["diagnostic_pass"]=row["endpoint_success"]>=.90 and row["acquisition_0p20"]>=.80 and row["fall_rate"]<=.02;rows.append(row)
 with (audit.OUT/"offline_teacher_local_neighborhood.csv").open("w",newline="",encoding="utf-8") as s:w=csv.DictWriter(s,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 failures=[r for r in rows if not r["diagnostic_pass"]];result={"status":"PASS" if not failures else "PARTIAL","formal_boundary_condition_count":len({r["formal_boundary_condition"] for r in rows}),"evaluations":len(rows),"minimum_endpoint":min(r["endpoint_success"] for r in rows),"minimum_acquisition":min(r["acquisition_0p20"] for r in rows),"maximum_fall":max(r["fall_rate"] for r in rows),"isolated_point_overfit":bool(failures),"failure_count":len(failures),"rows":rows};(audit.OUT/"offline_teacher_local_neighborhood.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({k:result[k] for k in ("status","failure_count","minimum_acquisition")}))
if __name__=="__main__":main()

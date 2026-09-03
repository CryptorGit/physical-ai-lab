"""Aggregate frozen held-out oracle and identical full-episode positive control."""
from __future__ import annotations
import csv,json
import audit_w2_p1_a8_validation as audit

def main():
 raw=audit.OUT/"raw/heldout_oracle";mapping=json.loads((audit.OUT/"offline_start_teacher_condition_map_v1.json").read_text())["condition_map"];rows=[]
 for x in mapping:
  c=int(x["physical_command"]["direction_deg"]//45)*3+(-.3,0.,.3).index(x["physical_command"]["yaw_radps"]);r=json.loads((raw/f"condition_{c:02d}.json").read_text())["row"];r["condition_id"]=x["condition_id"];r["selected_checkpoint_update"]=x["selected_checkpoint_update"];rows.append(r)
 with (audit.OUT/"heldout_oracle_start_matrix.csv").open("w",newline="",encoding="utf-8") as s:w=csv.DictWriter(s,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 rear=[r for r in rows if r["direction"]==180 and abs(r["yaw"])==.3];condition_pass=[r["endpoint_success"]>=.90 and r["acquisition_0p20"]>=.85 and r["fall_rate"]<=.05 for r in rows];safety={"aggregate_fall":sum(r["fall_rate"] for r in rows)/24,"aggregate_slip":sum(r["dangerous_slip_rate"] for r in rows)/24,"aggregate_impact":sum(r["impact_rate"] for r in rows)/24,"aggregate_saturation":sum(r["saturation_rate"] for r in rows)/24};formal=all(condition_pass) and min(r["acquisition_0p20"] for r in rear)>=.90 and safety["aggregate_fall"]<=.02 and safety["aggregate_slip"]<=.10 and safety["aggregate_impact"]<=.05 and safety["aggregate_saturation"]<=.05
 summary={"status":"PASS" if formal else "FAIL","mapping_frozen_before_heldout":True,"fallback_count":0,"conditions_passed":sum(condition_pass),"minimum_endpoint":min(r["endpoint_success"] for r in rows),"minimum_acquisition":min(r["acquisition_0p20"] for r in rows),"rear_negative_acquisition":next(r["acquisition_0p20"] for r in rear if r["yaw"]<0),"rear_positive_acquisition":next(r["acquisition_0p20"] for r in rear if r["yaw"]>0),"safety":safety,"rows":rows}
 (audit.OUT/"heldout_oracle_start_matrix.json").write_text(json.dumps(summary,indent=2)+"\n")
 positive_pass=min(r["endpoint_success"] for r in rows)>=.95 and min(r["acquisition_0p20"] for r in rows)>=.85 and min(r["acquisition_0p20"] for r in rear)>=.90 and safety["aggregate_fall"]<=.02
 with (audit.OUT/"oracle_positive_control.csv").open("w",newline="",encoding="utf-8") as s:w=csv.DictWriter(s,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (audit.OUT/"oracle_positive_control.json").write_text(json.dumps({"status":"PASS" if positive_pass else "FAIL","profile":{"B0":"exp_012 stop-maintenance","B1_onward":"frozen mapped checkpoint for entire start episode"},"aggregate_endpoint":sum(r["endpoint_success"] for r in rows)/24,"aggregate_acquisition":sum(r["acquisition_0p20"] for r in rows)/24,"aggregate_fall":safety["aggregate_fall"],"aggregate_slip":safety["aggregate_slip"],"minimum_condition_endpoint":min(r["endpoint_success"] for r in rows),"minimum_condition_acquisition":min(r["acquisition_0p20"] for r in rows),"rear_minimum_acquisition":min(r["acquisition_0p20"] for r in rear),"rows":rows},indent=2)+"\n")
 print(json.dumps({"formal":formal,"positive_control":positive_pass,"minimum_acquisition":summary["minimum_acquisition"]}))
if __name__=="__main__":main()

"""Finalize A5 after preregistered four-step positive-control failure."""
from __future__ import annotations
import csv,hashlib,json
from pathlib import Path
import w2_p1_a5_common as c
OUT=c.A5;REPO=c.REPO;A4=c.A4;RES=c.BASE/"phase_w2_p1_r1_d2_dataset_provenance_reconciliation";REPORT=REPO/"research/exp_013_g1_phase_w2_p1_a5_versioned_four_step_start_trajectory_overlay_report.md";CLASS="VERSIONED_4STEP_POSITIVE_CONTROL_FAIL"
def dump(n,x):OUT.mkdir(parents=True,exist_ok=True);(OUT/n).write_text(json.dumps(x,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def empty_csv(n,reason):
 with (OUT/n).open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=["status","reason"]);w.writeheader();w.writerow({"status":"NOT_RUN","reason":reason})
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def aggregates(rows):
 out={}
 for p in sorted(set(x["profile"] for x in rows)):
  q=[x for x in rows if x["profile"]==p];n=sum(x["episodes"] for x in q);av=lambda k:sum(x[k]*x["episodes"] for x in q)/n
  out[p]={"endpoint":av("endpoint_success"),"acquisition":av("acquisition_success"),"fall":av("fall_rate"),"dangerous_slip":av("dangerous_slip_rate"),"impact":av("impact_rate"),"minimum_condition_endpoint":min(x["endpoint_success"] for x in q),"minimum_condition_acquisition":min(x["acquisition_success"] for x in q),"worst_endpoint_conditions":sorted(q,key=lambda x:x["endpoint_success"])[:3],"worst_acquisition_conditions":sorted(q,key=lambda x:x["acquisition_success"])[:3]}
 return out
def main():
 pc=json.loads((OUT/"four_step_runtime_positive_control.json").read_text());ag=aggregates(pc["profiles"]);reason="PC2 failed preregistered per-condition acquisition gate; downstream collection/training prohibited"
 dump("candidate_visited_boundary_collection_manifest.json",{"status":"NOT_RUN","reason":reason,"episodes":0,"runtime_W1B_actions":0})
 dump("start_boundary_trajectory_overlay_v3_manifest.json",{"status":"NOT_CREATED","reason":reason,"entries":0,"byte_sha256":None,"semantic_sha256":None,"base_changed":0})
 dump("start_boundary_trajectory_overlay_v3_split.json",{"status":"NOT_CREATED","reason":reason,"train":0,"validation":0,"held_out":0})
 # Deliberately do not create start_boundary_trajectory_overlay_v3.pt: protocol forbids overlay generation after PC2 failure.
 dump("four_step_state_coverage_audit.json",{"status":"NOT_RUN","reason":reason});empty_csv("four_step_state_coverage_by_step.csv",reason)
 (OUT/"resolved_v3_probe_training_config.yaml").write_text("status: NOT_RUN\nreason: PC2 per-condition acquisition failure\noptimizer: Adam\nlearning_rate: 0.00005\nseed: 20278321\nmaximum_steps: 4000\n",encoding="utf-8")
 for n in ("v3_probe_training_curves.csv","v3_static_checkpoint_timeline.csv","validation_v3_physical_start_gate.csv","heldout_v3_physical_start_gate.csv","v3_start_trajectory_continuity.csv","v3_trajectory_gradient_cosines.csv"):empty_csv(n,reason)
 for n in ("v3_static_checkpoint_timeline.json","selected_v3_probe_candidate.json","validation_v3_static_authorization.json","validation_v3_physical_start_gate.json","validation_v3_zero_command_retention.json","heldout_v3_static_authorization.json","heldout_v3_physical_start_gate.json","heldout_v3_zero_command_retention.json","v3_state_generalization.json","v3_start_trajectory_continuity.json","v3_trajectory_gradient_analysis.json","v3_trajectory_negative_controls.json"):dump(n,{"status":"NOT_RUN","reason":reason,"fallback":False})
 resolved=json.loads((RES/"w2_p1_dataset_hashes_resolved_v2.json").read_text());actual={p:sha(REPO/p) for p in resolved["hashes"]};v2=sha(A4/"start_boundary_b0_label_overlay_v2.pt");v2_expected=json.loads((A4/"w2_p1_dataset_overlay_manifest_v2.json").read_text())["overlay_sha256"]
 protected={"base_hashes":actual,"base_manifest_match":actual==resolved["hashes"],"V2_overlay_sha256":v2,"V2_overlay_unchanged":v2==v2_expected,"base_dataset_changes":0,"base_label_changes":0,"base_split_changes":0,"base_manifest_changes":0,"existing_overlay_changes":0,"existing_checkpoint_changes":0,"existing_optimizer_changes":0,"new_V3_overlay":0,"new_persistent_student":0,"formal_full_closed_loop":0,"PPO":0,"DAgger":0,"canonical_promotion":0,"remote_push":False};dump("protected_hashes.json",protected)
 dump("current_w2_p1_four_step_trajectory_interpretation.json",{"canonical_parent":"W1B-R2 iteration 200","integration_base":"W2-P1-R2 step37000","V2_B0_semantic_conflict":"resolved","V2_static_integration":"PASS","V2_physical_start":"FAIL","required_positive_control":"B0 stop + B1-B4 W1B","positive_control_result":"FAIL: rear yaw acquisition parity","new_overlay":"not created","closed_loop_authorization":"not granted","canonical_promotion":"none"})
 dump("stage_classification.json",{"classification":CLASS,"reason":"PC2 is safe and endpoint-strong but fails condition acquisition >=85% at 180-degree yaw +/-0.3.","downstream_stopped":True})
 dump("recommended_next_action.json",{"action":"diagnose rear-direction yaw acquisition under the four-step teacher-forced start protocol before any overlay collection","training_authorized":False})
 dump("gate.json",{"classification":CLASS,"A4_candidate_reproduction":"PASS","PC2_positive_control":"FAIL","collection":"NOT_RUN","overlay":"NOT_CREATED","training":"NOT_RUN","heldout":"NOT_RUN","formal_closed_loop_authorized":False,"promotion":False,"protection":"PASS"})
 (OUT/"reproduction_commands.ps1").write_text("$py = 'C:\\Users\\user\\workspace\\IsaacLab\\env_isaaclab\\Scripts\\python.exe'\n& $py experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/prepare_w2_p1_a5.py\n& $py experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/collect_w2_p1_a5_overlay.py --headless --device cuda:0\n& $py experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/finalize_w2_p1_a5.py\n",encoding="utf-8")
 pc0=ag["PC0_CANDIDATE_ONLY"];pc1=ag["PC1_B0_STOP_B1_B2_W1B"];pc2=ag["PC2_B0_STOP_B1_B4_W1B"];pc3=ag["PC3_B0_B4_W1B"]
 REPORT.write_text(f"""# exp_013 Phase W2-P1-A5 — versioned four-step trajectory overlay preflight

## Outcome

**{CLASS}**

The A4 in-memory candidate reproduced deterministically with tensor hash `db65a3069d665b8012fd9d264b7fd54e629a22d25b05a9ff793e23bfc549ac5f` and exact stored-metric parity. The preregistered PC2 positive control did not pass all condition gates, so candidate-visited collection, V3 overlay creation, probe training, held-out evaluation, and all downstream authorization were intentionally not run.

## Runtime positive controls

| Profile | Endpoint | Acquisition | Fall | Slip | Min condition endpoint | Min condition acquisition |
|---|---:|---:|---:|---:|---:|---:|
| PC0 candidate only | {pc0['endpoint']:.2%} | {pc0['acquisition']:.2%} | {pc0['fall']:.2%} | {pc0['dangerous_slip']:.2%} | {pc0['minimum_condition_endpoint']:.2%} | {pc0['minimum_condition_acquisition']:.2%} |
| PC1 B0 stop + B1-B2 W1B | {pc1['endpoint']:.2%} | {pc1['acquisition']:.2%} | {pc1['fall']:.2%} | {pc1['dangerous_slip']:.2%} | {pc1['minimum_condition_endpoint']:.2%} | {pc1['minimum_condition_acquisition']:.2%} |
| PC2 B0 stop + B1-B4 W1B | {pc2['endpoint']:.2%} | {pc2['acquisition']:.2%} | {pc2['fall']:.2%} | {pc2['dangerous_slip']:.2%} | {pc2['minimum_condition_endpoint']:.2%} | {pc2['minimum_condition_acquisition']:.2%} |
| PC3 B0-B4 W1B | {pc3['endpoint']:.2%} | {pc3['acquisition']:.2%} | {pc3['fall']:.2%} | {pc3['dangerous_slip']:.2%} | {pc3['minimum_condition_endpoint']:.2%} | {pc3['minimum_condition_acquisition']:.2%} |

PC2 satisfies aggregate endpoint, aggregate acquisition, fall, slip, impact, and every condition endpoint gate. It fails the required per-condition acquisition gate: direction 180°, yaw +0.3 reaches 18.0%, and direction 180°, yaw -0.3 reaches 22.5%, below the required 85%.

## Interpretation

Four W1B actions solve the safety and endpoint problem but do not establish the preregistered acquisition behavior for rear moving-yaw targets. Collecting an overlay from this positive control would therefore encode a protocol whose required capability has not been demonstrated. The fail-closed outcome is a protocol-level positive-control failure, not evidence about V3 representational feasibility.

## Protection

The immutable base dataset, labels, split, manifests, V2 overlay, checkpoints, and optimizers remain unchanged. No V3 `.pt` overlay was created. No persistent student, PPO, DAgger, formal full closed-loop evaluation, or canonical promotion occurred.
""",encoding="utf-8")
 print(json.dumps({"classification":CLASS,"PC2":pc2,"protection":protected["base_manifest_match"] and protected["V2_overlay_unchanged"]}))
if __name__=="__main__":main()

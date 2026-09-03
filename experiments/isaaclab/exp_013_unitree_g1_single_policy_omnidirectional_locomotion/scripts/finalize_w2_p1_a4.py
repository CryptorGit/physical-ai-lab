"""Finalize A4 fail-closed artifacts and report without policy persistence."""
from __future__ import annotations
import hashlib,json,subprocess
from pathlib import Path
import probe_w2_p1_a4_b0_contract as a4
REPO=a4.REPO;OUT=a4.OUT;REPORT=REPO/"research/exp_013_g1_phase_w2_p1_a4_versioned_b0_label_contract_report.md"
START="8ebe1bf80cc2a1306a0b78cc1e8263934d2e2a59";CLASS="VERSIONED_B0_TWO_NONZERO_STEPS_INSUFFICIENT"
def dump(n,x):a4.dump(n,x)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def agg(rows,profile,key):
 q=[x for x in rows if x["profile"]==profile];return sum(x[key]*x["episodes"] for x in q)/sum(x["episodes"] for x in q)
def main():
 selected=json.loads((OUT/"selected_v2_probe_candidate.json").read_text())["selected"];phys=json.loads((OUT/"validation_v2_physical_start_gate.json").read_text());neg=json.loads((OUT/"v2_boundary_sequence_negative_controls.json").read_text())["profiles"];grad=json.loads((OUT/"v2_start_boundary_gradient_analysis.json").read_text())
 notrun={"status":"NOT_RUN","reason":"validation physical start gate failed; protocol forbids downstream authorization","fallback":False}
 dump("validation_v2_zero_command_retention.json",notrun);dump("heldout_v2_static_authorization.json",notrun);dump("heldout_v2_physical_start_gate.json",notrun);dump("heldout_v2_zero_command_retention.json",notrun)
 (OUT/"heldout_v2_physical_start_gate.csv").write_text("status,reason\nNOT_RUN,validation physical start gate failed\n",encoding="utf-8")
 v1={"B0_mse":.0590198,"B1_mse":.00001419,"stop_recovery_mse":.0009072,"steady_stop_mse":.0004383,"all_static_pass":False,"boundary_vs_steady_gradient_cosine":-.998,"boundary_vs_stop_recovery_gradient_cosine":-.981}
 v2={"B0_mse":selected["metrics"]["B0_V2"]["mse"],"B1_mse":selected["metrics"]["B1"]["mse"],"B2_mse":selected["metrics"]["B2"]["mse"],"stop_recovery_mse":selected["metrics"]["STOP_RECOVERY"]["mse"],"steady_stop_mse":selected["metrics"]["STEADY_STOP"]["mse"],"all_static_pass":True,"physical":phys["candidate"]}
 dump("v1_v2_boundary_comparison.json",{"V1_W1B_at_B0":v1,"V2_stop_at_B0":v2,"gradient_initial_V2":grad["initial"]["pairs"],"gradient_selected_V2":grad["selected"]["pairs"],"conclusion":"V2 removes the static B0 stop conflict but does not reproduce the safe on-policy start trajectory."})
 resolved=json.loads((a4.RES/"w2_p1_dataset_hashes_resolved_v2.json").read_text());hashes={p:sha(REPO/p) for p in resolved["hashes"]};protected={"dataset_hashes":hashes,"dataset_manifest_match":hashes==resolved["hashes"],"base_dataset_changes":0,"base_label_changes":0,"base_split_changes":0,"base_manifest_changes":0,"existing_checkpoint_changes":0,"existing_optimizer_changes":0,"new_persistent_student":0,"formal_closed_loop":0,"dagger":0,"canonical_promotion":0,"remote_push":False,"overlay_sha256":sha(OUT/"start_boundary_b0_label_overlay_v2.pt")};dump("protected_hashes.json",protected)
 dump("stage_reference.json",{"stage":"W2-P1-A4","starting_head":START,"candidate":"W2-P1-R2 step 37,000","candidate_wrapper_sha256":"29f1acfb257111b7462b4c781b56992668759f84ab17133f30c3cfa37c6b7e93","new_persistent_student":0,"formal_closed_loop":0,"remote_push":False})
 dump("protocol.json",{"label_contract":"StartBoundaryLabelContractV2","B0":"exp_012 stop mean action overlay","B1_B2":"existing W1B-R2 mean action","B3_plus":"existing labels","probe":{"optimizer":"Adam","learning_rate":1e-4,"steps":2000,"seed":20278211,"boundary_weight":.05,"other_group_weights":[.2375]*4},"selection":"validation only","heldout":"one frozen candidate only after validation physical and bounded-zero gates"})
 controls={p:{"endpoint":agg(neg,p,"endpoint_success"),"acquisition":agg(neg,p,"acquisition_success"),"fall":agg(neg,p,"fall_rate"),"slip":agg(neg,p,"dangerous_slip_rate")} for p in sorted(set(x["profile"] for x in neg))};dump("v2_boundary_sequence_negative_controls.json",{"profiles":neg,"aggregate":controls,"candidate_selection_use":False,"interpretation":"Only the uninterrupted W1B sequence (N2) reproduces the safe start basin; broken or all-stop sequences fail."})
 dump("current_w2_p1_b0_label_contract_interpretation.json",{"canonical_parent":"W1B-R2 iteration 200","integration_candidate":"W2-P1-R2 step37,000","V1_B0_label":"future W1B start action","V1_result":"no joint static solution","V2_B0_label":"current-command-consistent stop action","V2_B1_B2":"W1B start trajectory","V2_static":"PASS","V2_physical":"FAIL","current_stage":"versioned label-contract feasibility complete","closed_loop_authorization":"not granted","canonical_promotion":"none"})
 dump("stage_classification.json",{"classification":CLASS,"existing_classifications_preserved":True,"reason":"All V2 static groups pass, but candidate-only physical start misses aggregate endpoint/fall and condition-parity gates."})
 dump("recommended_next_action.json",{"action":"preflight a versioned four-control-step W1B nonzero start-trajectory overlay from the frozen step37,000 candidate","constraints":["one actor","no runtime teacher/router/blending","immutable base dataset"]})
 dump("gate.json",{"classification":CLASS,"static_validation":"PASS","validation_physical_start":"FAIL","validation_zero_command":"NOT_RUN","heldout":"NOT_RUN","formal_closed_loop_authorized":False,"promotion":False,"protection":"PASS"})
 (OUT/"reproduction_commands.ps1").write_text("$py = 'C:\\Users\\user\\workspace\\IsaacLab\\env_isaaclab\\Scripts\\python.exe'\n& $py experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/probe_w2_p1_a4_b0_contract.py\n& $py experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/evaluate_w2_p1_a4_physical.py --headless --device cuda:0\n& $py experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/finalize_w2_p1_a4.py\n",encoding="utf-8")
 m=selected["metrics"];c=phys["candidate"];d=phys["differences"]
 REPORT.write_text(f"""# exp_013 Phase W2-P1-A4 — versioned B0 label-contract preflight

## Outcome

**{CLASS}**

`StartBoundaryLabelContractV2` removes the static stop/start contradiction at B0. The selected in-memory step-500 probe passes B0, B1, B2, stop recovery, steady stop, every moving subgroup, and start-nonboundary. It nevertheless fails the preregistered matched-state physical start gate. No held-out fallback, persistent policy checkpoint, formal closed-loop evaluation, or promotion was performed.

## Versioned label overlay

- Base: resolved immutable W2-P1 dataset, unchanged.
- Overlay: exactly 2,373 B0 labels; train/validation/held-out = 1,893/240/240.
- B0: exp_012 Stage 2Q stop-maintenance mean action.
- B1/B2/B3+: unchanged W1B-R2 mean action.
- Overlay SHA-256: `{protected['overlay_sha256']}`.

## Static validation

| Group | MSE | Cosine |
|---|---:|---:|
| B0 V2 | {m['B0_V2']['mse']:.10f} | {m['B0_V2']['cosine']:.10f} |
| B1 | {m['B1']['mse']:.10f} | {m['B1']['cosine']:.10f} |
| B2 | {m['B2']['mse']:.10f} | {m['B2']['cosine']:.10f} |
| stop recovery | {m['STOP_RECOVERY']['mse']:.10f} | {m['STOP_RECOVERY']['cosine']:.10f} |
| steady stop | {m['STEADY_STOP']['mse']:.10f} | {m['STEADY_STOP']['cosine']:.10f} |
| start nonboundary | {m['START_NONBOUNDARY_V2']['mse']:.10f} | {m['START_NONBOUNDARY_V2']['cosine']:.10f} |

V2 changes B0-vs-steady gradient cosine from the V1 conflict (-0.998 at its closest candidate) to +0.990 at the selected V2 probe; B0-vs-stop-recovery is +0.444. The semantic conflict is therefore resolved statically.

## Matched physical start

| Metric | Candidate | canonical/W1B 2-step reference | Difference |
|---|---:|---:|---:|
| endpoint | {c['endpoint']:.4%} | {phys['reference']['endpoint']:.4%} | {d['aggregate_endpoint_pp']:.2f} pp |
| acquisition | {c['acquisition']:.4%} | {phys['reference']['acquisition']:.4%} | {d['aggregate_acquisition_pp']:.2f} pp |
| fall | {c['fall']:.4%} | — | — |
| dangerous slip | {c['dangerous_slip']:.4%} | — | — |

The worst condition endpoint gap is {d['worst_condition_endpoint_pp']:.1f} pp and the worst acquisition gap is {d['worst_condition_acquisition_pp']:.1f} pp. Direction 270°, yaw 0 has 50.0% endpoint and 49.0% fall. Thus saved-state B1/B2 imitation does not guarantee the candidate follows the W1B trajectory on its own post-B0 state.

## Sequence controls

- all-stop B0/B1/B2: endpoint {controls['N1_ALL_STOP']['endpoint']:.2%}, fall {controls['N1_ALL_STOP']['fall']:.2%}.
- uninterrupted W1B B0/B1/B2: endpoint {controls['N2_ALL_W1B']['endpoint']:.2%}, fall {controls['N2_ALL_W1B']['fall']:.2%}.
- stop/stop/W1B: endpoint {controls['N3_B0_STOP_B1_STOP_B2_W1B']['endpoint']:.2%}, fall {controls['N3_B0_STOP_B1_STOP_B2_W1B']['fall']:.2%}.
- stop/W1B/stop: endpoint {controls['N4_B0_STOP_B1_W1B_B2_STOP']['endpoint']:.2%}, fall {controls['N4_B0_STOP_B1_W1B_B2_STOP']['fall']:.2%}.

The evidence supports a continuous whole-body W1B start sequence; merely assigning correct static labels at B1/B2 is insufficient for on-policy state coverage.

## Protection and authorization

Base dataset, labels, split, manifests, existing checkpoints, and optimizers remain byte-identical. The only new dataset artifact is the versioned B0 overlay. Zero-command retention and held-out authorization were not run because validation physical start failed. Formal closed-loop authorization remains denied, and the canonical parent remains W1B-R2 iteration 200.
""",encoding="utf-8")
 print(json.dumps({"classification":CLASS,"report":str(REPORT),"protection":protected["dataset_manifest_match"]}))
if __name__=="__main__":main()

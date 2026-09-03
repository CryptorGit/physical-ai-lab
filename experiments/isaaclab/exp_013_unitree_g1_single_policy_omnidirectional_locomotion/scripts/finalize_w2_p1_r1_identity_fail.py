"""Fail-closed W2-P1-R1 finalization for immutable dataset identity mismatch."""
from __future__ import annotations
import csv,hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve();REPO=HERE.parents[4]
OUT=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w2_p1_r1_group_balanced_stop_integration"
SRC=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w2_p1_practical_stop_endpoint_acquisition"
D1=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w2_p1_d1_static_representation_conflict_diagnosis"
REPORT=REPO/"research/exp_013_g1_phase_w2_p1_r1_group_balanced_stop_integration_report.md"
PARENT_SHA="61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d"
CLASS="EXP013_W2_P1_R1_DATASET_IDENTITY_FAIL"
def sha(p):
 h=hashlib.sha256();f=Path(p).open('rb')
 while b:=f.read(8<<20):h.update(b)
 f.close();return h.hexdigest()
def dump(n,v):OUT.mkdir(parents=True,exist_ok=True);(OUT/n).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def csv_status(n,fields):
 with (OUT/n).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerow({fields[0]:'NOT_EXECUTED_UPSTREAM_DATASET_IDENTITY_FAIL'})
def main():
 stage=json.loads((OUT/'stage_reference.json').read_text());stage['starting_status_short']=[s for s in stage['starting_status_short'] if 'w2_p1_r1' not in s.lower() and 'phase_w2_p1_r1_group_balanced_stop_integration' not in s and 'exp_013_g1_phase_w2_p1_r1_group_balanced_stop_integration_report.md' not in s];stage['starting_status_capture']='initial turn shell capture; R1-created paths filtered from the later machine-readable snapshot';dump('stage_reference.json',stage)
 audit=json.loads((OUT/'dataset_identity_audit.json').read_text());expected=audit['expected'];actual=audit['actual'];mismatch=[{'path':k,'expected_sha256':expected[k],'actual_sha256':actual[k]} for k in expected if expected[k]!=actual[k]]
 d1=json.loads((D1/'protected_hashes.json').read_text());d1_base=d1['baseline']
 for row in mismatch:row['matches_d1_baseline']=d1_base.get(row['path'])==row['actual_sha256']
 audit.update({'mismatches':mismatch,'classification':CLASS,'training_authorized':False,'p3_authorized':False,'note':'actual chunks match the D1 baseline but not the immutable W2-P1 dataset hash manifest named as R1 source of truth'});dump('dataset_identity_audit.json',audit)
 reason={'status':'NOT_EXECUTED','reason':'upstream dataset identity gate failed','classification':CLASS}
 dump('p3_probe_reproduction.json',reason);dump('p3_probe_process_parity.json',reason)
 csv_status('group_balanced_static_checkpoint_timeline.csv',['status','step','group']);dump('group_balanced_static_checkpoint_timeline.json',reason)
 csv_status('training_curves.csv',['status','step','loss']);dump('checkpoint_manifest.json',{'checkpoints':[],'new_persistent_checkpoints':0,**reason});dump('selected_checkpoint.json',{'selected':None,**reason});dump('selected_checkpoint_process_parity.json',reason)
 dump('static_heldout_results.json',reason);dump('exact_zero_boundary_results.json',reason)
 csv_status('closed_loop_static_stop.csv',['status','episode']);dump('closed_loop_static_stop.json',reason);dump('closed_loop_moving_retention.json',reason)
 for stem in ('formal_moving_to_stop_matrix','formal_stop_to_moving_matrix','formal_stop_move_stop_sequence'):
  csv_status(stem+'.csv',['status','condition']);dump(stem+'.json',reason)
 dump('dagger_rounds.json',{'rounds':0,'collected_states':0,'needed':False,'reason':'not authorized before dataset identity and static gates'})
 dump('safety_summary.json',reason);dump('transition_symmetry.json',reason)
 dump('single_checkpoint_audit.json',{'runtime_evaluation_executed':False,'new_checkpoint_count':0,'teacher_runtime':0,'expert':0,'router':0,'checkpoint_switch':0,'action_blending':0,'external_stop_controller':0})
 dump('canonical_stop_capable_walk_parent.json',{'promotion':False,'reason':CLASS,'canonical_parent_maintained':{'checkpoint':'W1B-R2 iteration 200','sha256':PARENT_SHA},'new_checkpoint':None})
 dump('current_w2_p1_r1_artifact_interpretation.json',{'stop_teacher':'24/24 recovery positive control PASS','124D_architecture':'joint static representation feasible in D1 probe','original_W2_P1_failure':'optimization-path dependent','group_balanced_student':'not created; dataset identity gate failed','canonical_parent_before_PASS':'W1B-R2 iteration 200','full_W2':'not restarted','RUN':'not evaluated'})
 dump('stage_classification.json',{'classification':CLASS,'mismatch_count':len(mismatch),'mismatches':mismatch,'formal_training_started':False,'closed_loop_started':False,'canonical_promotion':False})
 dump('recommended_next_action.json',{'classification':CLASS,'one_method':'reconcile the immutable W2-P1 dataset hash manifest against the D1-protected chunks, without changing dataset or label bytes, before authorizing R1 again'})
 protected={k:{'starting_sha256':actual[k],'ending_sha256':sha(REPO/k),'unchanged_during_R1':actual[k]==sha(REPO/k)} for k in actual}
 dump('protected_hashes.json',{'datasets':protected,'all_dataset_bytes_unchanged_during_R1':all(v['unchanged_during_R1'] for v in protected.values()),'existing_checkpoints_changed':False,'existing_optimizers_changed':False,'existing_stages_changed':False})
 dump('gate.json',{'dataset_identity':'FAIL','p3_reproduction':'NOT_EXECUTED','persistent_training':'NOT_EXECUTED','static_authorization':'NOT_EXECUTED','closed_loop':'NOT_EXECUTED','dagger':'NOT_EXECUTED','new_persistent_checkpoint':0,'canonical_promotion':0,'remote_push':False,'classification':CLASS})
 repro='''$repo = "C:\\Users\\user\\workspace\\physical-ai-lab"\n$python = "C:\\Users\\user\\workspace\\IsaacLab\\env_isaaclab\\Scripts\\python.exe"\nSet-Location $repo\ngit rev-parse HEAD\ngit status --short\ngit log --oneline --decorate -25\n& $python experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/prepare_w2_p1_r1.py\n# Fail closed: do not invoke P3 or formal training while dataset_identity_audit.json is FAIL.\n& $python experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/finalize_w2_p1_r1_identity_fail.py\n''';(OUT/'reproduction_commands.ps1').write_text(repro,encoding='utf-8')
 report=f'''# exp_013 Phase W2-P1-R1 group-balanced stop integration\n\n## Outcome\n\nClassification: `{CLASS}`. The formal run stopped at the first immutable-input gate. No P3 replay, persistent student training, closed-loop rollout, DAgger, checkpoint creation, or promotion was executed.\n\n## Dataset identity\n\nThe preregistered source of truth was `w2_p1_dataset_hashes.json`. Two existing chunks differ:\n\n'''+''.join(f"- `{r['path']}`: expected `{r['expected_sha256']}`, actual `{r['actual_sha256']}`. Actual matches D1 baseline: `{r['matches_d1_baseline']}`.\n" for r in mismatch)+f'''\nThe current bytes match the hashes captured at both the start and end of W2-P1-D1, so this run did not modify them. Nevertheless, R1 explicitly requires agreement with the existing W2-P1 hash manifest, and that condition is false. Treating the later D1 audit as a replacement source would silently change the requested provenance contract, so the run failed closed.\n\n## P3 and training\n\nThe committed D1 probe source contains a complete P3 contract (Adam, LR 2e-4, seed 20277717, 2,000 steps, clip 10, fixed pool/validation seeds, 25/25/25/25 objective). It was not executed because dataset identity is an earlier hard gate. No formal checkpoint exists.\n\n## Closed loop and DAgger\n\nNot authorized. All static-stop, moving-retention, transition, safety, and symmetry outputs explicitly record `NOT_EXECUTED_UPSTREAM_DATASET_IDENTITY_FAIL`.\n\n## Canonical artifact\n\nW1B-R2 iteration 200 (`{PARENT_SHA}`) remains canonical.\n\n## Next\n\nOne method only: reconcile the immutable W2-P1 dataset hash manifest against the D1-protected chunks without changing dataset or label bytes, then request R1 authorization again.\n\n## Protection\n\nAll current dataset bytes remained unchanged during R1; no existing checkpoint, optimizer, stage, sampler, reward, physics, calibration, evaluator, Isaac Lab core, or RSL-RL package was changed. Remote push was not performed.\n'''
 REPORT.parent.mkdir(parents=True,exist_ok=True);REPORT.write_text(report,encoding='utf-8');print(json.dumps({'classification':CLASS,'mismatches':len(mismatch),'training_started':False}))
if __name__=='__main__':main()

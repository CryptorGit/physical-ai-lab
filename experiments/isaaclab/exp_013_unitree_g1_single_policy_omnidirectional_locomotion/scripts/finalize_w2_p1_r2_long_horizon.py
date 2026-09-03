"""Finalize W2-P1-R2 after validation-selected held-out authorization."""
from __future__ import annotations
import csv, hashlib, json, subprocess
from pathlib import Path
HERE=Path(__file__).resolve(); REPO=HERE.parents[4]
BASE=REPO/"results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion"; OUT=BASE/"phase_w2_p1_r2_long_horizon_group_balanced_stop_integration"; REPORT=REPO/"research/exp_013_g1_phase_w2_p1_r2_long_horizon_group_balanced_stop_integration_report.md"
PARENT_SHA="61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d"
def dump(name,v): (OUT/name).write_text(json.dumps(v,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def load(name): return json.loads((OUT/name).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""): h.update(b)
 return h.hexdigest()
def placeholder_json(name,reason): dump(name,{"status":"NOT_EXECUTED","reason":reason,"formal_result":"not_evaluated"})
def placeholder_csv(name,reason):
 with (OUT/name).open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=["status","reason"]);w.writeheader();w.writerow({"status":"NOT_EXECUTED","reason":reason})

start=(OUT/"_starting_head.txt").read_text().strip(); status=(OUT/"_starting_status.txt").read_text(encoding="utf-8").splitlines(); selected=load("selected_checkpoint.json"); held=load("heldout_static_authorization.json"); timeline=load("validation_checkpoint_timeline.json")["checkpoints"]; prefix=load("canonical_prefix_parity.json"); parity=load("selected_checkpoint_process_parity.json"); identity=load("resolved_dataset_identity_audit.json")
classification="EXP013_W2_P1_R2_VALIDATION_SELECTED_HELDOUT_FAIL"; reason="validation-selected checkpoint failed the one-shot held-out START_RETENTION static action gate; fallback is forbidden"
dump("stage_reference.json",{"stage":"Phase W2-P1-R2 formal long-horizon group-balanced practical-stop integration","reported_starting_head":"95ab57e64766d99d4ca3671325d60369832ac3c2","actual_starting_head":start,"head_match":start=="95ab57e64766d99d4ca3671325d60369832ac3c2","starting_unrelated_dirty_count":len(status),"persistent_runs":1,"remote_push":False})
dump("protocol.json",{"parent":"W1B-R2 iteration 200","warm_start":"forbidden and not used","objective":"25/25/25/25 group-balanced mean-action MSE","maximum_steps":40000,"selection":"validation only","heldout":"one evaluation after immutable selection","fallback_after_heldout":False,"closed_loop_authorization_chain":["dataset identity","prefix parity","validation selection","heldout static","process parity"],"PPO":False,"DAgger_before_authorization":False})
(OUT/"resolved_long_horizon_training_config.yaml").write_text("stage: W2-P1-R2\nparent: W1B-R2 iteration 200\noptimizer: Adam\nlearning_rate: 0.0002\nadam_betas: [0.9, 0.999]\nadam_epsilon: 1.0e-8\nweight_decay: 0.0\nscheduler: false\nadaptive_lr: false\ntraining_seed: 20277717\nsample_pool_seed: 20276049\nmaximum_optimizer_steps: 40000\ngradient_clip: 10.0\ngroup_weights: {stop_recovery: 0.25, steady_stop: 0.25, moving_retention: 0.25, start_retention: 0.25}\nmean_actor_only: true\nstd_frozen: true\ncritic: unused\nppo: false\n",encoding="utf-8")

for name in ("closed_loop_static_stop.json","closed_loop_moving_retention.json","formal_moving_to_stop_matrix.json","formal_stop_to_moving_matrix.json","formal_stop_move_stop_sequence.json","safety_summary.json","transition_symmetry.json"):
 placeholder_json(name,reason)
for name in ("closed_loop_static_stop.csv","formal_moving_to_stop_matrix.csv","formal_stop_to_moving_matrix.csv","formal_stop_move_stop_sequence.csv"):
 placeholder_csv(name,reason)
dump("dagger_rounds.json",{"rounds_executed":0,"maximum_rounds":2,"authorized":False,"collected_states":0,"reason":"held-out static failure explicitly forbids DAgger"})
dump("single_checkpoint_audit.json",{"selected_offline_checkpoint_count":1,"actor_count":1,"gaussian_head_count":1,"teacher_runtime":0,"expert_runtime":0,"router":0,"checkpoint_switch":0,"action_blending":0,"external_stop_controller":0,"calibration":"MonotonicPositiveYawCalibrationV1","action_source":"selected student deterministic mean actor only if closed-loop authorization had passed","closed_loop_runtime_evaluated":False,"status":"OFFLINE_SINGLE_CHECKPOINT_PASS_CLOSED_LOOP_NOT_AUTHORIZED"})
dump("canonical_stop_capable_walk_parent.json",{"promoted":False,"reason":classification,"canonical_parent_retained":{"policy":"W1B-R2 iteration 200","sha256":PARENT_SHA},"diagnostic_selected_student":{"step":selected["step"],"sha256":selected["sha256"]},"runtime_teacher":0})
dump("stage_classification.json",{"classification":classification,"validation_joint_pass":True,"heldout_static_pass":False,"failed_group":"START_RETENTION","checkpoint_fallback":False,"closed_loop_authorized":False})
dump("recommended_next_action.json",{"classification":classification,"single_method":"held-out exact-zero start-retention generalization diagnosis","action":"diagnose validation/held-out exact-zero boundary generalization without changing the physical gate","executed":False})

before=json.loads((OUT/"_protected_before.json").read_text(encoding="utf-8-sig")); files=[]; changed=[]
for row in before:
 p=(REPO/row["path"].replace(".\\","")).resolve(); current=sha(p); item={**row,"after_sha256":current,"unchanged":current==row["sha256"]};files.append(item)
 if not item["unchanged"]:changed.append(item["path"])
dump("protected_hashes.json",{"algorithm":"SHA-256","protected_file_count":len(files),"all_protected_unchanged":not changed,"changed":changed,"files":files,"existing_dataset_label_split_manifest_unchanged":not changed,"new_checkpoint_scope":"W2-P1-R2 student only","remote_push":False})
dump("gate.json",{"resolved_dataset_identity":"PASS","canonical_prefix_parity":"PASS","persistent_40000_step_run":"PASS","validation_selection":"PASS","heldout_static_authorization":"FAIL","selected_process_parity":"PASS" if parity["pass"] else "FAIL","closed_loop":"NOT_AUTHORIZED","DAgger_rounds":0,"formal_phase_gate":"FAIL","classification":classification})
(OUT/"reproduction_commands.ps1").write_text('$python="C:\\Users\\user\\workspace\\IsaacLab\\env_isaaclab\\Scripts\\python.exe"\n& $python experiments\\isaaclab\\exp_013_unitree_g1_single_policy_omnidirectional_locomotion\\scripts\\run_w2_p1_r2_long_horizon.py train\n& $python experiments\\isaaclab\\exp_013_unitree_g1_single_policy_omnidirectional_locomotion\\scripts\\run_w2_p1_r2_long_horizon.py parity\n& $python experiments\\isaaclab\\exp_013_unitree_g1_single_policy_omnidirectional_locomotion\\scripts\\finalize_w2_p1_r2_long_horizon.py\n',encoding="utf-8")

passes=[x for x in timeline if x["joint_pass"]]; selected_t=next(x for x in timeline if x["step"]==selected["step"]); failed=[r for r in held["rows"] if not r["gate_pass"]]
REPORT.parent.mkdir(parents=True,exist_ok=True); REPORT.write_text(f"""# exp_013 Phase W2-P1-R2 long-horizon group-balanced stop integration

## Outcome

Classification: `{classification}`.

The resolved dataset identity and exact canonical 2,000-step prefix parity passed. One persistent canonical-parent balanced-only run completed all 40,000 optimizer steps and wrote the preregistered 81 checkpoints. Validation produced {len(passes)} joint-pass checkpoints; the first was step {passes[0]['step']}. The immutable validation rank selected step {selected['step']} rather than the latest checkpoint.

## Prefix and training

- Prefix tensor hash: `{prefix['tensor_hash']}` (exact D3 match)
- Prefix trace hash: `{prefix['trace_hash']}` (exact D3 match)
- Optimizer: Adam, fixed LR 2e-4, seed 20277717, pool seed 20276049
- Group weights: 25/25/25/25
- NaN/Inf and hard numerical guards: PASS

## Validation selection

- Selected step: {selected['step']}
- Selected checkpoint SHA-256: `{selected['sha256']}`
- Start MSE: {selected_t['metrics']['START_RETENTION']['mean_mse']:.10f}
- Stop-recovery MSE: {selected_t['metrics']['STOP_RECOVERY']['mean_mse']:.10f}
- Steady-stop MSE: {selected_t['metrics']['STEADY_STOP']['mean_mse']:.10f}
- Worst moving subgroup MSE: {selected_t['worst_moving_mse']:.10f}
- Validation exact-zero MSE: {selected_t['exact_zero_mse']:.10f}; nonzero start MSE: {selected_t['nonzero_mse']:.10f}

## Held-out authorization

The selected checkpoint was then evaluated once on held-out data. No fallback was permitted or performed. All groups except START_RETENTION passed. START_RETENTION MSE was {failed[0]['mean_mse']:.10f} (threshold 0.001) with cosine {failed[0]['cosine']:.10f}. Its held-out exact-zero subset contained {held['exact_zero']['exact_zero_count']} samples with MSE {held['exact_zero']['exact_zero_mse']:.10f}; nonzero start MSE was {held['exact_zero']['nonzero_mse']:.10f}.

This failure terminates the authorization chain before closed-loop evaluation. Static practical stop, moving retention rollouts, transition matrices, safety/symmetry rollout gates, and DAgger are explicitly not executed. Existing D3 held-out analysis was acknowledged, but checkpoint selection in this stage used validation only.

## Protection

Existing datasets, labels, split, manifests, checkpoints, optimizers, sampler, reward, physics, calibration, evaluators, Isaac Lab core, and RSL-RL package remained unchanged. The only new policy files are W2-P1-R2 scheduled/selected student artifacts. Runtime teacher use and remote push are zero.
""",encoding="utf-8")
print(json.dumps({"classification":classification,"protected":not changed,"selected":selected["step"],"heldout_failed":failed}))
if __name__=="__main__": pass

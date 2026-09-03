"""Finalize D18 after its fail-closed causal preflight."""
from __future__ import annotations
import csv, hashlib, json, subprocess
from pathlib import Path

HERE=Path(__file__).resolve(); REPO=HERE.parents[4]
OUT=REPO/"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d18_early_support_yaw_objective"; RAW=OUT/"raw"
REPORT=REPO/"research/exp_014_phase_2_d18_early_support_yaw_objective_report.md"; START="34820fb6fe413a4e1ed1979ccd12323e631da155"
def dump(name,x): OUT.mkdir(parents=True,exist_ok=True); (OUT/name).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def csvwrite(name,rows):
 p=OUT/name
 if not rows: p.write_text("status,reason\nNOT_EXECUTED,causal preflight failed before persistent training\n",encoding="utf-8-sig"); return
 keys=[]
 for r in rows:
  for k in r:
   if k not in keys: keys.append(k)
 with p.open("w",newline="",encoding="utf-8-sig") as f: w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

x=json.loads((RAW/"d18_results.json").read_text(encoding="utf-8")); refs=x["reference"]; cal=x["calibration"]; pre=x["causal_preflight"]; stab=x["first_update_stability"]
classification="EXP014_D18_SUPPORT_TRANSFER_OBJECTIVE_NO_EFFECT"
dump("stage_reference.json",{"phase":"2-D18","starting_head":START,"actual_starting_head":START,"actual_head_is_source_of_truth":True,"D17_classification":"EXP014_D17_YAW_REWARD_CAUSALITY_FAIL","condition":{"direction_deg":0,"speed_mps":.3,"yaw_rate_rad_s":0},"persistent_updates":x["persistent_updates"],"remote_push":False})
dump("protocol.json",{"name":"Exp014EarlySupportYawPrecursorExperimentV1","architecture":"D16 frozen W_MOVE plus additive residual","parent":"D16 initial zero-residual checkpoint","seed":20279401,"maximum_updates":40,"rollout_steps":100,"episode_steps":150,"learning_rate":1.5e-5,"interaction_budget":1904000,"preflight_fail_closed":True,"C2_or_later":0,"actor_input_dimension":141,"privileged_reward_only":True})
dump("reference_distribution_manifest.json",{"train_only":True,"source_snapshots":refs["source_snapshot_count"],"source_refresh_steps_per_snapshot":10,"wmove_basin_states":refs["basin_state_count"],"statistics":["median","p75","p90","p95"],"S_HOLD_source":refs["source"],"W_MOVE_basin":refs["wmove_basin"],"derived_scales":refs["derived_scales"],"target_load_imbalance_peak":refs["target_load_imbalance_peak"],"validation_used_for_scale":False})
dump("privileged_quantity_contract.json",{"actor_input_added_dimensions":0,"use":"reward and diagnostics only","sampling":"after control-step physics, dt=0.02 s","frames":{"contact_force":"world","body_COM":"world","Lz_axis":"world up","base_yaw_acceleration":"world z"},"Lz":"sum_i cross(p_i-COM,m_i*v_i)_z + (I_world_i*omega_world_i)_z","dLz_dt":"backward finite difference / 0.02","contact_yaw_moment":"sum feet cross(foot_pos-COM, net_contact_force)_z","support_force":"max(world-z foot force,0)","filter":"none; control-rate samples","quantities":["left/right vertical contact force","foot tangential speed","Lz","dLz/dt","yaw acceleration","contact yaw moment","pelvis vertical velocity","roll/pitch proxy","total vertical support"]})
dump("support_schedule_contract.json",{"name":"Exp014ForwardStartSupportScheduleV1","symmetric":True,"support_leg_scripted":False,"target_peak":refs["target_load_imbalance_peak"],"target_peak_derivation":"median W_MOVE dominant/single-support load imbalance clamped to [0.35,0.70]","schedule":[{"range_s":[0,.35],"target":"minimum jerk 0 to peak"},{"range_s":[.35,.5],"target":"peak hold"},{"range_s":[.5,.75],"weight":"minimum jerk to zero"},{"range_s":[.75,None],"weight":0}]})
dump("reward_v2_contract.json",{"name":"Exp014OmnidirectionalStartRewardV2","preventive_yaw":{"terms":["exp(-(Lz/sigma_Lz)^2)","exp(-(dLz_dt/sigma_dLz)^2)","exp(-(contact_yaw_moment/sigma_Mz)^2)"],"full_s":[0,.5],"decay_s":[.5,.75]},"support_transfer":{"load_transfer":True,"total_support_preservation":True,"support_load_weighted_slip":True,"swing_unloading_s":[.2,.6]},"velocity_tracking_schedule":[[0,.2,.15],[.2,.5,".15 to .60"],[.5,.75,".60 to 1.0"],[.75,None,1]],"yaw_tracking_schedule":[[0,.2,.25],[.2,.5,".25 to .75"],[.5,None,1]],"existing_safety_terms_unchanged":True,"hard_success_bonus":False,"imitation":False,"calibrated_family_scales":cal["deterministic_proportional_scales"]})
dump("reward_gradient_calibration.json",cal); dump("one_update_causal_preflight.json",pre); dump("first_update_stability.json",stab)
csvwrite("training_timeline.csv",x["timeline"]); dump("training_timeline.json",{"status":"NOT_EXECUTED","reason":"P_SUPPORT causal probe failed the registered 10% load-error reduction gate","persistent_updates":0,"interactions":0,"rows":x["timeline"]})
dump("checkpoint_manifest.json",{"status":"NOT_EXECUTED","reason":"persistent training was not authorized","new_checkpoints":0,"initial_parent":"D16 initial zero residual","manifest":x["checkpoint_manifest"]})
csvwrite("forward_validation_timeline.csv",x["validation_timeline"]); dump("forward_validation_timeline.json",{"status":"NOT_EXECUTED","reason":"no persistent checkpoint existed","rows":x["validation_timeline"]})
b=pre["baseline"]; py=pre["probes"]["P_PREVENTIVE_YAW"]; ps=pre["probes"]["P_SUPPORT"]
dump("yaw_momentum_metrics.json",{"baseline":{"yaw_p95_0p5s":b["yaw_p95_0p5s"],"early_Lz_p95":b["early_abs_Lz_p95"],"early_dLz_dt_p95":b["early_abs_dLz_dt_p95"],"contact_yaw_moment_p95":b["contact_yaw_moment_p95"]},"P_PREVENTIVE_YAW":{"yaw_p95_0p5s":py["yaw_p95_0p5s"],"early_Lz_p95":py["early_abs_Lz_p95"],"early_dLz_dt_p95":py["early_abs_dLz_dt_p95"],"contact_yaw_moment_p95":py["contact_yaw_moment_p95"]},"Lz_reduction":pre["preventive_reduction"]["Lz"],"dLz_reduction":pre["preventive_reduction"]["dLz_dt"],"yaw_p95_reduction":1-py["yaw_p95_0p5s"]/b["yaw_p95_0p5s"],"causal_probe_pass":pre["preventive_yaw_pass"]})
dump("support_transfer_metrics.json",{"baseline":{"load_target_error":b["load_target_error_mean"],"total_support_error":b["total_support_error_mean"],"support_slip":b["support_slip_mean"]},"P_SUPPORT":{"load_target_error":ps["load_target_error_mean"],"total_support_error":ps["total_support_error_mean"],"support_slip":ps["support_slip_mean"]},"load_target_error_reduction":pre["support_target_error_reduction"],"total_support_error_change":ps["total_support_error_mean"]-b["total_support_error_mean"],"dominant_support_time":"NOT_EVALUATED_PERSISTENTLY","swing_unload_time":"NOT_EVALUATED_PERSISTENTLY","causal_probe_pass":pre["support_transfer_pass"]})
dump("basin_distance_timeline.json",{"status":"NOT_EXECUTED","reason":"registered preflight stopped before persistent training; D18 reward explicitly forbids basin-distance reward"})
dump("selected_checkpoint.json",{"status":"NOT_SELECTED","reason":"support-transfer causal preflight failed","persistent_updates":0,"new_checkpoint":0})
dump("stage_classification.json",{"classification":classification,"preflight":"FAIL","preventive_yaw_probe":"PASS","support_transfer_probe":"FAIL","persistent_updates":0,"D17_unchanged":True})
dump("recommended_next_action.json",{"single_experiment":"diagnose the first support-transfer objective failure before any persistent residual PPO","do_not_train":True,"do_not_expand_C2":True,"direct_141D_transition_actor_not_yet_authorized_by_D18_No-Go_rule":True,"reason":"the registered causal preflight failed before the 40-update No-Go condition could be tested"})
diff=subprocess.check_output(["git","diff","--name-only",START],cwd=REPO,text=True).splitlines(); protected=[p for p in diff if any(f"phase_2_d{i}" in p.replace("\\","/") for i in range(6,18)) or any(f"exp_{i:03d}_" in p.replace("\\","/") for i in range(5,14))]
dump("protected_hashes.json",{"starting_head":START,"exp005_exp013_changed_by_D18":0,"D6_D17_changed_by_D18":0,"preexisting_unrelated_dirty_paths_preserved":protected,"W_MOVE_unchanged":True,"S_HOLD_unchanged":True,"S_STOP_OMNI_unchanged":True,"persistent_updates":0,"C2_or_later":0,"RUN":0,"integrated_student":0,"remote_push":False})
(OUT/"reproduction_commands.ps1").write_text("& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d18_precursor.py --headless --device cuda:0\npython experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/finalize_phase2_d18.py\n",encoding="utf-8")
REPORT.write_text(f"""# EXP014 Phase 2-D18 early support/yaw precursor objective

## Outcome

**{classification}**. The registered causal preflight failed, so persistent PPO did not start: 0 updates, 0 persistent interactions, and 0 new checkpoints.

## Reference scales

The train-only reference used 64 S_HOLD source snapshots (10 contact-refresh samples each) and 10,240 safe W_MOVE forward-basin states. The deterministic dominant-support target was {refs['target_load_imbalance_peak']:.4f}. Derived scales were sigma_Lz={refs['derived_scales']['sigma_Lz']:.6f}, sigma_dLz={refs['derived_scales']['sigma_dLz']:.6f}, sigma_Mz={refs['derived_scales']['sigma_Mz']:.6f}, sigma_load={refs['derived_scales']['sigma_load']:.6f}, and sigma_support={refs['derived_scales']['sigma_support']:.6f}.

## Gradient calibration and causal probes

One result-blind proportional calibration placed preventive yaw, support transfer, tracking, and safety/regularization gradient ratios at {cal['gradient_ratios_to_total']['preventive_yaw']:.2%}, {cal['gradient_ratios_to_total']['support_transfer']:.2%}, {cal['gradient_ratios_to_total']['tracking']:.2%}, and {cal['gradient_ratios_to_total']['safety_regularization']:.2%}; all registered ranges passed.

The preventive-yaw one-update probe reduced early |Lz| p95 by {pre['preventive_reduction']['Lz']:.2%}, with safety degradation within the 2 pp allowance, so that sub-gate passed. It did not reduce |dLz/dt| (change {pre['preventive_reduction']['dLz_dt']:.2%}) and yaw p95 changed from {b['yaw_p95_0p5s']:.4f} to {py['yaw_p95_0p5s']:.4f}.

The support probe failed decisively: load-target error changed from {b['load_target_error_mean']:.4f} to {ps['load_target_error_mean']:.4f}, a {pre['support_target_error_reduction']:.2%} reduction (negative means regression), while total-support error changed from {b['total_support_error_mean']:.4f} to {ps['total_support_error_mean']:.4f}. The required 10% improvement was not present.

## Stability and safety

The all-V2 temporary update itself was numerically stable: exact KL {stab['exact_kl']:.8f}, all-step KL {stab['all_step_kl']:.8f}, clip fraction {stab['clip_fraction']:.1%}, mean final-action shift {stab['mean_final_action_shift']:.4f}, with 0% fall, dangerous slip, and torque saturation. Failure is therefore causal ineffectiveness of the support precursor, not optimizer instability.

## Decision

Per the registered stop rule, no persistent update, C2 expansion, or checkpoint selection was allowed. The next single experiment is a focused diagnosis of why the symmetric support-transfer reward produces the wrong load trajectory; D18 does not meet the separate 40-update criterion needed to declare the entire additive-residual PPO route No-Go.

## Protection

exp_005-exp_013, D6-D17 artifacts, W_MOVE, S_HOLD, S_STOP_OMNI, datasets, optimizer, physics, PD gains, friction, robot assets, command/observation contracts, and formal gates were unchanged. RUN, integrated Student, C2+, and remote push are zero.
""",encoding="utf-8")
print(json.dumps({"classification":classification,"persistent_updates":0,"artifacts":23},indent=2))

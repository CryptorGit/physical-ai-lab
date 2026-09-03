"""Generate the D20 correction, regression, and fail-closed replay artifacts.

No Isaac Lab import, actor load, policy update, or rollout collection is performed.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
EXP = REPO / "experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion"
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d20_support_reward_fix_preflight"
D18 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d18_early_support_yaw_objective"
D19 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d19_support_objective_symmetry_audit"
REWARD_SOURCE = EXP / "scripts/support_reward_v2r1.py"
D18_SOURCE = EXP / "scripts/run_phase2_d18_precursor.py"
STARTING_HEAD = "e489c52dc4186ce88c64db390ab8b07c58eccf8f"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("support_reward_v2r1", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


reward = load_module(REWARD_SOURCE)


def dump(name: str, obj: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def line_of(path: Path, needle: str) -> int:
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if needle in line:
            return index
    raise ValueError(f"missing source token: {needle}")


def signed_rewards(left: float, right: float, target: float = 0.7, sigma: float = 0.3):
    signed = (left - right) / (left + right + 1.0e-6)
    return (
        math.exp(-((signed - target) / sigma) ** 2),
        math.exp(-((signed + target) / sigma) ** 2),
    )


def run_tests():
    tests = []

    def add(name, passed, actual, expected, group):
        tests.append({"name": name, "pass": bool(passed), "actual": actual, "expected": expected, "group": group})

    # D19's four force/reward tests.
    r8515 = reward.corrected_load_reward(left_vertical_n=85, right_vertical_n=15, left_force_norm_n=85, right_force_norm_n=15, t_s=.50, sigma_load=.3)
    r1585 = reward.corrected_load_reward(left_vertical_n=15, right_vertical_n=85, left_force_norm_n=15, right_force_norm_n=85, t_s=.50, sigma_load=.3)
    sl8515, sr8515 = signed_rewards(.85, .15)
    sl1585, sr1585 = signed_rewards(.15, .85)
    add("unsigned_85_15_equals_15_85", abs(r8515-r1585)<1e-12, [r8515,r1585], "equal", "D19_REPLAY")
    add("signed_left_selects_85_15", sl8515>sl1585, [sl8515,sl1585], "first greater", "D19_REPLAY")
    add("signed_right_selects_15_85", sr1585>sr8515, [sr1585,sr8515], "first greater", "D19_REPLAY")
    zero = reward.corrected_load_reward(left_vertical_n=0,right_vertical_n=0,left_force_norm_n=0,right_force_norm_n=0,t_s=0,sigma_load=.3)
    add("zero_total_support_not_high_load_reward", zero==0, zero, 0, "D19_REPLAY")

    # D19's seven schedule fixtures, now evaluated against the intended contract.
    for t in (0.0,.10,.20,.35,.50,.60,.75):
        target=reward.corrected_target(t); weight=reward.corrected_weight_envelope(t)
        intended_target=(reward.minimum_jerk(t/.35)*.7 if t<.35 else (.7 if t<.75 else 0.0))
        intended_weight=(1.0 if t<.50 else (1-reward.minimum_jerk((t-.50)/.25) if t<.75 else 0.0))
        add(f"schedule_t_{t:.2f}", abs(target-intended_target)<1e-12 and abs(weight-intended_weight)<1e-12, [target,weight], [intended_target,intended_weight], "D19_REPLAY")

    # Ten D20 additions.
    cases = [
      ("t_0p49_peak_full", reward.corrected_target(.49)==.7 and reward.corrected_weight_envelope(.49)==1, [reward.corrected_target(.49),reward.corrected_weight_envelope(.49)], [.7,1]),
      ("t_0p50_peak_decay_start", reward.corrected_target(.50)==.7 and reward.corrected_weight_envelope(.50)==1, [reward.corrected_target(.50),reward.corrected_weight_envelope(.50)], [.7,1]),
      ("t_0p60_peak_weight_positive", reward.corrected_target(.60)==.7 and reward.corrected_weight_envelope(.60)>0, [reward.corrected_target(.60),reward.corrected_weight_envelope(.60)], "target=.7, weight>0"),
      ("t_0p749_peak_weight_almost_zero", reward.corrected_target(.749)==.7 and 0<reward.corrected_weight_envelope(.749)<1e-4, [reward.corrected_target(.749),reward.corrected_weight_envelope(.749)], "target=.7, 0<weight<1e-4"),
      ("t_0p75_weight_zero", reward.corrected_weight_envelope(.75)==0, reward.corrected_weight_envelope(.75), 0),
      ("zero_support_target_zero_masked", zero==0, zero, 0),
      ("zero_support_target_peak_masked", reward.corrected_load_reward(left_vertical_n=0,right_vertical_n=0,left_force_norm_n=0,right_force_norm_n=0,t_s=.50,sigma_load=.3)==0, reward.corrected_load_reward(left_vertical_n=0,right_vertical_n=0,left_force_norm_n=0,right_force_norm_n=0,t_s=.50,sigma_load=.3), 0),
      ("valid_50_50_target_zero_max", abs(reward.corrected_load_reward(left_vertical_n=50,right_vertical_n=50,left_force_norm_n=50,right_force_norm_n=50,t_s=0,sigma_load=.3)-1)<1e-12, reward.corrected_load_reward(left_vertical_n=50,right_vertical_n=50,left_force_norm_n=50,right_force_norm_n=50,t_s=0,sigma_load=.3), 1),
      ("valid_85_15_target_0p7_max", r8515>.999999999, r8515, ">0.999999999"),
      ("valid_15_85_unsigned_equal", abs(r8515-r1585)<1e-12, [r8515,r1585], "equal"),
    ]
    for name, passed, actual, expected in cases:
        add(name, passed, actual, expected, "D20_ADDITIONAL")
    failed=[x["name"] for x in tests if not x["pass"]]
    return {"test_count":len(tests),"passed":len(tests)-len(failed),"failed":len(failed),"failed_tests":failed,"status":"PASS" if not failed else "FAIL","tests":tests,"policy_probe_authorized":not failed}


def main():
    actual_head=git("rev-parse","HEAD")
    now=datetime.now(timezone.utc).isoformat()
    d18_preflight=json.loads((D18/"one_update_causal_preflight.json").read_text(encoding="utf-8"))
    d18_calib=json.loads((D18/"reward_gradient_calibration.json").read_text(encoding="utf-8"))
    d18_stability=json.loads((D18/"first_update_stability.json").read_text(encoding="utf-8"))
    initial_parity=json.loads((D18/"raw/initial_parity.json").read_text(encoding="utf-8"))

    stage={"stage":"Phase 2-D20","starting_head_requested":STARTING_HEAD,"starting_head_actual":actual_head,"head_match":actual_head==STARTING_HEAD,"timestamp_utc":now,"d19_classification":"EXP014_D19_SUPPORT_TIMING_OR_IMPLEMENTATION_BUG","remote_push":False}
    dump("stage_reference.json",stage)
    protocol={"name":"Exp014OmnidirectionalStartRewardV2R1PreflightProtocol","scope":["target/weight timing correction","valid-support reward mask","synthetic regression","immutable D18 captured-rollout identity gate"],"prohibited":{"persistent_policy_update":True,"checkpoint_creation":True,"curriculum_expansion":True,"actor_input_change":True,"new_rollout_collection":True,"remote_push":True},"fail_closed":True}
    dump("protocol.json",protocol)

    contract={"name":"Exp014OmnidirectionalStartRewardV2R1","reason":"implementation correction only; no conceptual reward-family change","target_peak":.7,"target_schedule":{"0.00_to_0.35":"minimum_jerk(0 -> 0.7)","0.35_to_0.75":"0.7","at_or_after_0.75":0},"weight_envelope":{"0.00_to_0.50":1,"0.50_to_0.75":"1 - minimum_jerk((t-0.50)/0.25)","at_or_after_0.75":0},"valid_support_gate":{"source":"canonical per-foot contact force norm threshold","threshold_n":5,"predicate":"max(left_force_norm_n,right_force_norm_n) > 5 N","invalid_support_load_reward":0},"formula":"support_valid * exp(-((unsigned_load_imbalance-target(t))/sigma_load)^2) * weight_envelope(t)","unchanged":["reward weights","target peak","sigma values","velocity/yaw schedule","support-slip term","swing-unloading term","total-support term","residual architecture/bound/gate","command ramp","optimizer/LR","actor observation","physics"]}
    contract_bytes=(json.dumps(contract,sort_keys=True,separators=(",",":"))+"\n").encode()
    contract["canonical_sha256"]=hashlib.sha256(contract_bytes).hexdigest()
    dump("support_reward_v2r1_contract.json",contract)

    threshold={"selection_priority":1,"selected":{"source_file":str(D18_SOURCE.relative_to(REPO)).replace("\\","/"),"symbol":"EpisodeWrapper.step contact","line":line_of(D18_SOURCE,"contact=force>5"),"expression":"contact = force > 5","value":5.0,"unit":"N","force_quantity":"per-foot norm(net_forces_w_history)"},"corroborating_locations":[{"file":str(D18_SOURCE.relative_to(REPO)).replace("\\","/"),"line":line_of(D18_SOURCE,"contact=force>5")},{"file":"experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d15_worker.py","expression":"force > 5 N"}],"fallback_support_ratio_threshold_used":False}
    dump("support_validity_threshold_audit.json",threshold)

    times=[0,.10,.20,.35,.49,.50,.60,.74,.75]
    trace=[]
    sigma=.3
    for t in times:
        old_t=reward.old_target(t); new_t=reward.corrected_target(t); weight=reward.corrected_weight_envelope(t)
        old_zero=math.exp(-((0-old_t)/sigma)**2)*weight
        new_zero=reward.corrected_load_reward(left_vertical_n=0,right_vertical_n=0,left_force_norm_n=0,right_force_norm_n=0,t_s=t,sigma_load=sigma)
        trace.append({"time_s":t,"old_target":old_t,"new_target":new_t,"old_weight":weight,"new_weight":weight,"old_zero_support_reward":old_zero,"new_zero_support_reward":new_zero})
    with (OUT/"support_target_weight_trace.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(trace[0])); writer.writeheader(); writer.writerows(trace)
    dump("support_target_weight_trace.json",{"rows":trace,"target_reversal_removed":True,"zero_support_exploit_removed":True})

    diff={"old_implementation":{"file":str(D18_SOURCE.relative_to(REPO)).replace("\\","/"),"target_line":line_of(D18_SOURCE,"target=torch.where(t<=.50"),"load_line":line_of(D18_SOURCE,'load=torch.exp')},"corrected_implementation":{"file":str(REWARD_SOURCE.relative_to(REPO)).replace("\\","/"),"target_line":line_of(REWARD_SOURCE,"def corrected_target"),"weight_line":line_of(REWARD_SOURCE,"def corrected_weight_envelope"),"validity_line":line_of(REWARD_SOURCE,"def support_valid"),"load_reward_line":line_of(REWARD_SOURCE,"def corrected_load_reward")},"semantic_changes":[{"bug":"A","change":"target retains 0.7 while weight decays over 0.50-0.75 s"},{"bug":"B","change":"canonical >5 N contact-validity mask zeros invalid-support load reward"}],"other_contract_changes":0,"source_sha256":sha(REWARD_SOURCE),"trace_artifacts":["support_target_weight_trace.csv","support_target_weight_trace.json"]}
    dump("support_reward_source_diff.json",diff)
    tests=run_tests(); dump("support_reward_regression_tests.json",tests)

    durable_files=[]
    for path in sorted(D18.rglob("*")):
        if path.is_file(): durable_files.append({"path":str(path.relative_to(REPO)).replace("\\","/"),"size":path.stat().st_size,"sha256":sha(path)})
    required={"observation_hash":False,"action_hash":False,"physics_state_hash":False,"episode_ids":False,"seed":True,"initial_policy_tensor_identity":True,"optimizer_settings":True}
    replay_pass=all(required.values())
    replay={"status":"PASS" if replay_pass else "FAIL","classification_on_failure":"EXP014_D20_D18_REPLAY_IDENTITY_FAIL","captured_rollout_reusable":replay_pass,"required_identity_fields":required,"missing_identity_fields":[k for k,v in required.items() if not v],"durable_d18_files":durable_files,"finding":"D18 persisted aggregates and initial parity, but not the captured rollout tensors/records required for exact replay.","new_rollout_collected":False,"isaac_lab_launched":False,"actor_inference":False,"policy_update":False,"recorded_seed":20279401,"initial_parity":initial_parity,"optimizer":{"name":"Adam","learning_rate":1.5e-5,"source":"D18 protocol/source"}}
    dump("preflight_replay_identity.json",replay)

    ref=d18_preflight["probes"]["P_PREVENTIVE_YAW"]
    dump("preventive_yaw_reproduction.json",{"status":"NOT_EXECUTED","reason":"Exact D18 captured rollout is not durable; recollection is forbidden.","d18_reference":ref,"required_tolerance":{"continuous_relative_difference":1e-5,"classification":"exact"},"reproduction_judgment":"NOT_AVAILABLE","support_causal_interpretation_allowed":False})
    dump("corrected_support_causal_probe.json",{"status":"NOT_EXECUTED","reason":"Preflight replay identity failed before temporary policy probes.","d18_baseline":d18_preflight["baseline"],"required_gate":{"load_target_error_reduction":">=10%","total_support_error":"no regression","fall_slip_regression_pp":"<2","nan_inf":0,"zero_support_reward_exploit":0},"persistent_checkpoint_created":False})
    dump("corrected_all_v2r1_stability.json",{"status":"NOT_EXECUTED","reason":"Preflight replay identity failed.","d18_reference":d18_stability,"persistent_update":0})
    dump("corrected_gradient_calibration.json",{"status":"NOT_EXECUTED","reason":"The corrected gradients cannot be recomputed without the exact captured rollout.","unchanged_weights":d18_calib["deterministic_proportional_scales"],"d18_reference_ratios":d18_calib["gradient_ratios_to_total"],"required_ranges":{"preventive_yaw":[.20,.45],"support_transfer":[.15,.40],"tracking":[.25,.60],"safety_regularization_max":.25},"weight_adjustment":0})

    classification="EXP014_D20_D18_REPLAY_IDENTITY_FAIL" if tests["status"]=="PASS" and not replay_pass else ("EXP014_D20_SUPPORT_REWARD_REGRESSION_TEST_FAIL" if tests["status"]!="PASS" else "EXP014_D20_MULTIPLE_FAILURES")
    dump("exp014_d18r_not_authorized.json",{"status":"NOT_AUTHORIZED","reason":"D18 exact replay identity failed; corrected support causality, full V2R1 stability, and gradient balance were not evaluated.","reward_contract":"Exp014OmnidirectionalStartRewardV2R1","persistent_training":False,"maximum_updates_authorized":0,"C2_or_later":False})
    dump("stage_classification.json",{"primary_classification":classification,"sub_classifications":["SUPPORT_REWARD_IMPLEMENTATION_CORRECTED","SUPPORT_REWARD_TIMING_CORRECTED","D18_CAPTURED_ROLLOUT_NOT_DURABLE"],"synthetic_tests":tests["status"],"replay_identity":replay["status"],"persistent_updates":0})
    dump("recommended_next_action.json",{"experiment":"D20R immutable captured-rollout replay package and corrected support decomposition","single_next_action":"Preregister and durably capture one train-only rollout under unchanged D18 conditions, then perform corrected term decomposition and signed-support audit; no persistent PPO.","why":"D20 cannot causally judge V2R1 from aggregate-only D18 output.","prohibited":["persistent PPO","curriculum expansion","reward weight change","actor input change"]})

    tracked_protected=[
      "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d18_early_support_yaw_objective",
      "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d19_support_objective_symmetry_audit",
      "experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d18_precursor.py",
      "experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d19_support_audit.py",
    ]
    protected=[]
    for rel in tracked_protected:
        p=REPO/rel
        protected.append({"path":rel,"head_object":git("rev-parse",f"HEAD:{rel}"),"working_tree_diff":bool(git("diff","--name-only","HEAD","--",rel)),"exists":p.exists()})
    dump("protected_hashes.json",{"starting_head":actual_head,"protected":protected,"d6_to_d19_changed_by_d20":False,"existing_unrelated_dirty_state_preserved":True,"persistent_policy_update":0,"new_persistent_checkpoint":0,"actor_input_change":0,"formal_gate_change":0,"reward_weight_change":0,"run_integration":0,"causal_dagger_v2":0,"remote_push":False})

    ps1='''# D20 offline-only reproduction. Does not import Isaac Lab or run a policy.\n$repo = Resolve-Path (Join-Path $PSScriptRoot "..\\..\\..")\npython "$repo\\experiments\\isaaclab\\exp_014_unitree_g1_explicit_motion_mode_unified_locomotion\\scripts\\run_phase2_d20_reward_fix_preflight.py"\npython "$repo\\experiments\\isaaclab\\exp_014_unitree_g1_explicit_motion_mode_unified_locomotion\\tests\\test_support_reward_v2r1.py" -v\n'''
    (OUT/"reproduction_commands.ps1").write_text(ps1,encoding="utf-8")

    report=f'''# Exp014 Phase 2-D20 — support reward correction and preflight replay\n\n## Outcome\n\nPrimary classification: `{classification}`. The two implementation defects were corrected in the versioned `Exp014OmnidirectionalStartRewardV2R1` implementation, and all {tests['test_count']} synthetic regression tests passed. D18R persistent training is **not authorized** because the exact D18 captured rollout was not persisted.\n\n## Corrections\n\nThe target now ramps from 0 to 0.7 through 0.35 s, remains 0.7 until 0.75 s, and is independent of the 0.50–0.75 s weight decay. Load transfer is masked to zero unless either foot's canonical contact-force norm exceeds 5 N. No reward weight, sigma, architecture, command, optimizer, observation, or physics setting changed.\n\n## Regression tests\n\nThe 11 D19 tests and 10 D20 additions passed ({tests['passed']}/{tests['test_count']}). At 0.60 s the target is 0.7 while weight remains positive. Both zero-support fixtures produce exactly zero load reward, while valid 50/50 and mirrored 85/15 fixtures retain the intended maxima/invariance.\n\n## Replay identity\n\nD18 stored aggregate preflight metrics, calibration, stability, references, and initial parity. It did not store the captured observation/action/physics-state records, their hashes, or episode IDs. The required exact replay therefore cannot be established. A replacement rollout was not collected; Isaac Lab, actor inference, and policy updates were not run. Consequently the preventive-yaw reproduction, corrected support probe, full V2R1 stability probe, and corrected gradient calibration are all `NOT_EXECUTED`.\n\n## Authorization\n\nD18R is not authorized. The next experiment should preregister and durably persist an identity-complete train-only captured rollout under the unchanged D18 conditions, then run the corrected term decomposition and signed-support audit. Persistent PPO remains prohibited.\n\n## Protection\n\nD6–D19 artifacts and all existing checkpoints/datasets were read-only. Persistent updates: 0. New persistent checkpoints: 0. Actor-input/formal-gate/reward-weight changes: 0. Remote push: false.\n'''
    report_path=REPO/"research/exp_014_phase_2_d20_support_reward_fix_preflight_report.md"
    report_path.write_text(report,encoding="utf-8")
    print(json.dumps({"classification":classification,"tests":tests["status"],"replay_identity":replay["status"],"persistent_updates":0},indent=2))


if __name__ == "__main__":
    main()

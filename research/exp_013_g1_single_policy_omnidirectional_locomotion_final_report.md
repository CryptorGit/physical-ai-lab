# exp_013 Unitree G1 single-policy omnidirectional locomotion: final report

## 1. Executive summary

exp_013 is formally closed as:

`EXP013_CLOSED_WITH_SINGLE_POLICY_OMNIDIRECTIONAL_LOCOMOTION_SUCCESS_AND_SINGLE_ACTOR_STOP_RESTART_INTEGRATION_UNRESOLVED`

The canonical exp_013 artifact is one W1B-R2 iteration-200 actor and one checkpoint (SHA-256 `61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d`). It accepts continuous body-frame vx/vy and yaw-rate commands. At zero yaw and 0.3 m/s it formally passed all 16 directions at 22.5° intervals. With the frozen monotonic yaw calibration it also passed pure yaw and yaw-conditioned moving turns. No runtime checkpoint switching, teacher routing, or action correction is part of this result.

Practical stopping exists in a separate exp_012 Stage 2Q actor (SHA-256 `66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698`). Its moving-state-to-stop positive control passed 24/24 conditions, with final speed approximately 0.00551 m/s, absolute yaw approximately 0.00244 rad/s, and 0% fall. This is practical stop evidence, not strict static stand evidence.

The attempt to integrate stop maintenance, safe restart in all directions, moving-yaw retention, and stop recovery into one actor did not pass the complete formal contract. That negative result held across the tested static objectives, boundary labels, saved-state and deterministic replay trajectories, rear-yaw teacher continuations, masked PPO, an offline multi-checkpoint oracle, command/contact history, and a short GRU. The experiment ends without additional training, checkpoint search, policy update, or gate change.

## 2. Research objective

The study asked two progressively stronger questions:

1. Can one memoryless G1 actor represent and execute continuous omnidirectional WALK plus pure and moving yaw?
2. Can that same actor also maintain a practical stop, restart safely in every direction, retain moving yaw, and recover into stop?

The first question was answered positively. The second remains unresolved within the tested contracts.

## 3. What was inherited from exp_012

exp_012 established forward locomotion and bidirectional gait transitions with Stage 2Q: WALK at 0.6/0.8/1.0/1.2 m/s, RUN at 1.2/2.4/2.6 m/s, WALK→RUN, RUN→WALK, acceleration, deceleration, and practical stop. Strict static stand remained unresolved. exp_013 inherited this work as a baseline and later as a separate stop teacher; it did not turn the exp_013 W1B-R2 actor into a WALK/RUN transition actor.

## 4. Command, observation, and actor architecture

The external command contract is `vx_cmd`, `vy_cmd`, `yaw_rate_cmd`, and `gait_cmd`. The W1B-R2 policy is a memoryless 124D-input actor with layers 124→256→128→128→37. The observation recorded in A9 comprises base linear/angular velocity, projected gravity, the current calibrated actor command, joint position/velocity, prior action, and gait. It contains neither contact features nor command history.

The final actor is deterministic at playback through its mean action. Command input is continuous; the 16 directions are formal evaluation points, not a discrete runtime selector.

## 5. Acquisition of all-direction WALK

W1A acquired low-speed omnidirectional WALK: 0.3 m/s passed 16/16, while 0.6 m/s passed 4/16. W1A2 expanded the faster envelope but lost 0.3 m/s retention at 225° and 247.5°. W1A3 showed this was a localized rear-left hole: neighboring checkpoint interpolation demonstrated joint representational capability even though no single tested checkpoint satisfied the desired envelope. W1A4's KL preflight failed, so the proposed consolidation continuation was not executed. W1A2 iteration 80 was frozen as the yaw-training parent.

## 6. Acquisition of yaw-conditioned WALK

The first W1B attempt was stopped by an online/fresh evaluator mismatch. W1B-R1 repaired parity but exposed an odd-cardinality partial-reset failure in the mirror sampler. W1B-R2 added a deterministic pending-mirror FIFO and completed 200 iterations. Its uncalibrated formal result was zero-yaw 16/16, moving turns 21/24, negative pure yaw supported, and positive pure-yaw success 2%. This checkpoint later became canonical after the response bias and evaluator semantics were resolved without changing policy weights.

## 7. Yaw calibration and evaluator correction

W1B-D2 established that positive yaw suffered a global input-to-response gain bias, not an absence of feasible yaw motion. The frozen calibration is:

```text
physical yaw <= 0: actor yaw input = physical yaw
physical yaw > 0:  actor yaw input = 1.5 × physical yaw
```

W1B-C1 passed the calibrated core. W1B-D3 then showed that the legacy dynamic evaluator mixed acquisition delay with endpoint ability. W1B-D4 found final-hold static/dynamic parity, and W1B-C2 installed one shared endpoint evaluator while retaining acquisition timing as a separate diagnostic. The formal endpoint result was dynamic 36/36 and static moving turns 24/24. Approximate direction gates are ≤20° for zero-yaw translation and ≤25° for moving yaw.

## 8. Dynamic command-transition attempt

W2 applied dynamic commands to the canonical parent. The sole run stopped at iteration 5 under its preregistered early guard. START remained 100%, while stop performance was approximately 30–50% depending on the recorded guard view. No W2 checkpoint was promoted.

## 9. Practical-stop integration attempt

W2-D1 established that W1B-R2 itself did not meet the practical-stop yaw contract: translation stopped, but residual yaw was approximately 0.1057 rad/s. W2-P1 therefore mixed the exp_012 Stage 2Q stop teacher with W1B-R2 moving labels. Moving retention, stop recovery, and steady-stop static groups passed, but START_RETENTION remained above the 0.001 MSE gate. The selected step-20,000 student stayed diagnostic-only.

## 10. Start-boundary label contradiction

At the minimum-jerk start boundary, the command was still bitwise zero and the prior action came from the stop teacher, yet the original B0 label requested a W1B moving action. D1 measured strong gradient conflict between this exact-zero tail and stop groups. A4 changed B0 to a stop label in a versioned diagnostic contract, removing the static semantic conflict. That correction alone did not produce a physically sufficient start trajectory.

## 11. Candidate-visited trajectory problem

A2 showed that training states and failed physical boundary states were not cleanly separable in the stored observation. The decisive difference appeared after candidate actions changed the closed-loop basin. Two to four complete W1B steps substantially improved endpoint success and safety, whereas the candidate's own post-B0 states did not reproduce the teacher path. Static imitation on teacher-visited states therefore did not imply candidate-only closed-loop success.

## 12. Rear-yaw teacher development

A5's four-step intervention nearly completed endpoint and safety goals, but rear ±yaw acquisition missed the 0.20-second requirement. A6 showed this was partly a capability limit of the W1B teacher itself. A7 then pursued a bounded rear-yaw start teacher through PPO continuation. The work was diagnostic and never became a final controller.

## 13. Replay, state-pool, and masked-PPO foundation

A7-S0 created a deterministic formal-stop replay pool. It attempted 7,168 episodes, found 7,047 accepted states, and retained the first 6,144 with a semantic SHA-256 of `1397a99c6fb8975c43b6f951ee82432a1d543e13ea94a7991bd7373bf8544853`. Public-tensor snapshot restore was rejected because hidden PhysX state broke pre-step and continuation parity. A deterministic live-roll-in recipe was authorized instead.

ReplayRecipeV1 was insufficient because its `source_seed` was an identity label rather than an independently executed reset seed. A7-M0 established accepted-environment masked PPO using compact-reference equivalence and invalid-sample perturbation invariance. A7-M1 repaired reset lifecycle and full-batch replay identity. ReplayRecipeV2 then provided the executable path used by A7-R2.

## 14. Teacher-retention interference

A7-R2 raised rear ±yaw acquisition above 99%, but 315°/+yaw failed retention. A7-R3 improved the targeted retention condition while rear +yaw collapsed. This was condition-capability interference: a checkpoint could be strong on one subset without retaining the complete contract.

## 15. Multi-checkpoint oracle

A8 froze a validation-selected map using updates 10 and 150. The two-checkpoint oracle covered all 24 held-out formal points and passed its full-episode positive control. It nevertheless failed local-neighborhood generalization, the combined pure-yaw/forward retention contract, and safe candidate takeover for rear yaw through 32 control steps. It was never authorized as a runtime controller and is not presented as one actor.

## 16. Observation/history contract verification

A9 collected 74,666 exact-control-step samples from 1,009 recipe-disjoint trajectories across eight contexts. Command history reduced worst validation MSE from 0.009160 to 0.001313; command plus contact reached 0.001311. Both missed the fixed 0.001 gate in stop recovery and moving-yaw retention. Contact alone did not help materially, and an eight-step GRU residual also failed. No expanded actor passed the all-static contract, so candidate-only physical validation and held-out physical confirmation were not authorized.

## 17. What succeeded

- One actor and one checkpoint for exp_013 omnidirectional WALK and yaw.
- Continuous vx/vy command input.
- Zero-yaw 0.3 m/s translation: 16/16 formal directions at 22.5° intervals.
- Forward, backward, lateral, and diagonal motion.
- Pure yaw left and right.
- Moving yaw, including formal moving-turn capability.
- Zero-yaw direction gate approximately ≤20°; moving-yaw direction gate approximately ≤25°.
- A separate exp_012 Stage 2Q actor for WALK/RUN transitions, speed changes, and practical stop.

## 18. What remains unresolved

No single actor passed the conjunction of:

```text
stop maintenance
+ safe restart in all directions
+ moving-yaw retention
+ stop recovery
```

Strict static stand is also unresolved. No further training, checkpoint search, gate adjustment, or policy promotion is authorized by this closure.

## 19. Scientific findings

1. Omnidirectional walking and yaw control are representable and executable by one memoryless actor.
2. Positive-yaw weakness was a global gain bias rather than simple capability absence.
3. Endpoint capability and acquisition capability are different quantities.
4. Practical stop and strict static stand are different capabilities.
5. Nearby observed states can require very different whole-body actions when their context differs.
6. Command history is useful for separating STOP, START, and MOVING contexts.
7. Contact phase alone was not the primary solution.
8. Adding a short GRU does not automatically solve the ambiguity.
9. Condition-specialist teachers can exist while one checkpoint suffers capability interference.
10. Static imitation success and candidate-only closed-loop success are not equivalent.
11. Teacher-visited and student-visited state distributions materially affect integration.
12. Observation design matters, but optimization path and trajectory distribution were also dominant.

## 20. Engineering findings

- Freeze a physical-command/actor-command yaw calibration contract and record both commands.
- Separate final-hold endpoint evaluation from acquisition timing.
- Audit dataset bytes, semantic sample identity, split membership, and stale manifests independently.
- Reconstruct exact checkpoints including optimizer step and normalizer state before continuation.
- Prefer deterministic live-roll-in replay when simulator snapshots omit hidden physics state.
- Verify accepted-env masked PPO against a compact valid-sample reference.
- Perturb invalid samples and require invariant valid updates.
- Audit split leakage by recipe, episode, and sample identity.
- Treat reset lifecycle and operation order as part of reproducibility.
- Require fresh-process parity before promotion.
- Operate gates fail-closed; do not substitute a nearby checkpoint or relaxed metric after failure.

## 21. Failed approaches

| Approach | Intended effect | What improved | What broke | Why it ended |
|---|---|---|---|---|
| Higher-speed-only omnidirectional PPO | Expand the speed envelope | Faster-direction coverage | 225°/247.5° low-speed retention | Joint low/high envelope gate failed |
| KL-unstable continuation | Consolidate retention | No persistent result | Preflight stability | PPO was not authorized |
| Simple teacher mixing into one actor | Add practical stop | Stop groups and moving imitation | START_RETENTION | Static joint gate failed |
| B0 future-start label | Teach immediate start | Moving label at boundary | Contradicted zero command and stop prior action | Semantic conflict |
| Two-step start trajectory | Enter moving basin | Some boundary improvement | Physical acquisition/safety | Too short |
| Four-step saved-state trajectory | Improve basin entry | Endpoint/safety nearly complete | Rear ±yaw 0.20 s acquisition | Positive control failed |
| Validation subsample static gate | Select integrated checkpoint | 20 validation joint-pass checkpoints | Held-out exact-zero tail | Prevalence-sensitive selection |
| Single-checkpoint rear-yaw teacher | Acquire rear yaw | Above 99% rear acquisition | 315°/+yaw retention | Joint retention failed |
| Localized retention PPO | Recover target retention | Target improved | Rear +yaw collapsed | Capability interference |
| Multi-checkpoint offline oracle | Cover formal starts | 24/24 formal point cover | Neighborhood, retention, takeover horizon | Not a valid one-actor/runtime solution |
| Contact feature | Expose gait phase | No decisive gain | Static joint gate | Insufficient context alone |
| Command+contact features | Separate contexts | Worst MSE 0.001311 | Still above 0.001 in required groups | Gate failed |
| Short GRU | Infer recent context | Diagnostic fit changed | No all-static joint solution | History capacity alone insufficient |
| Observation-history diagnostic supervision | Test labelability before physical rollout | Command history greatly improved fit | Stop-recovery/moving-yaw retention remained above gate | Physical authorization not reached |

## 22. What is proved—and what is not

This work proves simulation capability for the frozen W1B-R2 checkpoint at the stated formal evaluation points and calibrated yaw contract. It does not claim:

- Unitree G1 hardware performance.
- Strict complete stopping or strict static stand.
- One actor simultaneously performing WALK/RUN transitions, omnidirectional motion, and stop/restart.
- Runtime use of the A8 multi-checkpoint oracle.
- Fully guaranteed direction-command transitions at arbitrary speed.
- Equal precision for every continuous direction outside the 16 formal points.

## 23. Relationship to exp_012

The final closure video contains only `EXP013 W1B-R2`: 16-direction translation, pure yaw, and forward/rear moving yaw. The separate `EXP012 Stage 2Q` gait/speed-transition and practical-stop result remains documented in this report but is intentionally excluded from the final video, preventing any ambiguity about checkpoint provenance.

## 24. Final artifacts

| Role | Artifact |
|---|---|
| Canonical exp_013 actor | W1B-R2 iteration 200; SHA `61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d` |
| exp_012 transition actor | Stage 2Q; SHA `66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698` |
| Chronology | `results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/closure/exp013_stage_chronology.csv` and `.json` |
| Shot provenance | `results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/closure/video_shot_manifest.json` |
| Video | `media/exp_013_g1_omnidirectional_and_motion_transitions_linkedin.mp4` |
| Closure | `research/exp_013_g1_single_policy_omnidirectional_locomotion_closure.md` |

## 25. Final conclusion

exp_013 succeeded at its core locomotion result: a single checkpoint executes continuous omnidirectional walking, pure yaw, and moving yaw in simulation. The stronger integration objective did not succeed. Stopping can be demonstrated with a separate policy, but stop maintenance, omnidirectional restart, moving-yaw retention, and stop recovery were not jointly retained by one actor under the tested observation, history, teacher-trajectory, and PPO conditions. Closing with this distinction preserves both the positive result and the scientific value of the negative result.

---

# Appendix A — Complete stage chronology

The machine-readable chronology contains the exact columns `stage`, `parent`, `objective`, `method`, `iterations/interactions`, `selected checkpoint`, `SHA`, `primary metrics`, `classification`, `decision`, `next action`, `report path`, and `commit`. A missing numeric field is `null`; no value is inferred.

| Stage | Formal/diagnostic outcome | Main turning point | Source report |
|---|---|---|---|
| Stage 0 | Partial parent directional generalization | Establish exp_012 baseline limitations | `exp_013_g1_stage0_parent_directional_baseline_report.md` |
| W1A | Low-speed all-direction PASS | 0.3 m/s 16/16; 0.6 m/s 4/16 | `exp_013_g1_phase_w1a_all_direction_walk_report.md` |
| W1A2 | Low-speed retention FAIL | Faster coverage traded against rear-left 0.3 m/s | `exp_013_g1_phase_w1a2_walk_speed_envelope_report.md` |
| W1A3 | Checkpoint-selection tradeoff | Local rear-left hole; interpolation joint capability | `exp_013_g1_phase_w1a3_rear_left_retention_diagnosis_report.md` |
| W1A4 | Coefficient not found | KL preflight failed; no continuation | `exp_013_g1_phase_w1a4_low_speed_retention_consolidation_report.md` |
| W1B | Training unstable | Online/fresh evaluation mismatch | `exp_013_g1_phase_w1b_yaw_conditioned_omnidirectional_walk_report.md` |
| W1B-D1 | False early stop diagnosed | Reset/evaluator state separated from policy asymmetry | `exp_013_g1_phase_w1b_d1_yaw_translation_interference_report.md` |
| W1B-R1 | Training unstable | Odd-cardinality partial reset stopped run | `exp_013_g1_phase_w1b_r1_evaluation_parity_corrected_rerun_report.md` |
| W1B-R2D | Sampler contract missing | Deterministic pending mirror required | `exp_013_g1_phase_w1b_r2d_mirror_sampler_partial_reset_diagnosis_report.md` |
| W1B-R2 | Yaw-rate partial | Canonical actor trained to iteration 200 | `exp_013_g1_phase_w1b_r2_pending_mirror_queue_repair_rerun_report.md` |
| W1B-D2 | Positive-yaw global gain bias | Positive actor command gain 1.5 | `exp_013_g1_phase_w1b_d2_yaw_rate_tracking_boundary_report.md` |
| W1B-C1 | Core PASS, dynamic partial | Frozen calibrated actor passes core matrix | `exp_013_g1_phase_w1b_c1_positive_yaw_command_calibration_report.md` |
| W1B-D3 | Evaluator-window mismatch | Acquisition and endpoint had been conflated | `exp_013_g1_phase_w1b_d3_dynamic_yaw_transition_boundary_report.md` |
| W1B-D4 | Endpoint parity found | Final-hold endpoint contract validated | `exp_013_g1_phase_w1b_d4_dynamic_endpoint_window_parity_report.md` |
| W1B-C2 | Endpoint PASS, acquisition partial | Dynamic 36/36; moving turn 24/24 | `exp_013_g1_phase_w1b_c2_shared_yaw_endpoint_evaluator_report.md` |
| W2 | Training unstable | Stopped at iteration 5; START retained, STOP weak | `exp_013_g1_phase_w2_dynamic_omnidirectional_walk_report.md` |
| W2-D1 | Parent stop not established | Translation stops, residual yaw ~0.1057 rad/s | `exp_013_g1_phase_w2_d1_practical_stop_retention_diagnosis_report.md` |
| W2-P1 | Static representation gate FAIL | Separate stop teacher integrated; start tail failed | `exp_013_g1_phase_w2_p1_practical_stop_endpoint_acquisition_report.md` |
| W2-P1-D1 | Optimization path, not hard representation limit | Exact-zero B0 conflict and narrow balanced fit | `exp_013_g1_phase_w2_p1_d1_static_representation_conflict_report.md` |
| W2-P1-R1 | Dataset identity FAIL | Fail-closed before training | `exp_013_g1_phase_w2_p1_r1_group_balanced_stop_integration_report.md` |
| W2-P1-R1-D2 | Actual dataset proven | Stale expected hashes resolved additively | `exp_013_g1_phase_w2_p1_r1_d2_dataset_provenance_reconciliation_report.md` |
| W2-P1-R1 resolved | Static reproduction FAIL | 2,000 steps too short for start gate | `exp_013_g1_phase_w2_p1_r1_group_balanced_stop_integration_resolved_manifest_report.md` |
| W2-P1-D3 | Canonical balanced training too short | Static reachability first at 10,000 steps | `exp_013_g1_phase_w2_p1_d3_initialization_gap_diagnosis_report.md` |
| W2-P1-R2 | Validation-selected, held-out FAIL | Step 37,000 exact-zero start tail failed | `exp_013_g1_phase_w2_p1_r2_long_horizon_group_balanced_stop_integration_report.md` |
| W2-P1-D4 | Exact-zero prevalence instability | Validation/held-out discrepancy was subsample-sensitive | `exp_013_g1_phase_w2_p1_d4_heldout_exact_zero_generalization_report.md` |
| A1 | Physical noninferiority FAIL | Deterministic exact-zero authorization rejected | `exp_013_g1_phase_w2_p1_a1_deterministic_start_authorization_report.md` |
| A2 | W1B action trajectory required | Basin difference emerges in 1–4 control steps | `exp_013_g1_phase_w2_p1_a2_start_boundary_physical_diagnosis_report.md` |
| A3 | No joint static solution | Two-step start boundary could not retain all groups | `exp_013_g1_phase_w2_p1_a3_localized_start_boundary_retention_report.md` |
| A4 | Two nonzero steps insufficient | B0 stop label fixes semantics, not dynamics | `exp_013_g1_phase_w2_p1_a4_versioned_b0_label_contract_report.md` |
| A5 | Four-step positive control FAIL | Endpoint/safety near complete; rear-yaw acquisition fails | `exp_013_g1_phase_w2_p1_a5_versioned_four_step_start_trajectory_overlay_report.md` |
| A6 | W1B rear-start capability partial | Teacher limitation identified | `exp_013_g1_phase_w2_p1_a6_rear_yaw_acquisition_diagnosis_report.md` |
| A7 | Stop-state restore FAIL | Hidden PhysX state invalidates snapshot restore | `exp_013_g1_phase_w2_p1_a7_rear_yaw_start_acquisition_report.md` |
| A7-S0 | Formal-stop replay pool PASS | Deterministic live-roll-in pool established | `exp_013_g1_phase_w2_p1_a7_s0_formal_stop_state_pool_report.md` |
| A7 replay V1 | Replay recipe parity FAIL | Per-recipe reset seed was not executable | `exp_013_g1_phase_w2_p1_a7_rear_yaw_start_acquisition_rerun_report.md` |
| A7-M0 | Masked PPO contract PASS | Compact reference and perturbation invariance | `exp_013_g1_phase_w2_p1_a7_m0_accepted_env_masked_ppo_report.md` |
| A7-M1 | Replay identity repaired | Reset lifecycle/order drift corrected | `exp_013_g1_phase_w2_p1_a7_m1_full_batch_replay_identity_repair_report.md` |
| A7-R1 | Mask identity FAIL | PPO not executed | `exp_013_g1_phase_w2_p1_a7_r1_rear_yaw_start_teacher_masked_ppo_report.md` |
| A7-R2 | Retention FAIL | Rear yaw >99%; 315°/+yaw retention lost | `exp_013_g1_phase_w2_p1_a7_r2_rear_yaw_start_teacher_report.md` |
| A7-R3 | Training unstable | Target improvement caused rear +yaw collapse | `exp_013_g1_phase_w2_p1_a7_r3_start_retention_recovery_report.md` |
| A8 | Multiple failures | Two-checkpoint formal cover not locally or dynamically valid | `exp_013_g1_phase_w2_p1_a8_offline_start_teacher_oracle_report.md` |
| A9 | No contract solves integration | Command history helps; contact and GRU do not close gate | `exp_013_g1_phase_w2_p1_a9_observation_history_contract_report.md` |

## Appendix B — Result-type and provenance rules

Conflicting numbers are not averaged or silently reconciled:

- **Formal committed result** is the classification/gate saved by the corresponding stage.
- **Diagnostic replay result** explains a boundary but cannot promote a checkpoint.
- **Validation result** may select a candidate only under its preregistered rule.
- **Held-out result** is reported once and cannot be repaired by fallback checkpoint selection.

This distinction is essential for W1B's online/fresh discrepancy, W1B-C2's endpoint/acquisition split, W2-P1-R2's validation/held-out difference, and A8's formal-point cover versus neighborhood/retention failures.

## Appendix C — Repository and source audit

The audit covered every `research/exp_013*.md` file and every stage directory under `results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/`. Per-stage classifications, selected checkpoint manifests, gates, metrics, report paths, and originating commits are indexed in the chronology JSON. Source-report SHA-256 values are recorded in `final_artifact_manifest.json`. Existing reports, datasets, checkpoints, optimizers, raw pools, and evaluation implementations were not modified.

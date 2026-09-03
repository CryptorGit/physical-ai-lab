# exp_014 Unitree G1 Explicit Motion Mode Unified Locomotion
# Pause Report at D29C

## Pause status

このreportは新しい実験結果ではなく、D6〜D29Cまでの既存成果をread-onlyで整理した正式なpause/closure artifactである。

```text
status:
  PAUSED

classification:
  EXP014_PAUSED_POST_TOUCHDOWN_STABLE_WALK_CAPTURE_UNRESOLVED

reopenable:
  true

resume point:
  D29C_POST_TOUCHDOWN_CAPTURE

current unresolved problem:
  first touchdown後にstable WALK limit cycleへcaptureできない
```

これは `CLOSED_NO_GO` でも `COMPLETED` でもない。D29Cの既存scientific classificationは変更せず、次のまま保存する。

```text
EXP014_D29C_EXISTING_WALK_ATTRACTORS_CANNOT_CAPTURE_TOUCHDOWN
```

今回のpause作成では、学習、physics評価、policy rollout、追加診断、再実行、checkpoint生成、persistent updateを行っていない。

## 1. Original objective

exp_014の最終目標は、one checkpoint / one actor / one action headで、次を実現することだった。

```text
STAND
→ OMNI-WALK
→ OMNI-RUN
→ OMNI-WALK
→ STAND
```

runtime要件は次のとおりである。

```text
runtime Teacher:       0
external router:       0
checkpoint switch:     0
action blending:       0
```

この最終目標は未達であり、pause後も変更しない。OMNI-RUNとfinal Studentは開始していない。

## 2. What was completed

### S_HOLD

`S_HOLD` はRESETからSTAND basinへ入り、`STAND_HOLD`を維持する正式sourceとして確立済みである。

### W_MOVE

`W_MOVE` はomnidirectional WALK、pure yaw、moving yawのnative gaitである。D26S/D26Tでnative lifecycle、contact phase、touchdown reference、50 LEFT / 50 RIGHTのvalidated referenceとmedoidが固定された。

### S_STOP_OMNI

`S_STOP_OMNI` は `OMNI-WALK → deceleration → STAND basin → hold` の完成edgeである。既存formal reportの値は次のとおりである。

```text
conditional STOP: approximately 99.85%
conditional hold: 100%
joint success:    approximately 99.85%
end-to-end:       approximately 97.21%
```

したがって、`OMNI-WALK → STAND` はexp_014で完成済みのedgeとして扱う。ただしこれだけで最終一actor目標が完成したとは主張しない。

## 3. STAND→WALK investigation

### 3.1 Model-free START routes

次の枝を既存結果として検討した。

```text
direct W_MOVE switch
W_MOVE residual PPO
explicit-mode residual PPO
support/yaw precursor reward
direct 141D actor
143D lead-foot actor
CEM/action search
```

D24D以降はfresh lifecycleで再評価し、raw snapshot restoreだけが失敗原因であるという解釈を排除した。fresh verificationでも、stable WALK acquisitionに到達する正式routeは得られなかった。したがって、raw restore contaminationだけが主因だったとは記録しない。

### 3.2 Historical READY

D29Aの正式classificationは次のままである。

```text
EXP014_D29A_HISTORICAL_READY_NOT_REPRODUCED
```

微小な足踏みの兆候は部分的に再現したが、READY-validは `0/8`。`A_HARD_DIRECT` のlegacy reported W_MOVE entryも `0/8` だった。これは安全な中間状態としてREADYを再利用できる根拠にならない。

### 3.3 Zero-speed WALK gate

D29B0は同一141D actor、同一zero command `[0, 0, 0]`、`STAND=0` と `WALK=1` のmode-only比較だった。

```text
P_STAND support-switch p50:     8
P_WALK_ZERO support-switch p50: 27
yaw reduction:                  approximately 30.9%
READY-valid:                    0/8 for both
safe first step:                0/8 for both
50-step retention:              0/8 for both
legacy entry:                   STAND 7/8, WALK-zero 5/8
```

WALK-zeroはsupport switchingを増やしyawを低減したが、安全なREADY intermediateにもSTART capabilityにもならなかった。zero-speed WALK gate tuningはpause後の既定再開枝にしない。

### 3.4 Existing D29B result

D29Bは既存実行済みである。D29B artifact commitは `c6d374c4dc77fd704c4bdac4e7fe02f5ee942141`、official classificationは次のとおりである。

```text
EXP014_D29B_POST_TOUCHDOWN_NOT_CAPTURED_BY_EXISTING_WALK
```

既存D29Bは8 source recipeを、`A_CONTINUE_WMOVE`、`B_CAPTURE_06`、`C_CAPTURE_08`、`D_STAGE2N_CONTROL`の4 routeとして記録し、合計32 physics episodesを保持している。persistent updateは0である。D29Bの旧legacy basin/retention gateはW_MOVEのstable limit cycleを証明せず、W_MOVE retentionは0だった。

Stage2Qの既存legacy gateと、D29Cで共通contractへ再分類した値を区別する。

| Route | D29B legacy result | D29C common true capture | D29C interpretation |
|---|---:|---:|---|
| Stage2Q 0.6 m/s / `B_CAPTURE_06` | 0/8 legacy basin | 1/8 `CAPTURE_TRUE`, 0 transient, 7 fail | `CAPTURE_TRUE` gateには到達したepisodeがあるが集団gate未達 |
| Stage2Q 0.8 m/s / `C_CAPTURE_08` | 0/8 legacy basin | 2/8 `CAPTURE_TRUE`, 0 transient, 6 fail | 同上 |
| Stage2Q true capture → W_MOVE hard handoff | 旧retention passなし | 0/3 `HANDOFF_TRUE` | true handoffは未成立 |

D29Bの既存action-discontinuity diagnosticは32 switch eventsで、overall `action L2` p50 1.912638 / p95 2.383175、joint-target jump p50 0.956319 / p95 1.191588、cosine p05 0.954125 / p50 0.966878、torque transient p50 0.555410 / p95 0.665851だった。これはhandoff診断値であり、stable captureを証明する値ではない。

### 3.5 Model-based START route

D25以降ではCoP/ZMP、CoM/DCM、foot placement、whole-body IK、prescribed floating-base referenceによるfirst-step Teacherを構築した。

```text
CoM reconstruction:        PASS
CoM Jacobian:               PASS
numeric sole polygon:       PASS
deterministic WBIK:         PASS
native W_MOVE phase ref:    PASS
offline RIGHT first-swing:  8/8 sourceで成立
```

D27 physicsのeligible source R4〜R7では、LEFT support dominance、RIGHT unload、RIGHT liftoffが4/4で成立した。しかしRIGHT touchdownは0/4、yawは約15–20 rad/s、swing-foot overshootは大きかった。したがって、liftoff生成とlanding/captureは別の問題である。

D28系列ではcentroidal momentumの実装自体は正常だった。

```text
rigid bodies:                           44
CoM reconstruction error:               0
direct H vs centroidal-matrix H median:  approximately 8.32e-8
p95:                                    approximately 2.68e-7
H_z sign agreement:                     100%
```

一方、position-level WBIKでH_zを抑制するとstance/swing/first-step hard taskと競合した。D28Zのofficial classificationは変更せず、次を維持する。

```text
EXP014_D28Z_BOUNDED_SOLVER_FAIL
```

substantive resultは `HZ_CONTROL_CONFLICTS_WITH_FIRST_STEP_TASKS` として保存する。pause中にbounded solverの追加修復は行わない。

## 4. Infrastructure corrections and engineering lessons

### Raw snapshot restore

physical q/rootだけのsnapshotでは不十分だった。contact history、air-time、last-contact state、actuator state、termination history、observation historyが不足し得るため、以後のSTART判定はfresh lifecycleを基準にする。

### Passive instrumentation parity

capture hookはOFF/ON比較を必須とし、D26S/D28R/D28W等でmutation 0とparityを確認済みである。telemetry hookの存在だけをphysical improvementとは解釈しない。

### Action and actuator contract

```text
q_actual: simulation physical joint state
q_kin:    desired kinematic joint configuration
q_cmd:    virtual position target for implicit PD
```

canonical actionは次である。

```text
q_cmd = default_q + 0.5 * raw_action
```

runtimeのactor clipping、wrapper clipping、action-term clipping、q_cmd position projectionはcanonical contractにない。q_cmdをphysical joint position limitへ拘束することはnoncanonicalである。

implicit actuatorは次を復元済みである。

```text
computed = stiffness * (q_cmd - q_actual)
         + damping * (dq_cmd - dq_actual)
         + feedforward
applied  = clip(computed, effort_limit)
```

D28W substep captureでcomputed parity max error 0、applied parity max error 0、clipping classification 100%を確認した。これはactuator implementation parityの結果であり、START capabilityの結果ではない。

### True basin definition

旧10-step proximity/continuous confirmationはstable WALK basin captureと同一視しない。D29Cからは次のE0〜E3を使う。

```text
E0_NOT_ENTERED
  phase-conditioned W_MOVE neighborhoodへ一度も入らない

E1_ENTRY_CROSSING
  旧10-step entryだけ満たし、3 alternating touchdown未満または発散

E2_TRANSIENT_CYCLE_CAPTURE
  3 alternating touchdown以上、2 full stride完走、100-step retention fail

E3_STABLE_LIMIT_CYCLE_CAPTURE
  3 alternating touchdown以上、2 full stride、非発散、100-step retention、canonical safety pass
```

## 5. Current physical interpretation

### L0–L5

| Level | Meaning | Current result |
|---|---|---|
| L0 | liftoff | 成立。D29C `A_CONTINUE_WMOVE` 8/8 |
| L1 | touchdown | 成立。D29C `A_CONTINUE_WMOVE` 8/8 |
| L2 | native W_MOVE neighborhood crossing | legacy evaluator上の一時通過はあったが、common phase-conditioned crossingとしては未確立。A 0/8 |
| L3 | multiple alternating contacts | 成立。A 8/8 |
| L4 | stable WALK limit-cycle capture | 未達。E3 0/8 |
| L5 | 100-step steady WALK retention | 未達。0/8 |

従って、現在の本当のボトルネックは「第一歩が作れない」ではない。

```text
post-touchdown
→ stable WALK limit-cycle capture
```

### D29C route progression

| Route | L0 liftoff | L1 touchdown | L2 common neighborhood | L3 alternating | L4 stable E3 | L5 100-step | E-class summary |
|---|---:|---:|---:|---:|---:|---:|---|
| `A_CONTINUE_WMOVE` | 8/8 | 8/8 | 0/8 | 8/8 | 0/8 | 0/8 | E0 8/8 |
| `B_CAPTURE_06` | 4/8 | 4/8 | 0/8 | 4/8 | 0/8 | 0/8 | E0 8/8 |
| `C_CAPTURE_08` | 4/8 | 4/8 | 0/8 | 4/8 | 0/8 | 0/8 | E0 8/8 |
| `D_STAGE2N_CONTROL` | 4/8 | 4/8 | 0/8 | 4/8 | 0/8 | 0/8 | E0 8/8 |
| `R_A29A` | 8/8 | 8/8 | 0/8 | 8/8 | 0/8 | 0/8 | E0 8/8 |
| `R_A29B0` | 8/8 | 8/8 | 0/8 | 8/8 | 0/8 | 0/8 | E2 7/8, E0 1/8 |
| `R_B29B0` | 8/8 | 8/8 | 0/8 | 8/8 | 0/8 | 0/8 | E2 5/8, E0 3/8 |

ここでL3は「足が交互に接地した」ことを意味し、L4のattractor captureを意味しない。

## 6. D29A / D29B0 entry discrepancy audit

### Evaluator contract difference

`D29A 0/8` と `D29B0 A_STAND_PRECONDITION 7/8` の差は、まずentry evaluator contractの差を含む。

| Evaluator | Distance/schema | Normalization/reference | Phase / touchdown | Confirmation |
|---|---|---|---|---|
| D29A | last ten W_MOVE/READY_RAMP rows。route speed/lateral/yaw/safety中心 | D29A READY-to-W_MOVE p95系。phase-conditioned state distanceなし | phase conditioningなし、touchdown requirementなし | 10 steps |
| D29B0 | any ten consecutive W_MOVE rows。D29A READY-to-W_MOVE p95 + velocity/yaw/safety | distance threshold 12.8774970171285 | phase conditioningなし、touchdown requirementなし | 10 steps |
| D29B | post-touchdown/handoff rows。nearest D26T medoid、contact phase、velocity/yaw/safety | D26T 50 LEFT + 50 RIGHT validated references | phase-conditioned、touchdown requirementあり | 10 steps |
| D29C common | first strict touchdown後のsame-side D26T referenceへのphase-conditioned nearest distance | side-specific p95 reference、command/historyを距離から除外 | `TD0`以降、stable captureとentryを分離 | 旧entryは診断値のみ |

したがって正式なsub-classificationは次のとおりである。

```text
ENTRY_EVALUATOR_CONTRACT_DIFFERENCE
```

旧reported entry countは変更していない。新common判定は別artifactとして保存し、D29C capability gateを優先する。

### Physical state / history difference

比較点は固定step 99、すなわち2秒STAND preconditionまたは対応lifecycleの直後で、step 100 W_MOVE switchの直前である。D29Aのfull body artifactが不足しているため、D29B formal A traceをexact route proxyとして使った。

```text
D29A proxy vs D29B0 P_STAND:
  physical state L2 median: approximately 6.05
  previous action L2 median: approximately 0.165
  controller/runtime: D29A S_HOLD actor vs D29B0 explicit actor

D29B0 P_STAND vs P_WALK_ZERO:
  physical state L2 median: approximately 184.999
  previous action L2 median: approximately 3.765
  contact/air-time/last-contact history mismatch: 8/8
  mode-history mismatch: 8/8 (target mode)
  command/previous-command mismatch: 0/8
  switch alignment: same step 99 before fixed step-100 switch
```

P_STANDとP_WALK_ZEROは同一actor・同一commandでも、2秒間のmode-conditioned preconditioningによりactual posture/velocity/contact history/previous actionが変わっていた。D29A proxyとP_STANDにもactor/runtime差がある。従って「evaluatorだけ」とは言えず、最終interpretationは次である。

```text
MULTIPLE_FACTORS
  - ENTRY_EVALUATOR_CONTRACT_DIFFERENCE
  - PHYSICAL_POSTURE_CONDITIONING
  - PREVIOUS_ACTION_CONDITIONING
  - CONTACT_HISTORY_CONDITIONING
  - MODE_HISTORY_CONDITIONING
```

`D29B0 7/8` はlegacy ten-step proximityのcountであり、stable WALK acquisitionのcountではない。

## 7. Common phase-conditioned return map

D29CはD26S native W_MOVE bundleを参照した。bundle SHA-256は次である。

```text
e4f2250a35a5feee2d1adb415d11121e52164018648bc7678dcf91a47e0894f6
```

referenceは50 LEFT / 50 RIGHTで、exact medoidはLEFT `episode 52 / step 111`、RIGHT `episode 187 / step 115`。phase-state distanceにはcommand/history dimensionを入れていない。保存featureはbase velocity/yaw、projected gravity、joint position/velocity、previous action、CoM/DCM、foot pose/velocity、contact force、support sideを含む。

代表的なA routeのstrict touchdown sequenceは次である。

```text
TD0 LEFT   step 117
TD1 RIGHT  step 125   interval 8
TD2 LEFT   step 133   interval 8
TD3 RIGHT  step 140   interval 7
TD4 LEFT   step 148   interval 8
TD5 RIGHT  step 156   interval 8
```

同side reference distanceはTD0–TD2–TD4およびTD1–TD3–TD5で保存し、raw ratio `d1/d0`、`d2/d1`も保存した。判定は固定marginでpass/failを作らず、source別のraw ratioを優先する。

source-levelのdiagnosticは混在しており、stable captureを支持しない。

```text
A_CONTINUE_WMOVE:  6/8 DIVERGING, 2/8 CONTRACTING
R_A29B0:           5/8 DIVERGING, 3/8 CONTRACTING
R_B29B0:           7/8 DIVERGING, 1/8 CONTRACTING
B_CAPTURE_06:      4 source with usable map, 4 unavailable
C_CAPTURE_08:      4 source with usable map, 4 unavailable
D_STAGE2N_CONTROL: 4 DIVERGING, 4 unavailable
```

例としてA route aggregate raw diagnosticはmedian `d1/d0` approximately `0.737`だが、同じrouteのsource-levelでは6/8がdivergingであり、途中のdistance explosionも記録される。このためaggregate ratio単独をstable captureの証拠にしない。

first-divergence artifactでは、post-touchdownの主要なW_MOVE entry failureがA/B/C/Dでcontrol step 118–119付近、`R_B29B0`では102–108付近に出る。support lossとtorque saturationの既存safety recordもそのまま保持し、entry failureと混同しない。

## 8. Stage2Q and W_MOVE handoff interpretation

Stage2Q common classificationは、0.6 m/sで `CAPTURE_TRUE 1/8`、0.8 m/sで `CAPTURE_TRUE 2/8`。どちらもpopulation gate `>=6/8`に届かず、true-capture episodeのみを対象にしたStage2Q→W_MOVE handoffは `0/3 HANDOFF_TRUE`だった。

従って現時点での分類は次である。

```text
Stage2Q 0.6: CAPTURE_TRUE exists but teacher-positive population gate FAIL
Stage2Q 0.8: CAPTURE_TRUE exists but teacher-positive population gate FAIL
Stage2Q → W_MOVE: HANDOFF_TRUE 0/3; no true handoff established
```

これは「Stage2QをTeacher routeとしてfreezeできる」とは意味しない。D29Cの主分類はStage2Q単独のstable capture不足と、W_MOVE attractorへのhandoff不足を合わせて、既存attractor hard-switch枝を終了するものである。

## 9. Closed or deprioritized hypotheses

新しい証拠がない限り、以下はpause後のdefault resume枝にしない。

```text
historical READY search
zero-speed WALK gate tuning
direct W_MOVE action search
START PPO
START CEM
full first-step WBIK redesign
D28 joint-limit probe series
D28 bounded solver micro-repair
Stage2Q hard-switch combinations
```

ここで「closed」は研究全体の終了を意味せず、D29C時点の証拠に対して主routeとして再開しないという意味である。

## 10. Current unresolved question and why paused

現在の未解決問題は、first touchdownの生成ではない。

```text
touchdown
→ 2–3 alternating strides
→ phase-conditioned W_MOVE limit cycle
→ 100-step retention
```

既存policy hard switch、historical READY、zero-speed WALK gate、model-free探索、full first-step model-based WBIK、position-level centroidal correctionを検証した。liftoff、touchdown、複数alternating contactは作れる一方、stable WALK limit cycle captureは未達だった。

これ以上同じ枝を微調整するより、研究を一時停止して優先順位を見直す。これは「研究が完全に失敗した」という記録ではない。D6以降で、WALK→STAND edge、native W_MOVE reference、fresh lifecycle、action/actuator contract、post-touchdownの真のボトルネックが確定したため、再開地点を狭く固定できたという意味である。

## 11. Resume plan

再開時の唯一の推奨地点は次である。

```text
RESUME_POINT:
  D29C_POST_TOUCHDOWN_CAPTURE
```

推奨手法は、first strict touchdown後の約0.2–1.0秒だけを対象にした短時間のdynamics-constrained post-touchdown capture controllerである。目的は第一歩全体の再設計ではなく、既存touchdown状態から2–3 stable stridesを経てW_MOVE limit cycleへ入ることである。

再開時も、実験停止gateは次の3層に限定する。

```text
Integrity Gate:
  identity / parity / trace completeness / NaN-Inf

Capability Gate:
  3 alternating touchdowns / 2 complete strides /
  100-step retention / canonical safety

Formal Gate:
  source coverage / direction coverage / held conditions
```

single-step distance、KL、condition number等は診断値に留め、単独で実験全体を停止させない。

## 12. Claims supported at pause

現在支持できる主張は次である。

```text
OMNI-WALK→STAND transition Teacherを構築できた。

STANDからliftoffとtouchdownまでは再現可能である。

複数回のalternating touchdownを生成できる。

既存WALK actorへの単純切替だけではstable WALK limit-cycle captureを実現できなかった。

zero-speed WALK modeはmicro-step/support switchingを増加させるが、
安全なREADY intermediateとしては成立しなかった。

position-level centroidal H_z correctionは、
現在のfirst-step hard tasksと強い競合を示した。
```

## 13. Non-claims

次は主張しない。

```text
STAND→WALK完成
S_START_OMNI完成
OMNI-RUN完成
最終single actor完成
全方向統合完成
G1実機で再現
position-level controlが一般に不可能
torque-level WBCが必要と証明済み
```

## 14. Protection audit

pause作成時の扱いは次のとおりである。

| Item | Result |
|---|---:|
| exp_005〜exp_013 unchanged | true（既存protection assertion） |
| D6〜D29C existing artifacts rewritten | false |
| S_HOLD / W_MOVE / S_STOP_OMNI / Stage2N / Stage2Q rewritten | false |
| all checkpoints unchanged | true（既存D29C assertion） |
| new physics | 0 |
| new rollout | 0 |
| new learned checkpoint | 0 |
| persistent update | 0 |
| PPO | 0 |
| CEM | 0 |
| trajectory optimization | 0 |
| validation / held-out | 0 |
| RUN integration | 0 |
| raw restore | 0 |
| remote push | false |

保護対象のhash snapshotは `results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/pause_d29c/protected_hashes.json` に保存した。既存の無関係なdirty/untracked stateは変更・削除・stageしない。

## 15. Pause artifact index

```text
research/exp_014_pause_report_d29c.md
results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/pause_d29c/pause_status.json
results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/pause_d29c/capability_status.json
results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/pause_d29c/resume_contract.json
results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/pause_d29c/experiment_timeline.md
results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/pause_d29c/protected_hashes.json
```

READMEには既存内容を削除・書換えせず、D29C pause statusとresume pointだけを最小追記する。

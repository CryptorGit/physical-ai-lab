# exp_006_unitree_g1_command_skills

Unitree G1のcommand条件付き統合運動スキル実験。RUN/TURNは正式合格、STOPはprototype、CROUCH_SHALLOWは凍結standing baseとscripted primitiveで正式合格している。

## Skill status (2026-07-22)

| Skill | Status | Contract |
|---|---|---|
| RUN | PASS | formal gate passed |
| TURN | PASS | left/right 45/90 deg formal gate passed |
| STOP | PROTOTYPE | formal gate未達。保存済みoverlayを変更しない |
| CROUCH_SHALLOW | PASS | relative pelvis drop 0.08--0.10 m |
| CROUCH_DEEP | NOT_SUPPORTED | `DEEP_CROUCH_RETURN_UNRESOLVED` |
| STEP_OVER_SHALLOW | NOT_SUPPORTED | 5 cm quasi-static pose chain unresolved; safe rejection only |
| LAND_SHALLOW | NOT_SUPPORTED | Stage 2 baseline tolerates the audited 0.02 m point, but scripted v0 has no passing supported range |

CROUCH_SHALLOWの範囲外commandはproductionではclampしない。`command_supported=false`、`applied_depth_m=0`としてCROUCHを開始せず、Stage 2 standing baseを維持して`unsupported_crouch_depth`を記録する。debug clampは診断専用で、技能成功には数えない。

### STEP_OVER Stage 2 single-step audit (2026-07-22)

standing pose-offset方式はNo-Goで終了した。凍結Stage 2 model_4246へminimum-jerk forward command pulseを与え、liftoff、touchdown、double-support recoveryを接触eventでgateするaudit-only optionを実装した。productionのSTEP_OVERは引き続きfail-closedであり、このoptionはaction routeへ接続していない。

144条件の速度/pulse sweepと左右各10 episodeの指定lead試験では、安定した一歩の切り出しとlead選択を同時に満たさなかった。左右指定成功は左20%、右10%、lead選択自体は左50%、右70%だった。厳格なsingle-step（touchdown後40 ms以内にdouble supportへ復帰）の最大安全forward reachは0.0209 m、toe clearanceは0.0253 mで、既存5 cm障害物の約0.36 m placementと0.07 m clearanceに届かない。障害物試験とtwo-step chainはGo条件未達のため実行していない。分類は`PHASE_SELECTION_FAILURE`および`STEP_REACH_INSUFFICIENT`。次候補はwhole-body trajectory optimizationまたは専用stepping policyであり、PPO residualの単純追加には進まない。

### STEP_OVER whole-body trajectory audit (2026-07-22)

固定left-lead contact scheduleについて、Stage 2 standing stateをIsaacから取得し、既存MuJoCo G1とSciPyを用いた段階的contact-constrained IK、minimum-jerk multiple shooting、inverse-dynamics/contact-wrench feasibilityを限定実行した。11/11 waypointはsoft joint limit内で収束し、support keypoint誤差最大0.0051 m、swing target誤差最大0.0013 m、sampled toe/sole/heel clearance最小0.1000 mだった。一方、完全pathにはankle/knee proxy-mesh collisionが4/162 knot残り、接触力solveは122/162 knotのみ収束、最大effort utilization 1.7836、floating-base dynamics残差66.70 Nだった。

したがって物理的な`KINEMATICALLY_INFEASIBLE`/`DYNAMICALLY_INFEASIBLE`の証明とはせず、限定solverの`OPTIMIZATION_FAILURE`と分類する。`TRAJECTORY_FEASIBLE`ではないためIsaac PD replayは実行せず、production STEP_OVERはfail-closed、statusは`NOT_SUPPORTED`のままとする。

旧`exp_006_unitree_g1_hurdle`の追跡対象はcommit `1ae8462`（`Archive Unitree G1 hurdle experiment`）へ保存済み。旧`logs/rsl_rl/physical_ai_g1_hurdle`、`results/exp_006_unitree_g1_hurdle`、checkpoint、動画は削除していない。`exp_005_unitree_g1_flat_run`とIsaac Lab本体は変更しない。

## 環境ID

主要routeは次のID。各IDに`-Play-v0`と`-Eval-v0`もある。

- `Isaac-Motion-Flat-G1-Command-Run-v0`
- `Isaac-Motion-Flat-G1-Command-Turn-v0`
- `Isaac-Motion-Flat-G1-Command-TurnFull-v0`
- `Isaac-Motion-Flat-G1-Command-Stop-v0`
- `Isaac-Motion-Flat-G1-Command-CrouchShallow-v0`
- `Isaac-Motion-Flat-G1-Command-Sequence-v0`

## Actor構造と凍結

RSL-RLの`class_name`が完全修飾した実験側modelを解決できるため、Isaac Lab/RSL-RL本体を変更せず、`g1_command_skills.models:G1CommandResidualActor`を使用する。

```text
base_action     = frozen_stage4_actor(robot_state[0:123])
command_code_i  = skill_command_encoder_i(command_observation[0:29])
state_code_i    = skill_state_adapter_i(robot_state[0:123])
residual_i      = 0.25 * tanh(skill_head_i(state_code_i, command_code_i))
gate            = crossfade(previous_one_hot, current_one_hot, transition_progress)
final_action    = base_action + sum_i(gate_i * residual_i)
```

CROUCH_SHALLOWだけはlearned routeを無効化し、次のbase-option endpointを使う。旧CROUCH PPO routeは比較・研究履歴としてcheckpoint内に残すが、production actionには入れない。

```text
crouch_action = frozen_stage2_standing_base(robot_state)
              + scripted_pose_lookup(relative_target_depth)
```

DOWN/RETURNの相対depthはminimum-jerk、poseは実測pelvis depthで校正した連続piecewise-linear lookupである。正式artifactは`artifacts/exp_006_unitree_g1_command_skills/crouch_shallow_scripted_v1/`、50 episode gateは`results/exp_006_unitree_g1_command_skills/crouch_shallow_scripted_v1/formal_50/gate.json`に保存する。

- base actor: Stage 4と同じ`123 → 256 → 128 → 128 → 37`、全段固定。
- command encoder: skillごとに独立した`29 → 64 → 32`。技能追加時に既習encoderは完全固定。
- state adapter: skillごとに独立した`123 → 128 → 64`。技能追加時に既習adapterは完全固定。
- residual head: 6個の独立`96 → 64 → 64 → 37`。出力層は全headでzero初期化、出力は±0.25に制限。
- 段階別経路: RUNではroute 0、TURNではroute 2、STOPではroute 1のencoder/adapter/headだけを更新する。既習routeと将来route 3–5は固定する。
- sequence段階: 受理済みRUN/TURN/STOP routeは固定し、criticとexploration stdだけを更新する。cross-fadeは固定済みresidual出力を補間する。
- criticとGaussian exploration stdは通常どおり更新する。

別々のskill checkpointを実行時に切り替える方式ではない。各段階は前段checkpoint全体を継承するため、STOP段階のcheckpointはRUN/TURN/STOP headを同時に含み、sequence段階のcheckpointが最終統合成果物となる。

## Command表現

既存123列は順序を変えず、body-frame velocity commandも列9–11に維持する。末尾へ29列を追加し、policy観測は152次元、actionは37次元のまま。

| 追加観測 | 次元 | 座標・表現 |
|---|---:|---|
| current / previous skill | 6 + 6 | RUN, STOP, TURN, CROUCH, STEP_OVER, LAND one-hot |
| path/target heading error | 2 | robot-local `sin`, `cos` |
| target displacement / RUN lookahead / TURN angle state | 2 | RUNはlookahead x/y、TURNはcommanded angle/accumulated yaw、STOPはrobot-local displacement |
| pelvis height | 1 | [m] |
| obstacle geometry / RUN path state | 4 | RUNではsigned lateral error、path前進速度、path横速度、curvature |
| vertical velocity | 1 | [m/s] |
| elapsed / remaining / phase | 3 | segment正規化値 |
| transition progress | 1 | 0–1 |
| recovery mode | 1 | 予約値 |
| target roll/pitch | 2 | [rad] |

one-hotをgateに直接使い、同じ連続commandを異なるheadへ明示的にroutingする。連続parameterはencoderへ同時入力する。RUNでは直線centerlineをpath-localに保持し、1 m先のlookaheadをrobot-localへ変換する。学習時の初期signed lateral errorは±0.40 m、評価時は0 m。絶対world位置はactorへ渡さず、観測次元も増やさない。

## Warm start

初期値は`artifacts/exp_005_unitree_g1_flat_run/g1_periodic_run_stage4_best.pt`（2.3–2.6 m/s、source iteration 5244）。5 m/sのStage 9とcourse固有の旧alignment方策は初期pilotに使わない。

`transfer_checkpoint.py`は旧actorの`mlp.*`を`base_mlp.*`へshape一致で完全copyする。全residual出力が0なので変換直後のactor meanはStage 4と一致する。criticの旧123入力列を完全copyし、新29列だけ0初期化する。optimizerとiterationはresetする。共有encoder時代のRUN_BEST model_39はload/rebase時に旧encoder/adapterを各skill routeへ複製し、pure RUN actionをbitwise維持する。技能境界の`-NewStage`は新architecture用optimizerを再構築する。中断再開では`-NewStage`を付けない。

## Rehearsalと遷移

学習環境はepisode種別を次の比率でsampleする。

| 段階 | RUN task | TURN task | STOP task | full sequence |
|---|---:|---:|---:|---:|
| RUN | 100% | 0% | 0% | 0% |
| TURN | 30% | 70% | 0% | 0% |
| STOP | 20% | 20% | 60% | 0% |
| sequence | 10% | 10% | 10% | 70% |

TURN taskは`RUN→TURN→RUN`、STOP taskは`RUN→STOP`を含み、入口と復帰も評価する。最初のTURN curriculumは左右45°、yaw rate上限0.45 rad/s、TURN 4.5秒である。45° gate合格後だけ`TurnFull`へ進み、90°を70%、45°rehearsalを30%でsampleする。model_0のfull-range監査で右旋回だけ成功80%だったため、次pilotは右70%・左30%とする。full sequenceの基準時間は3.0/2.5/2.0/3.5秒で、学習時は各phaseを独立に±25%変更する。

current/previous one-hotとtransition progressにより、residual actionとskill固有rewardを0.4秒線形cross-fadeする。常時有効なのは転倒、姿勢、torque、加速度、action rate、関節制限、足滑り等の安全項だけ。速度・yawの汎用tracking項とfeet-air-time項は無効化する。

RUNは自由空間の速度追従ではなくpath追従として定義する。path方向速度・path heading・周期走行を維持し、centerlineの±0.20 mは自然な左右運動として罰しない。その外側だけを二乗hingeで罰し、path横速度penaltyとcenterlineへ近づいたstepのprogress rewardを加える。評価の`course_deviation`も同じsigned lateral errorの絶対最大値を使う。

## 評価とbest checkpoint gate

`evaluate.py`は`skills.csv`、`episodes.csv`、`summary.csv/json`を保存する。TURNは開始時world yawを固定し、毎stepのwrapped yaw deltaを加算してunwrapped累積yawを作る。成功は`abs(commanded angle - accumulated yaw) ≤ 0.12 rad`に加え、後続RUNのheading誤差≤0.12 rad、path lateral p95≤0.75 mを要求する。平均local heading errorは`mean_step_heading_error_rad`として診断専用に分離する。

TURN CSV/JSONには`commanded_turn_angle_rad`、`actual_accumulated_yaw_rad`、`final/max turn angle error`、完了時間、turn/straight-recovery成功、post-turn heading/path lateral errorを保存する。full評価は左45°、右45°、左90°、右90°を決定論的に循環し、各成功率を個別集計する。

`diagnose_commands.py`は旧123列と新29列を同じcommandから一貫して変更し、次を保存する。

- command encoder第1層の全29入力列weight norm
- RUN/TURN/STOP別encoder出力
- 同一robot stateで左45°/右45°、0°/左45°、0°/右45°、左45°/左90°、右45°/右90°を比較したbase actor、residual、final actionの各L2差
- 各比較で変わった旧command列9–11と新command列0–28のindex/value
- skill別bounded residual normとgate値
- lateral errorだけを`+0.50/-0.50 m`へ変えた反実仮想action差
- RUN residualのpath-correction関節RMSとsagittal propulsion関節RMS
- lateral errorとresidual normの相関
- 0.4秒cross-fadeを5点sampleしたaction変化
- `evaluate_all.ps1`による`new_command_zero`、`legacy_command_zero`、`all_command_zero`と、新旧commandを対応させた`shuffle`性能

TURNのcommand-sensitive gateは左右と0°/非0°の完全command変更に対してbase、residual、final actionのいずれかが0.005以上変わることを要求する。TURN residualがzeroでも、凍結base actorが旧yaw commandへ反応して実episodeの方向・角度が変われば受理する。左右でactionが変わらない固定時間旋回は不合格にする。

`gate_checkpoints.ps1`はrun directory内の全`model_*.pt`を既習全技能で自動評価する。前段の`gate.json`をbaselineに渡し、以前の技能成功率が5ポイント以上低下した候補をbestから除外する。さらに次のpilot閾値を適用する。

- RUN: 成功≥90%、course-deviation失敗≤5%、転倒≤5%、周期走行≥90%、path速度誤差≤0.25 m/s、path heading誤差≤0.12 rad、Stage 4周期/転倒/速度の大幅低下なし。
- TURN: curriculum対象角度を左右別に成功≥90%、最終累積角度誤差≤0.12 rad、直進復帰≥90%、転倒≤5%、RUN成功≥95%かつ低下<5ポイント。
- STOP: 到達≥90%、位置誤差≤0.5 m、停止速度≤0.2 m/s、直立、RUN/TURN低下<5ポイント。
- sequence: 完走≥80%、転倒≤5%、各skill成功≥90%。

## 短いpilot

repository rootのPowerShellで実行する。`gate_checkpoints.ps1`の`best_checkpoint.json`に選択checkpointが記録される。

```powershell
$s = ".\experiments\isaaclab\exp_006_unitree_g1_command_skills\scripts"
$stage4 = ".\artifacts\exp_005_unitree_g1_flat_run\g1_periodic_run_stage4_best.pt"

# 1. RUN path residual: 40 iterations
& "$s\train.ps1" -Skill run -WarmStartCheckpoint $stage4 -NumEnvs 1024 -MaxIterations 40 -RunName pilot_run_path

# 転移直後をlocomotion preservation baselineとして保存（path gate不合格でもbaseline JSONは生成される）
& "$s\evaluate_all.ps1" -Checkpoint <TRANSFER_MODEL_0> -Stage run -Episodes 50 `
  -Output results/exp_006_unitree_g1_command_skills/stage4_path_baseline
& "$s\gate_checkpoints.ps1" -RunDirectory <RUN_RUN_DIR> -Stage run -Episodes 50 `
  -BaselineGate results/exp_006_unitree_g1_command_skills/stage4_path_baseline/gate.json

# 2a. TURN +/-45 deg: RUN_BEST model_39から40 iterations
& "$s\train.ps1" -Skill turn -TurnCurriculum 45 -Checkpoint <RUN_MODEL_39> -NewStage `
  -ParentGate <RUN_GATE_JSON> -NumEnvs 1024 -MaxIterations 40 -RunName pilot_turn45_skill_route
& "$s\gate_checkpoints.ps1" -RunDirectory <TURN45_RUN_DIR> -Stage turn -TurnCurriculum 45 `
  -BaselineGate <RUN_GATE_JSON> -Episodes 50

# 2b. +/-45 deg合格後だけ90 degを追加
& "$s\train.ps1" -Skill turn -TurnCurriculum full -Checkpoint <TURN45_BEST> -NewStage `
  -ParentGate <TURN45_GATE_JSON> -NumEnvs 1024 -MaxIterations 40 -RunName pilot_turn_full_skill_route
& "$s\gate_checkpoints.ps1" -RunDirectory <TURN_FULL_RUN_DIR> -Stage turn -TurnCurriculum full `
  -BaselineGate <TURN45_GATE_JSON> -Episodes 50

# 学習なしの短い左右45/90確認（4 category x 5 parallel episodes）
& "$s\evaluate_turn_matrix.ps1" -Checkpoint <TURN45_BEST> -EpisodesPerCategory 5 `
  -SavedRunSummary <TURN45_RUN_NORMAL_SUMMARY> -Output results/exp_006_unitree_g1_command_skills/turn_full_short

# 3. STOP residual: 125 iterations; RUN+TURN bestを保持したままhead 1だけ追加
& "$s\train.ps1" -Skill stop -Checkpoint <TURN_BEST> -NewStage -ParentGate <TURN_GATE_JSON> -NumEnvs 1024 -MaxIterations 125 -RunName pilot_stop_residual
& "$s\gate_checkpoints.ps1" -RunDirectory <STOP_RUN_DIR> -Stage stop -BaselineGate <TURN_GATE_JSON> -Episodes 50

# 4. 統合sequence: 200 iterations
& "$s\train.ps1" -Skill sequence -Checkpoint <STOP_BEST> -NewStage -ParentGate <STOP_GATE_JSON> -NumEnvs 1024 -MaxIterations 200 -RunName pilot_integrated_sequence
& "$s\gate_checkpoints.ps1" -RunDirectory <SEQUENCE_RUN_DIR> -Stage sequence -BaselineGate <STOP_GATE_JSON> -Episodes 50 `
  -IntegratedCheckpoint ".\logs\rsl_rl\physical_ai_g1_command_skills\integrated_run_turn_stop_best.pt"

# 明示的なsequence評価、diagnostic、GUI再生
& "$s\evaluate.ps1" -Skill sequence -Checkpoint <INTEGRATED_CHECKPOINT> -Episodes 50 -Output results/exp_006_unitree_g1_command_skills/sequence_50ep
& "$s\diagnose_commands.ps1" -Checkpoint <INTEGRATED_CHECKPOINT> -Stage sequence
& "$s\play.ps1" -Skill sequence -Checkpoint <INTEGRATED_CHECKPOINT> -Visualizer kit
```

生成checkpoint、logs、results、CSV/JSON、動画はGitへ追加しない。smoke checkpointは性能選択に使わない。

## 次段階

CROUCHはhead 3、STEP_OVERはevent residual head 4、LAND/RECOVERはhead 5を同じgateへ追加する。各headを単独＋既習rehearsalで成立させてからsequenceへ加える。STEP_OVERはIsaac Lab cuboid primitiveの低い固定障害物から開始し、visual meshとphysics collisionを分離する。Blender MCPは今回使用していない。RUN/TURN/STOP統合成功後にのみtrack、lane、turn marker、cone、stop zone、低障害物のvisual assetへ使用する。

## Command system v1（2026-07-23）

新技能探索は終了し、152次元actor入力を変更しない外部routerとして統合した。STANDは7番目のpolicy skillではなく、凍結Stage 2 standing baseを選ぶcontroller stateである。正式familyは`RUNNING_FAMILY={RUN, TURN}`と`STANDING_FAMILY={STAND, CROUCH_SHALLOW}`。STOPはprototype、CROUCH_DEEP/STEP_OVER/LANDは非対応である。

family内transitionだけを正式対応する。cross-family requestは`CROSS_BASE_FAMILY_TRANSITION_UNRESOLVED`でfail-closedに拒否し、base、command、active controllerを変更しない。STOPを暗黙に挿入しない。LANDの0.02 m dropはstanding baseの観測済みpassive robustnessであり、LANDのsupported rangeではない。

machine-readable結果は`artifacts/exp_006_unitree_g1_command_skills/command_system_v1/`、GUI dispatcherは`scripts/play_command_system.ps1`に保存する。

本実験は`command_system_v1`の正式PASSをもって凍結した。以後のSTAND/WALK/RUN間transitionは`exp_007_unitree_g1_walk_centered_transitions`で扱い、exp_006のcheckpoint、formal metrics、supported rangeを変更しない。最終日本語レポートは`reports/exp_006_unitree_g1_command_system_v1_final_report_ja.md`を参照する。

## Isaac Sim showcase（表示・録画専用）

`scripts/play_command_system_showcase.ps1`は、凍結済み`command_system_v1`を映像で確認するための専用入口である。PPO学習、checkpoint、formal artifact、formal evaluator、command routerを変更しない。再生telemetryと動画だけを`results/exp_006_unitree_g1_command_skills/showcase_v1/`および指定した動画pathへ書く。

既存の`play_command_system.ps1 -Demo RUN_TURN_RUN`は90°showcaseではなかった。1環境の`TurnFull-Eval`が決定論的な`(+45, -45, +90, -90)`列の先頭を選ぶため、実commandは左45°（0.785398 rad）だった。既存GUI実測はactualとの差0.005998 radであり、headingと累積yawは正常に変化していた。target headingはTURN入口headingへcommand angleを加えた固定world heading、legacy yaw-rateはheading errorから±0.75 rad/s内で生成される。TURN終了は角度到達ではなく5.0秒の固定segment duration、旋回後RUNは同demo指定で2.5–3.2秒である。

既存evaluatorはcamera APIを呼ばず、継承viewerも`origin_type=world`のためrobot yaw追従と毎step transform上書きは行っていない。見えにくさの確定要因は、consoleが角度・方向を表示しないまま実際には45°を選んだことと、showcase向けにコース全体を構図化していなかったことである。showcaseは初期poseをworld +Xへ決定論化し、左右90°を明示配線し、world固定構図、grid、world軌跡、heading arrow、TURN開始/終了markerを追加する。

利用可能mode:

- `TURN_LEFT_90`: RUN 3秒 → 左90° → 新world方向へRUN 4秒。
- `TURN_RIGHT_90`: RUN 3秒 → 右90° → 新world方向へRUN 4秒。
- `TURN_S_CURVE`: RUN → 左90° → RUN → 右90° → RUN。対応済みRUN/TURNだけを使うが、複数TURNを明示注入するshowcase-only sequenceでありformal sequence評価ではない。
- `CROUCH_SHOWCASE`: STAND 2秒 → CROUCH 0.09 m → HOLD 2秒 → STAND → HOLD 2秒。
- `SAFE_REJECTION`: standing controllerを維持したままSTEP_OVERをrequestし、`supported=false`、`primitive_started=false`、primitive action discontinuity 0を表示する。
- `FULL_REEL`: 左90°、右90°、CROUCH、safe rejectionを4つの独立Isaac sceneとして順次起動する。scene間はreset/動画cutであり、RUNNING_FAMILYとSTANDING_FAMILYを連続遷移しない。

camera mode:

- `WORLD_FIXED`（default）: deterministic course envelopeから一度だけworld camera transformを決め、実行中は更新しない。
- `FOLLOW_POSITION`: camera XYをrobotへlow-pass追従させるが、eye-target vectorはworld固定でrobot yawを使わない。
- `TOP_DOWN`: course envelope中心のworld固定上空camera。

単独再生:

```powershell
.\experiments\isaaclab\exp_006_unitree_g1_command_skills\scripts\play_command_system_showcase.ps1 `
  -Showcase TURN_LEFT_90 `
  -Camera WORLD_FIXED
```

FULL_REELを1920×1080 MP4へ録画:

```powershell
.\experiments\isaaclab\exp_006_unitree_g1_command_skills\scripts\play_command_system_showcase.ps1 `
  -Showcase FULL_REEL `
  -Camera WORLD_FIXED `
  -Record `
  -OutputPath ".\videos\exp_006_command_system_v1_showcase.mp4"
```

現環境のIsaac Labが提供するKit/replicator `rgb_array` captureをGym recorderでMP4化し、同梱OpenCVでstatus overlayをburn-inする。FULL_REELは各scene clipの間へ1秒の`SCENE CUT / RESET` cardを挿入し、物理時間を変更せずに連結する。clipも指定動画と同じdirectoryへ保持する。GUIそのものをOBS等で録画する場合も同じcommandから`-Record`だけを外せばよく、viewport overlay、固定camera、1920×1080構図、決定論的timingは維持される。

各実行前preflightは選択mode/camera、RUN・standing・CROUCH checkpoint/artifact、角度、2.4 m/s RUN速度、sequence、unsupported request、record pathを表示する。TURN後はcommanded/actual/final errorをconsoleへ出し、全stepのworld XY、heading、target heading、legacy yaw-rate、actual accumulated yawを`showcase_telemetry.json`へ保存する。動画上のRUNはformal play設定と同じ2.4 m/s、TURN中は2.0 m/sであり、5 m/s旋回ではない。STOP、STEP_OVER、LAND、実機能力はPASS表示しない。

## STOP closed-loop pilot

STOPは開始時のworld停止目標、entry speed、必要減速度を固定し、robot-local残距離からbraking targetを毎step更新する。29-D追加commandの次元は増やさず、STOPで未使用のobstacle 4 slotをforward speed、required deceleration、braking target、hold progressに再利用する。curriculumは`-StopCurriculum A|B|C`で、A=`0.8–1.4 m/s, 1.5–2.5 m`、B=`1.4–2.0 m/s, 1.5–3.0 m`、C=`2.0–2.6 m/s, 2.0–4.0 m`である。全stageのholdは1.0 s、STOP/RUN/TURN rehearsalは60/20/20で、更新可能actor parameterはSTOP routeだけである。

評価のSTOP fall windowはSTOP entryからhold完了までである。RSL-RLの自動reset後のRUNをterminal segmentとして数えない。`episodes.csv`と`skills.csv`はfall/first-failure時刻、STOP entry/exit/min speed、hold、saturation joint、saturation fraction、最大actionを持ち、`stop_curve.csv`はstep単位の残距離・実速度・必要減速度・braking target・progressを持つ。

## 実装smoke（2026-07-19）

- static parse、環境登録、152-D観測、37-D action: pass
- command切替`0,2,0,1`、one-hot、0.4秒transition: pass
- residual warm start、RSL-RL load、PPO update/save: pass
- checkpoint resumeと追加PPO update: pass
- PPO更新後もStage 4 base全tensorがbitwise同一: pass
- 29-D command内のpath-local lookahead/lateral error/forward・lateral velocity/curvature: pass
- path reward全term生成、command診断JSON、RUN CSV/JSON評価列: pass
- 1-update smokeのlateral-error反実仮想差は閾値未達としてgateがreject: pass（意図した安全側判定）
- 旧model_39→skill-local RUN action、TURN 5 PPO update前後のRUN route 14 tensor/action bitwise一致: pass
- 同一RUN Eval episode 600 stepで親model_39とTURN更新後model_4のaction差が全step厳密0: pass
- 更新可能parameterがTURN encoder/adapter/head、exploration stdだけ（criticはactor外）: pass
- 左右45°/90°のcommand/累積yaw CSV、summary、gate heading値の完全一致: pass
- 修正評価器によるmodel_39の4-category smoke: 4/4成功（各1 episodeの配線確認であり性能評価には不使用）
- play経路のTorchScript/ONNX export生成とsimulation loop起動: pass
- 実pilot性能: 未実施（smoke checkpointは性能評価に不使用）

# exp_012 Unitree G1 single-policy bidirectional locomotion — 最終研究報告書

## 1. Executive summary

本研究は、Isaac Lab上のUnitree G1において、`STAND → WALK → RUN → WALK → STOP / STAND`を、実行時に一つのactor checkpointだけで生成できるかを検証した。中心的な問いは、WALKとRUNという異なる接触様式を一つのニューラルネットワークへ統合し、速度commandと歩容commandで明示的に選択・遷移できるか、である。runtimeでteacher、expert、router、checkpoint switching、action blendingを使わないことを最初から契約とした。

中心課題には肯定的な答えが得られた。124D入力（既存123D observationとscalar `gait_cmd`）、37D joint-position action、`124→256→128→128→37`の一つのmean actorにより、同じ1.2 m/sで`gait_cmd=0`ならWALK、`gait_cmd=1`ならRUNを100%の決定論的endpoint評価で出し分けた。WALK→RUNとRUN→WALKも各100%、0.6–1.2 m/s WALKおよび1.2–2.6 m/s RUNも各100%だった。最終sequence用のStage 2Q checkpointは、WALK/RUN、加減速、復帰WALKを一つの重みで実行し、走行後に平均0.0547 m/sまで減速した。

一方、最終sequenceのformal completionは0%だった。原因はWALK/RUNの失敗ではなく、初期・最終STANDに課した「episode中のflightが完全に0で、final double supportが95%以上」という厳格な接触gateである。最終方策には微小な足踏みが残った。Stage 2RではSTAND専門方策とWALK_TO_STAND専門方策を同じ環境でpositive controlとして再評価したが、formal successはいずれも3%に留まった。速度と転倒の観点ではほぼ静止していたため、実用的停止と厳格な接触定義による完全静止を区別する必要がある。

もう一つの未解決点は、gait semanticsを継続PPOで保持することだった。soft endpoint KL anchor、current-state anchor、固定LR、Adam moment attenuation/zero/resetを診断したが、5 updateのsemantic-retention gateを通過しなかった。現在の最良artifactは学習継続前の統合方策であり、追加PPOによる改善を主張しない。

したがって、project-level classificationを次とする。

```text
EXP_012_CLOSED_WITH_SINGLE_POLICY_LOCOMOTION_SUCCESS_AND_STRICT_STAND_LIMITATION
```

これは過去stageのformal gateを上書きしない。single-policy WALK/RUN統合という中心研究課題には十分な肯定的結果が得られ、残課題はstrict STANDとcontinued PPO retentionという独立問題に分離できたため、本研究をクローズする。

最重要artifactは二つである。

- gait-core: Stage 2N initial checkpoint、SHA-256 `04b43e5497bc35e2d00fa4476f9120f9439e0953283c69cf8ca1e9635dedd121`
- final sequence: Stage 2Q selected checkpoint、SHA-256 `66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698`

## 2. Research motivation

先行するG1実験では、STAND、WALK、RUN、各transitionは個別checkpointやexpertへ分離されていた。状態graphと専門家の組合せは能力を明示しやすい一方、runtimeの分岐、切替条件、handoffの失敗を持ち込む。本研究では「一つの重み」を厳密な契約とし、速度だけでなく歩容そのものを一つのpolicyから選択できるかを調べた。

最初は速度commandだけで全regimeを統合しようとした。しかし、同じ1.2 m/sでもWALKとRUNは異なる接触周期を持つ。速度追従に成功しても、要求した歩容へ移るとは限らない。この曖昧さを、学習不足、reward不足、局所到達性、state manifold、optimizer履歴、network capacityへ順に分解した。

## 3. Prior foundation

### 3.1 exp_005

exp_005はG1歩行方策を段階的に高速化し、周期走行の成立条件を作った。失敗をflight、alternating landing、安全性、heading、slip、impactへ分解し、報酬変更を必要最小限に限定した。Stage 2のWALK parent `model_4246.pt`とStage 4のRUN-capable parent `model_5244.pt`が、exp_012のpositive controlとteacherになった。RUN系列は最終的に5 m/s付近まで安定化したが、WALKとRUNは別の重みに残った。

### 3.2 exp_007〜exp_010

exp_007はSTAND、WALK、RUN_LOWを複数expertとstate graphで統合し、維持状態と遷移を分離した。STAND↔WALKとWALK→RUNは成立したが、RUN→WALKはtarget-WALK basinを保持できず、formal edge success 0%だった。exp_008はこの非対称graphを診断的にクローズした。exp_009はpost-hoc unified student、容量増加、multi-head、短horizon supervision、bounded residual、phase-conditioned base morphを検証したが、closed-loop WALK attractorを保持できず`CLOSED_NO_GO_UNIFIED_ACTION_MANIFOLD`となった。exp_010はpost-RUN low-speed attractorを別方向から調べたが、単純な統一表現の解決には至らなかった。

### 3.3 exp_011

exp_011はUnitree Go2で、一つのactor checkpointによる`0 → low → 2.0 → low → 0`を検証した。途中でfirst-update安定性、heading、contact kinematics、slipを段階的に診断した。この「速度域を一つのactorで往復する」肯定的結果が、より接触様式の差が大きいG1でsingle-policy統合へ再挑戦する動機になった。

## 4. exp_012 system contract

| 項目 | 契約 |
|---|---|
| Robot | Unitree G1 minimal asset |
| Original observation | 123D |
| Added input | scalar `gait_cmd` 1D |
| Total input | 124D |
| Action | 37D joint-position target |
| Action scale | 0.5 |
| Actor | `124→256→128→128→37`, ELU |
| Gaussian | one gait-conditioned diagonal head |
| WALK endpoint | `gait_cmd=0` |
| RUN endpoint | `gait_cmd=1` |
| Runtime checkpoint | 1 |
| Runtime teacher/expert/router | 0 |
| Checkpoint switch/action blend | 0 |

Physics dtは0.005 s、control dtは0.020 s、decimationは4である。既存123D observation、37D action、physics、PD gains、friction、robot assetは統合過程で変更していない。

## 5. 全stage chronology

### 5.1 Initial joint curriculum

最初の仮説は、`ZERO_HOLD`、`WALK_STEADY`、`RUN_HOLD`、`BIDIRECTIONAL_SEQUENCE`を同一PPO curriculumへ入れれば、速度commandだけで単一方策が全regimeを学べるというものだった。WALKは保持されたが、RUN completion basinへ安定して入れず、integrated sequenceの成功率も低かった。ここで「単に共同学習時間を増やせばよい」という仮説は弱くなった。

### 5.2 Yaw diagnostics — Stage 1 / 1B

高速域で速度依存yaw biasを発見した。pipelineのquaternionやcommand bugではなく、frozen policyのfeedforward biasだった。診断専用cancellerで補正可能性を確認し、`G1_YAW_BIAS_SPEED_CONDITIONED_CANCELABLE`、続いて`G1_SPEED_CONDITIONED_YAW_CANCELLATION_PASS`となった。ただしPPO中の外部controllerは常にOFFとし、学習結果へ混入させなかった。

### 5.3 Strict PPO resume bug — Stage 2A / 2B

Stage 2 pilotの事前局所診断は、yaw action perturbationだけではheadingを十分制御できず`G1_YAW_RATE_NOT_LOCALLY_CONTROLLABLE`だった。このため外部controllerを学習へ混ぜず、yaw command 0の契約で本体を評価した。

最初のPilot updateでは、復元optimizerのLRが約`2.25e-5`である一方、runtime `learning_rate`が`0.001`だった。adaptive KLが誤ったruntime LRをoptimizerへ再注入し、first updateを破壊した。この実runは`EXP012_FIRST_UPDATE_UNSTABLE`として停止した。Stage 2Aは現象を`PPO_FIRST_UPDATE_TRUE_DISTRIBUTION_SHIFT`と診断し、Stage 2Bではoptimizer parameter groupをsource of truthとしてruntime/schedulerを同期し、fresh processで再現して`PPO_RUNTIME_LR_RESUME_FIX_PASS`となった。以後の結果はこの修正後contractで評価した。

### 5.4 First pilot and retry

単一の許可されたretryは300 iterationsを完走し、iteration 100を選択した。formal evaluationはSTAND約94%、WALK約98–100%、RUN 2.4部分成功、RUN 2.6不安定、RUN→WALK弱、integrated sequence約38%だった。分類は`G1_SINGLE_POLICY_MULTIPLE_FAILURES`。WALK retentionは強かった一方、RUNとreverse transitionの不足が分離された。

### 5.5 Gradient interference diagnosis — Stage 2C

zero/walk/run/sequenceのregime別gradientを測った。RUNとWALKのcosineは正、RUNとzeroも小さな負に留まり、RUN対combinedの関係も「多regimeがRUN gradientを打ち消す」説明を支持しなかった。formal artifactのclassificationは`RUN_REWARD_REACHABILITY_FAIL`であり、結論はmulti-regime gradient interferenceではなく、RUN rewardが実際に到達するeventの不足へ調査対象を移すべき、というものだった。

### 5.6 RUN reward reachability — Stage 2D

run rewardを意味的に変えず、precursor、safe flight、alternating completionを追跡した。iteration 100ではprecursor 2、safe-flight 11、completion 0に対し、exp_005 positive controlではcompletion 5,601回だった。run-specific/base gradient比も約0.00272と極小だった。分類は`RUN_REWARD_REACHABILITY_FAIL`および`RUN_PRECURSOR_ONLY_NO_COMPLETION_BASIN`。reward実装が不可能なのではなく、walk parentがalternating completion basinへ入れないことが示された。

### 5.7 Phase A RUN-focused continuation — Stage 2E

RUN比率を高めたPhase Aで、学習中に241 completion eventが出現した。69/100 iterationsで発火し、最大密度は0.0335%だった。選択checkpointはiteration 50で、WALK/STAND retentionを維持した。しかし凍結deterministic mean policyではcompletion 0だった。分類`SINGLE_POLICY_RUN_COMPLETION_EMERGED_PARTIAL`は「探索で発見したがmeanへ定着しない」を表す。

### 5.8 Exploration-only boundary — Stage 2F

複数checkpoint、速度、std倍率を掃引するとcompletionはnoise下で再現したが、deterministicでは全checkpoint 0だった。成功actionは通常のGaussian sample分布内で、極端なoutlier依存ではなかった。completion gradient/totalは0.0738%で、約16倍密度なら1%を超える推定だった。主分類は`PHASE_A_BOUNDARY_MULTIPLE_CAUSES`、副因はexploration-only、mean非定着、periodic gait/reward quality gap、signal sparsityだった。

### 5.9 Event-stratified on-policy — Stage 2G

completion windowを同一on-policy update内で4/8/16倍にoversampleした。M4/M8/M16のcombined gradientとcompletion gradientのcosineは負側へ動き、completion-window lossとunsafe-window lossが悪化した。deterministic completionは全条件0、2.4 m/s periodicも0%、fallはM0の33.3%からstratified条件で46.7–53.3%へ悪化した。`EVENT_STRATIFIED_ON_POLICY_NO_EFFECT`となり、単純なsample頻度増加を閉じた。

### 5.10 Short-horizon completion replay — Stage 2H

現在方策自身が生成したcompletion windowを最大4 shadow iterations再利用した。importance ratio、KL、ESSのoff-policy validityは概ね良好だったが、completionはmeanへ定着せず、fallも悪化した。compute-matched backgroundを上回る因果効果がなく、`SHORT_HORIZON_COMPLETION_REPLAY_NO_EFFECT`となった。walk-parentからRUNをconsolidateする経路を正式に閉じた。

### 5.11 Reverse continuation from RUN parent — Stage 2I

次にexp_005 Stage 4 RUN-capable parentから逆向きに1.2 m/s WALKを加えた。RUN 2.4/2.6は100%を維持し、1.2 m/sへの速度追従もMAE 0.067 m/s、fall 0%だった。しかしcontact cycleは`PERIODIC_RUNNING` 100%のまま、2.4/2.6→1.2 WALK acquisitionは0%だった。分類は`REVERSE_SINGLE_POLICY_WALK_RECOVERY_FAIL`。速度制御と歩容制御が別問題であることが明確になった。

### 5.12 Low-speed action-manifold diagnosis — Stage 2J

同じ1.2 m/sでも、WALK positive controlはflight fraction 3.49%、stride frequency 1.40 Hz、RUNはflight 47.99%、stride 6.22 Hzだった。WALK returnはRUNより`+2.921`、95% CI `[2.794, 3.047]`で高かったため、reward indifferenceではない。

state分布のnonlinear classifier AUROCは約`0.999986`で`WALK_RUN_STATE_DISJOINT`。WALK states上のWALK/RUN action L2は2.957、WALK actionのRUN policy下importance-ratio有効率は0.16%だった。bounded single-step perturbationは0/1024、4-step searchは1/1024（0.098%）だけ成功した。減速履歴AUROCは0.553で、最終的には共通RUN-at-1.2 attractorへ収束した。結論は`WALK_MANIFOLD_DYNAMICAL_BARRIER`、主分類`LOW_SPEED_WALK_MANIFOLD_NOT_LOCALLY_REACHABLE`だった。

### 5.13 Gait-command latent — Stage 2K

速度commandだけでは同一速度の目標歩容を指定できないため、独立scalar `gait_cmd∈[0,1]`を追加した。WALK teacherを`g=0`、RUN teacherを`g=1`のendpoint labelとして、一つの124D actorへ教師あり統合した。

selected step 14,000では、1.2 m/s `g=0`がWALK 100%、1.2 m/s `g=1`がRUN 100%、2.4/2.6 RUNも100%。WALK→RUNは100%（平均0.558 s）、RUN→WALKは100%（平均0.980 s）だった。一つのmean actorが異なるWALK/RUN attractorを明示的に選択できた。mean表現、closed-loop authority、toggle、single-weight auditはPASSだったが、一つの固定stdで異なるteacher Gaussianを表現できず、既存formal classificationは`GAIT_LATENT_REPRESENTATION_FAIL`のまま維持された。

### 5.14 Gaussian std representation — Stage 2L

WALK teacherとRUN teacherは異なるstate-independent 37D stdを持っていた。一つの固定stdではendpoint Gaussian KL≤0.05を同時に満たせないため、`log_std_walk[37]`と`log_std_run[37]`を一つのheadに持たせ、gait_cmdでlog-space線形補間した。static Gaussian endpoint、serialization、deterministic bitwise regressionはPASSした。

ただしteacher full stdをclosed loopで使うと、student WALKはWALK_LIKE 0%、PERIODIC_RUNNING 98%、fall 2%へ崩れた。分類は`GAIT_CONDITIONED_STD_STATIC_PASS_CLOSED_LOOP_FAIL`。teacher stdは完成歩容の安全なruntime分布ではなく、teacher学習時の探索量である、という重要な区別が得られた。

### 5.15 Exploration calibration — Stage 2M

凍結meanに対しgait別std倍率を掃引した。最大safe multiplierはWALK 0.40、RUN-low 0.70、RUN 2.4 1.00、RUN 2.6 0.80で、RUN共通境界は0.70だった。境界より一段小さい`alpha_walk=0.30`、`alpha_run=0.65`を選択した。

候補分布ではWALK 100%/fall 0%、RUN-low 97%/fall 3%、paired authority 97%。WALK→RUN 100%/fall 0%、RUN→WALK 100%/fall 1%だった。分類は`SAFE_GAIT_CONDITIONED_EXPLORATION_WINDOW_FOUND`。

### 5.16 PPO endpoint retention — Stage 2N

通常rewardを変えず、開始時統合policyをfrozen referenceとするsoft exact-Gaussian KL anchorを追加した。4 endpointを均等weight、beta 0.01で一更新preflightはPASSした。しかし連続PPOではWALK anchor KLがiteration 1から`0.01857, 0.03395, 0.04699, 0.06655`と累積し、iteration 4でearly stopした。分類は`GAIT_CONDITIONED_PPO_MULTIPLE_FAILURES`。一更新の安全性は長期semantic retentionを保証しなかった。

### 5.17 Anchor accumulation diagnosis — Stage 2O

KL方向はreference||current、mean/std両方、4 endpoint各25%、全PPO minibatchで計算され、実装不整合はなかった。static anchor coverageは失敗域だったがcurrent-state anchorも改善せず、adaptive LRは副因、criticも主因ではなかった。beta 0.10/fixed LRの5-update最良値でもWALK KL 0.03838、RUN 1.2 0.02524、RUN 2.4 0.02834、RUN 2.6 0.02879で全endpoint≤0.03を満たさなかった。raw combined vs anchor cosine median `+0.164`に対しeffective Adam vs anchorは`+0.003`で、分類は`ADAM_HISTORY_SUPPRESSES_ANCHOR`となった。

### 5.18 Optimizer moment diagnosis — Stage 2P

RUN teacherからStage 2K integrated actorへparameter semanticsが大きく変化し、shape対応だけで移植したmomentは現在のanchorと整合しなかった。first moment 25% attenuation、zero、actor full resetを比較したが、5-update WALK KLは順に約0.03946、0.03626、0.03873で、baseline 0.03838をgate内へ押し戻せなかった。いずれかのRUN endpointも0.03を超えた。分類は`ACTOR_MOMENT_ADAPTATION_NO_EFFECT`。古いAdam momentは不整合だったが、semantic driftの主要因ではなかった。

### 5.19 Final supervised sequence integration — Stage 2Q

PPO継続を止め、STAND/WALK/RUN endpointと既に成功していたtoggleをmean-action supervisionで一つのstudentへ統合した。std headは凍結し、DAggerを2 rounds実施した。selected checkpointはSHA `66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698`。

WALK 0.6/0.8/1.0/1.2、RUN 1.2/2.4/2.6、STAND→WALK、WALK→RUN、RUN加速/減速、RUN→WALKはすべて100%、endpoint fallは0%だった。integrated sequenceではinitial WALK 97%、RUN 100%、return WALK 100%、fall 5%、最終speed mean 0.0547 m/s、heading p95 mean 0.0804 rad、dangerous slip 2%、impact 0%、long-dwell saturation 0%だった。

formal STANDとWALK→STANDは0%で、final sequence formal completionも0%。分類は`G1_FINAL_STAND_STOP_FAIL`。static held-out action MSE最悪0.000234、cosine≥0.99989だったため、単純な教師模倣不足とはしなかった。

### 5.20 True STAND source diagnosis — Stage 2R

STAND専門source `model_4246.pt`を同じStage 2Q環境で100 episodes再評価すると、formal success 3%、fall 1%、mean speed 0.00551 m/s、flight-zero 89%、final double support 3%だった。WALK_TO_STAND専門source `model_0.pt`もformal completion 3%、fall 2%、mean speed 0.00122 m/s、flight-zero 97%、final double support 0%だった。

両positive controlがstrict gateを通らなかったため、supervised integrationへ進まずfail-closedし、`G1_FINAL_STAND_POSITIVE_CONTROL_FAIL`とした。Stage 2Qの停止失敗を統合student固有の失敗と断定できないことが確定した。

## 6. Quantitative summary

| Stage | Parent | Method | WALK | RUN | W→R | R→W | STAND | Final sequence | Primary failure | Classification | Selected SHA |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| Stage 2 retry | WALK parent | joint PPO | 98–100% | 2.4 partial / 2.6 unstable | partial | weak | 94% | 38% | RUN basin | `G1_SINGLE_POLICY_MULTIPLE_FAILURES` | `8d8afac6…` |
| 2D | retry iter100 | reward reachability | retained | completion 0 | — | — | — | — | no completion basin | `RUN_PRECURSOR_ONLY_NO_COMPLETION_BASIN` | diagnostic |
| 2E | retry iter100 | RUN Phase A | retained | stochastic 241 events | — | — | retained | — | deterministic 0 | `SINGLE_POLICY_RUN_COMPLETION_EMERGED_PARTIAL` | `4edbb595…` |
| 2G | Phase A iter50 | event-stratified PPO | retained | deterministic 0 | — | — | retained | — | loss/fall worsen | `EVENT_STRATIFIED_ON_POLICY_NO_EFFECT` | shadow |
| 2H | Phase A iter50 | short replay | retained | no consolidation | — | — | retained | — | no causal gain | `SHORT_HORIZON_COMPLETION_REPLAY_NO_EFFECT` | shadow |
| 2I | RUN parent | reverse PPO | RUN gait at 1.2 | 100% | 100% | 0% | — | — | WALK recovery | `REVERSE_SINGLE_POLICY_WALK_RECOVERY_FAIL` | `707bd50a…` |
| 2J | W0/R0/R1 | manifold diagnosis | W0 100% | R0 100% | — | 0% local | — | — | dynamical barrier | `LOW_SPEED_WALK_MANIFOLD_NOT_LOCALLY_REACHABLE` | frozen |
| 2K | RUN-init 124D | endpoint supervision | 100% | 100% | 100% | 100% | — | — | shared std KL | `GAIT_LATENT_REPRESENTATION_FAIL` | `d0c46acd…` |
| 2L | 2K | conditioned std | deterministic 100% | deterministic 100% | 100% | 100% | — | — | full-std stochastic WALK | `GAIT_CONDITIONED_STD_STATIC_PASS_CLOSED_LOOP_FAIL` | `175131f7…` |
| 2M | 2L | std calibration | 100% | low 97%, high safe | 100% | 100% | — | — | none in calibrated scope | `SAFE_GAIT_CONDITIONED_EXPLORATION_WINDOW_FOUND` | wrapper |
| 2N | gait core | PPO+soft anchor | initially 100% | initially 99–100% | 100% | 100% | — | — | cumulative anchor drift | `GAIT_CONDITIONED_PPO_MULTIPLE_FAILURES` | `04b43e54…` initial |
| 2P | 2N initial | moment adaptation | initial retained | initial retained | initial retained | initial retained | — | — | no 5-update gate | `ACTOR_MOMENT_ADAPTATION_NO_EFFECT` | shadow |
| 2Q | 2N initial | supervision+2 DAgger | 100% 0.6–1.2 | 100% 1.2–2.6 | 100% | 100% | 0% strict | 0% strict | stand contact gate | `G1_FINAL_STAND_STOP_FAIL` | `66ca4575…` |
| 2R | specialist sources | positive control | retained sources | — | — | WTS source 3% strict | source 3% | — | positive control gate | `G1_FINAL_STAND_POSITIVE_CONTROL_FAIL` | no student |

## 7. Main scientific findings

1. **速度commandは歩容を一意に指定しない。** 1.2 m/sの速度追従に成功してもRUN contact cycleを維持できる。速度は「どれだけ速く」を制御するが、「どの接触様式で」を必ずしも決めない。
2. **WALKとRUNは同速度でも分離したstate/action manifoldを持つ。** AUROC約0.999986、action L2 2.957、局所到達0/1024は、単なるclassifier名の違いではなくdynamical attractorの差を示す。
3. **独立gait latentにより一つのmean actorが両attractorを選べる。** scalar一変数だけで同一速度のWALK/RUNを100%出し分け、双方向toggleも成立した。
4. **主問題はnetwork capacityよりoptimization pathとbasinだった。** 同じarchitectureはendpointを同時表現できたが、walk parentからRUN、run parentからWALKへの局所PPO pathは失敗した。
5. **teacher full stdはclosed-loop安全性を保証しない。** stdは学習時の探索parameterであり、完成歩容のruntime covarianceではない。gait別calibrationが必要だった。
6. **soft KL anchorだけでは長期PPOでgait semanticsを守れなかった。** rewardが直接表現しない意味は、一更新preflightを通っても累積driftした。
7. **教師ありendpoint統合は広い単一checkpoint動作列に有効だった。** endpoint、toggle retention、bounded DAggerを組み合わせることで低速WALKからRUN、復帰WALK、実用的停止まで拡張できた。
8. **strict STAND gateと物理的静止は分けて解釈すべきである。** 専門positive controlもほぼ停止しながらcontact gateを通らなかった。評価定義の厳格さを隠さず、実用停止との違いを報告する必要がある。

## 8. What was achieved

- 一つのcheckpoint / actor / mean network / gait-conditioned Gaussian head
- 同じ1.2 m/sでWALK/RUNを明示選択
- WALK→RUN→WALKの双方向遷移
- WALK 0.6/0.8/1.0/1.2 m/s
- RUN 1.2/2.4/2.6 m/s
- STAND→WALK
- RUN加速・減速
- 走行後の復帰WALKと平均約0.055 m/sまでの実用的停止
- runtime teacher/expert/router/switch/blend 0

## 9. What was not achieved

- strict flight-zero STAND
- strict final double-support
- integrated sequenceのformal gate
- continued PPO semantic retention
- scratch RLだけによる全歩容獲得
- 実機検証、Sim2Real、旋回、階段、不整地

## 10. Claims that must not be made

「全タスクを完全クリア」「完全静止」「scratch RLだけで全て獲得」「汎用ヒューマノイド制御」「実機検証」「Sim2Real達成」は主張しない。正確な主張は、Isaac Labシミュレーション上で一つのcheckpointにWALK/RUNを統合し、同一速度のgait選択、WALK→RUN→WALK、低速歩行から走行・再減速・実用的停止を示した一方、strict static-contact STANDは未解決、である。

## 11. Reproducibility

| 項目 | 値 |
|---|---|
| OS / shell | Windows / PowerShell |
| Python | 3.12.13 |
| Isaac Sim | 6.0.0-rc.59+release.41464.5f2772bc.gl |
| Isaac Lab package | 6.1.11 |
| Isaac Lab checkout | `ffff603eafc6b74264a5261cc0183d6a65390d78` |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 5090 Laptop GPU |
| RSL-RL | 5.4.1 |
| Robot asset | `G1_MINIMAL_CFG`, `${ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1_minimal.usd` |
| Observation/action | 124D / 37D (original observation 123D + gait command) |
| Gait-core SHA | `04b43e5497bc35e2d00fa4476f9120f9439e0953283c69cf8ca1e9635dedd121` |
| Final sequence SHA | `66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698` |
| Formal evaluation seeds | 各stageの`diagnostic_seed_manifest.json`およびresult JSON |
| Closure video seed set | 20270021–20270040 |
| Recording seed | `closure/video_seed_selection.json` |

再現コマンドは`results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/closure/reproduction_commands.ps1`、全artifact pathは`final_artifact_manifest.json`に保存する。large raw rolloutや一時tensorはGit管理対象外だが、tracked summary、checkpoint SHA、command、seed contractで検証可能にした。

## 12. Research closure

exp_012は、single-policy WALK/RUN統合という中心研究課題について十分な肯定的結果を得た。strict STANDと継続PPO retentionは独立した未解決課題として残る。既存専門STANDも同じstrict gateを通らず、soft-anchor/optimizer経路も因果診断を尽くした。追加の局所調整は費用対効果が低いため、本実験をクローズする。

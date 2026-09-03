# OpenDuckMini exp_003 Closure Report

作成日: 2026-08-02  
分類: **`CLOSED_NO_GO`**  
対象: OpenDuckMini calibrated-walk / omnidirectional single-policy research  
採用simulation parent: **v52 hybrid controller package**  
実機投入: **禁止**

## 1. Executive Summary

exp_003は、OpenDuckMiniに単一方策で停止・前後・左右・yaw・複合指令を追従させることを目標に進めた研究系列である。最終的に、単一omnidirectional policyは採用条件を満たさず、研究系列を再現可能なnegative resultとして閉じる。

この系列で最も重要だったのは、最初に見えていた「方策の失敗」の相当部分が、実は評価経路、teacher routing、scene、GPU batch physics、checkpoint境界、telemetry境界の問題と混ざっていたことである。そこで、性能改善を続ける前に、評価配線と学習基盤を段階的に監査した。

監査後にも残った本質的な問題は次の三点である。

1. v59のyaw overshootは評価バグだけではなく、`command_progress`が指令以上のyaw rateを報酬上有利にするobjective conflictを含んでいた。
2. objectiveをbounded化しても、yaw左右非対称、forward retention、stochastic fallを同時には改善できなかった。
3. C04 right-lateralを狙った局所的なgradient/advantage介入はoffline契約どおりに動作しても、on-policy closed-loopの状態分布変化により、転倒、C04、C09、replicate varianceを安定して改善できなかった。

最終的な科学的結論は以下である。

> C04 negative gradientへの局所介入はoffline契約どおり機能したが、on-policy closed-loopでは状態分布が変化し、転倒・C04・C09・replicate varianceを悪化させた。固定batch上の局所勾配修正だけでは、単一方策の全方向歩行を安定化できなかった。

これは「何も得られなかった」という結論ではない。exp_003は、評価equivalence、完全checkpoint、GPU MJXのbatch-only divergence、statistical resume、CUDA telemetry crash、batch-1 exact paired evaluation、command-conditioned gradient分解という、今後のlocomotion研究に再利用できる診断契約を残した。

## 2. Scope and Non-Claims

このclosureが主張するもの:

- exp_003のsingle-policy omnidirectional系列を`CLOSED_NO_GO`とする。
- v52 hybrid packageだけをadopted simulation parentとして維持する。
- v59、v60、reduced-LR、advantage/gradient介入checkpointを採用しない。
- 実機投入を許可しない。
- negative resultと診断基盤を次研究へ引き継ぐ。

このclosureが主張しないもの:

- v52が実機で歩けること。
- v52が全方向commandをformal acceptanceしたこと。
- 後退・横移動・全yaw範囲が実用上supportedであること。
- reward、PPO、MJX、fresh Adamのいずれか一つだけが全失敗の原因であること。
- すべての後続介入artifactが現在のGit treeに完全保存されていること。

後期pilotの一部は研究ログ上の確定結果が残る一方、対応artifact directoryが現在のworkspaceに存在しない。したがって本報告では、repository artifactで再検証できる結果と、確定済み研究記録から継承した結果を区別する。

## 3. Adopted Parent: v52 Hybrid Controller

v52は単一の「v52 neural checkpoint」ではない。構成は次のhybrid packageである。

```text
v45 actor checkpoint (step 47,349,760)
+ corresponding critic / normalizer lineage
+ v50 straight reverse periodic profile
+ v52 reverse-left and reverse-right periodic profiles
+ teacher/residual routing
+ calibrated/backlash MuJoCo scene
+ calibrated joint limits and action composition
```

v52 actor ONNXの既知SHA-256 prefixは`52205e...f3c`である。v52を今後使用する場合はactorだけでなく、reverse profiles、routing、scene、normalizer、observation/action contractを一体として扱わなければならない。

正式扱い:

```text
controller: v52 hybrid controller package
status: adopted simulation parent
formal hardware qualification: false
real-hardware deployment: prohibited
```

## 4. Version Lineage Was Not Linear

v54からv59は、単純な一本の継続学習ではなかった。

- v54はv22 step 10,485,760を親とする別branch。
- v55、v56、v57、v58は、それぞれv45 step 47,349,760を親にしたbranchを含む。
- v57はv56のcontinuationではない。
- v59はv58 policy/value/normalizerをrestoreしたが、optimizerはfresh Adam、interaction counterも再開始した。
- v58 60,293,120 interactionsとv59 41,779,200 interactionsの合計102,072,320は、単一checkpointのstepではない。

このため、version番号だけを見て「v54→v55→…→v59と一項目ずつ改善した」と解釈することはできない。親checkpoint、optimizer state、source tree、scene、teacher、sampler、evaluation contractをmanifestで固定する必要があることが明確になった。

## 5. Phase-by-Phase Research History

### 5.1 Calibrated baseline and reverse specialization

初期段階では、実機で手動取得したjoint zeroと可動域をMuJoCo modelへ反映し、official gait/referenceを安全域へscaleした。backlashを含むcalibrated sceneと、runtimeで利用するaction rangeを揃えた。

後退は単純なactor出力だけでは成立せず、periodic referenceとactor residualを組み合わせるhybrid構成へ移行した。v49/v50で後退直進、v52で左右後退turn profileを整備した。ここで得たv52 packageが、後続のsimulation parentとなった。

解決したこと:

- raw encoderとmodel-space zeroを分離。
- 左右mirror方向とoperational joint rangeを明文化。
- actor単体ではなく、periodic profileとresidual routingを含むcontroller packageとしてfreeze。

残ったこと:

- 実機での低速単独関節駆動、IMU bias、電源負荷、吊り下げpolicy試験は未完了。
- v52のformal practical capability rangeは未確定。

### 5.2 Omnidirectional branches v54–v59

停止、前後、左右、yaw、複合commandを一つのactorへ統合するため、continuous command sampler、teacher/residual、domain randomization、head commandを含むPPO学習を進めた。

問題は、nominal 6秒videoとformal 15秒JSON評価で結果が大きく異なったことである。当初は方策のrobustness不足に見えたが、監査により複数のcontract mismatchが判明した。

### 5.3 Legacy evaluation mismatch

trainingとlegacy evaluationには次の差があった。

- training: `scene_flat_terrain_backlash_calibrated.xml`
- formal JSON: `scene_flat_terrain.xml`
- training: reverse時にperiodic teacher target + actor residual
- formal JSON: scene-name gateによりreverse teacherが無効
- training: backlash observation、noise、delay、randomized reset
- legacy evaluation: これらの一部が欠落
- evaluation: positive-yaw時に`vy -= 0.06`という未記録補正
- nominal video: calibrated hybrid routing、6秒、home pose、外乱なし
- formal JSON: direct actor、15秒、randomized initialization

したがって、legacy評価でのreverse staticや強外乱failureを、そのままactor能力の失敗とは扱えなかった。

対応:

- deterministic controller parityを確認。
- stochastic sample injection parityを15/15で確認。
- GPU MJX fixed-input one-step repeatabilityを確認。
- training-equivalent scene、teacher routing、observation、motor compositionを使うcorrected GPU MJX evaluatorを作成。
- CPUとGPUの性能結果を混在させない契約へ変更。

### 5.4 Corrected v59 diagnostic

v59 step 33,423,360を、19 commands × 5 seeds × 15秒で再評価した。

Condition D:

```text
95/95 non-termination
TRACKING_OK       2/19
LINEAR_UNDERSHOOT 7/19
YAW_OVERSHOOT     5/19
MIXED_FAILURE     5/19
```

Condition S:

```text
28/95 falls
12/19 commands had at least one fall
9 falls before the scheduled perturbation
19 falls after it
```

corrected pathにより、legacy経路が作っていた偽のreverse-static評価は除去された。一方で、次は本物のfailureとして残った。

- linear undershoot
- backward start/tracking difficulty
- backward lateral/yaw drift
- yaw-only steady overshoot
- stochastic falls

yaw-onlyは15秒間の初期過渡ではなく定常的に過大だった。

```text
command ±0.600 rad/s
left  +1.291 rad/s = 2.152x
right -1.358 rad/s = 2.263x
```

### 5.5 Yaw objective conflict

reward auditで、dense yaw関連項は概念的に次を含んでいた。

```text
-20 * (yaw_actual - yaw_command)^2
+100 * yaw_actual * yaw_command
```

この合成項のstationary optimumは`yaw_actual = 3.5 * yaw_command`である。offline counterfactualでも、2.0x yawは1.0xより約`+18.801/step`有利で、yaw関連全termとempirical totalの最大は3.5x付近だった。

ここで初めて、yaw overshootを単なる評価配線やpolicy randomnessではなく、少なくとも一部`objective-induced behavior`と分類できた。

### 5.6 v60 bounded yaw causal pilot

旧objectiveのmatched controlと、command-centered bounded yaw objectiveを同一親・同一seed・同一予算で比較した。

結果は`STOP_AT_1M`。

- Treatmentは右yawを改善。
- 左yawは0.504xへundershoot。
- 左右response差は0.573。
- yaw MAE改善は13.2%に留まった。
- forward retentionが悪化。
- stochastic yaw-only fallはControl 7/10からTreatment 8/10へ悪化。

結論:

- unbounded yaw objectiveは確かに問題だった。
- しかしyaw項一つをbounded化するだけでは、learned controller全体のclosed-loop分布を安全に再編できなかった。
- linear/backward/fall問題が同時に解決したとは主張できない。

### 5.7 Instrumented training harness

v60でtraining中のcommand exposure、optimizer state、RNG、途中checkpointが不足していたため、state-complete training harnessを構築した。

取得・保存できたもの:

- actor、critic、normalizer
-完全なOptax/Adam state
- learner/rollout/evaluation RNG
- complete env stateとper-env RNG
- command/head、delay buffer、teacher phase
- exact domain-randomized MJX model
- intermediate checkpoint schedule
- exact command exposureとPPO effective sample counts
- gradient/update/Adam diagnostics

checkpoint payloadは約18.16 MBで、canonical bytesのround-tripはbit-exactだった。

しかし、4 update uninterruptedと2+save/load+2 updateが一致せず、最初のcontroller-visible差は初回physics step後のnormalized observationまで遡った。この時点でAdam moment差を原因扱いせず、physics pathの根因監査へ進んだ。

### 5.8 Batched MJX divergence

fixed motor target、fresh serialized input、donationなし、resetなしでもbatched GPU MJXが分岐することを確認した。

```text
batch 1: 20/20 bit-exact
batch 2: 10/20 divergent
batch 4: 19/20 divergent
each environment unbatched: 20/20 bit-exact
```

最初の差:

```text
environment index: 0
stage: fwd_position -> smooth.crb -> _impl/crb
element: [1, 2]
2.177311897277832 -> 2.177312135696411
absolute error: 2.384185791015625e-7
```

上流はreverse body-tree accumulationの`jax.ops.segment_sum`、HLOではscatter-addだった。contact/constraint構造が変わる前に分岐していた。

分類:

```text
BATCH_ONLY_MJX_DIVERGENCE
TRUE_BATCH_NONDETERMINISM_CONFIRMED
```

対応:

- serialized payloadとbatch-1 one-stepには引き続きbit-exactを要求。
- batch 2以上のGPU rollout/gradientには統計的再現性を要求。
- paired性能評価にはbatch size 1を使用し、episode-start contextを完全固定。

### 5.9 Resume and process-boundary audits

初期の20 uninterrupted / 20 fresh-process-resume比較は`STATISTICAL_RESUME_FAIL`だった。actor/critic delta、Adam second moment、entropy、return、tracking RMSEに事前閾値を超えるstandardized effectが出た。

その後、H/L/C/Fの4 modeを各20 trialで分解した。

```text
H: hot uninterrupted
L: same-process save/load
C: same-process load + recompile
F: fresh-process resume
```

80/80 trialが完走し、midpoint payloadはbit-exact。HとFを含むprimary metric差は実質閾値内で、process boundary固有の安定したshiftは再現しなかった。追加traceでnext observation、controller output、termination/fallまで閉じ、最終分類は`NO_STABLE_BOUNDARY_EFFECT`となった。

運用上は、原因が完全に消えたと過大解釈せず、null pilotをuninterruptedに限定し、mid-run checkpoint loadを使わない契約とした。

### 5.10 CUDA telemetry crash

production-size null continuationを6本開始したが、6/6が100k–150k interactionsで異常終了し、4本では`libcuda.so.1.1 SIGSEGV`が確認された。

host-boundary ablation:

```text
B0 telemetry off / checkpoint off: 3/3 completed 150k
B1 telemetry on  / checkpoint off: 0/3, exit 139 at 100k
B2 telemetry off / checkpoint on : 3/3 completed 150k
B3 telemetry on  / checkpoint on : 0/3, exit 139 at 100k
```

さらにtransfer isolationを行った。

```text
T0 no device_get, no write:            3/3 completed
T1 host-only write:                    3/3 completed
T2 direct telemetry device_get only:   0/3 at 100k
T3 detached 68-byte device_get only:   0/3 at 100k
T4 detached device_get + host write:   0/3 at 100k
```

分類:

```text
DIRECT_GPU_TRANSFER_RELATED
MATERIALIZED_DEVICE_TRANSFER_CRASH
```

ファイル書込み、checkpoint、明確なmemory growthは主因ではなかった。production training contractを次へ変更した。

```text
per-update device_get prohibited
GPU-derived telemetry callback prohibited
host-only heartbeat allowed
synchronous checkpoint save allowed
training uninterrupted
evaluation in a separate process after training
```

### 5.11 Exact paired retention

batched evaluationのvarianceを排除するため、batch-1、90 identical contexts、13 policies、1170 episodesのexact paired evaluationを実施した。context hash mismatchは0だった。

parent v52:

```text
fall 36/90
Condition D 6/45
Condition S 30/45
```

50k checkpoint aggregate:

```text
SF 31 / FS 37 / FF 179 / SS 293
net fall change -1.11 percentage points
fixed-horizon tracking -1.39% vs parent
```

100k checkpoint aggregate:

```text
SF 22 / FS 56 / FF 160 / SS 302
net fall change -6.30 percentage points
fixed-horizon tracking -3.45% vs parent
```

aggregate collapseは否定されたが、5 checkpointsが個別new-failure Gateへ違反し、right-lateralとforward+yawにcommand-specific instabilityが残った。same-seed backend trajectory varianceも大きく、fresh Adam continuationを安全とは判定できなかった。

### 5.12 Reduced learning-rate pilot

current LR `3.0e-4`と、唯一の差分を50% LR `1.5e-4`にしたmatched pilotを比較した。

100k aggregate:

```text
tracking Treatment-Control: -0.03%
95% paired CI: [-0.01473, +0.01361]
new-failure: Control 6.11%, Treatment 5.19%
motor-target delta: 0.012961 -> 0.008482 (-34.6%)
```

しかしcommand別には符号が分かれた。

```text
C04 right-lateral response:
Control 0.01686, Treatment 0.01126
10%以上低下run: 2/6 -> 5/6

C09 forward+yaw response:
Control 0.01086, Treatment 0.11154
10%以上低下run: 3/6 -> 1/6
```

Treatmentはmotor driftとC09 varianceを下げた一方、C04 varianceを355.1%増加させた。結論は`REDUCED_INITIAL_LR_INCONCLUSIVE`。

### 5.13 Command support and gradient conflict audit

C04とC09のtraining exposureをcontinuous samplerから再構成した。

```text
C04 neighborhood: 41,944 / 2,000,000 = 2.0972%
C09 neighborhood: 25,029 / 2,000,000 = 1.25145%
C04/C09 exposure ratio: 1.6758
```

C04は露出不足ではなかった。fixed batch actor gradientも直接競合していなかった。

```text
C04 norm 2.9848
C09 norm 2.3058
global cosine +0.3459
context cosine median +0.4340
all-layer range +0.2861 to +0.4786
```

代わりにadvantage signalの非対称が見つかった。

```text
C04 positive ratio 13.75%, |advantage| 0.9446
C09 positive ratio 35.25%, |advantage| 1.3366
```

分類: `ADVANTAGE_SIGNAL_IMBALANCE`。

### 5.14 Advantage normalization and bounded scaling

command regimeごとのmean-zero/unit-variance normalizationは、C04だけでなくC09をさらに増幅した。

```text
C04 gradient +122.6%
C09 gradient +203.7%
maximum layer ratio 3.2718
```

offline Gateで停止し、学習は実施しなかった。

次のbounded RMS scale-only補正も、C04とC09がともに最終scale 1.20へ飽和し、選択的補強にならなかった。さらにadvantage全体meanが`-0.09320`へずれたためoffline Gateで停止した。

ここで一般的なadvantage rescaling branchを閉じた。

### 5.15 Positive/negative gradient decomposition

C04の「advantageが弱い」を、positive改善信号とnegative抑制信号へ分解した。

```text
C04 positive: 55/400 = 13.75%
C04 negative: 345/400 = 86.25%
C04 positive gradient norm: 1.16696
C04 negative gradient norm: 2.71340
negative/positive: 2.325
positive-negative cosine: +0.0293
```

C09:

```text
positive 141/400 = 35.25%
negative 259/400 = 64.75%
positive gradient norm 1.57234
negative gradient norm 1.78625
```

PPO clip fractionはC04/C09、positive/negativeすべて0%。C04 negative dominanceはclip非対称ではなく、negative sampleが6.27倍多く、negative stateのpolicy sensitivityが約25%高いことで説明された。

分類: `NEGATIVE_ADVANTAGE_DOMINATED_C04`。

### 5.16 C04 negative contribution cap

C04 regimeのnegative actor-loss contributionだけを`0.65`倍する介入を設計した。係数は`1.50 / 2.325 = 0.645`から事前固定した。

offlineでは、C04 positive、C09、他regimeを維持しながらC04 negative gradientだけを目標範囲へ抑える契約が成立した。これは介入コードが意図どおり働いたことを示す。

しかしon-policy closed-loopでは、C04改善、C09 retention、fall、replicate varianceを同時に満たせず、`C04_NEGATIVE_ADVANTAGE_CAP_NOT_SUPPORTED`となった。

この結果が重要である。固定batch gradientで局所的に「正しい」変更をしても、その更新後policyが訪れる状態分布、teacherとの相互作用、contact transition、advantage再分布が変わる。したがってoffline projectionの改善をclosed-loop因果効果と同一視できない。

## 6. Problem → Intervention → Outcome Matrix

| 問題 | 仮説 | 実施した対策 | 結果 |
| --- | --- | --- | --- |
| reverse static | actor能力不足 | evaluation routing監査 | legacy evaluatorがteacherを外していた。評価誘発failureを除去 |
| CPU/GPU差 | backendは等価 | controller-visible one-step比較 | 非等価。性能表を分離しGPU MJXを診断backendへ |
| v59 yaw overshoot | policy tuning不足 | reward counterfactual | dense objectiveが3.5x yawを選好 |
| yaw objective conflict | bounded化で解決 | v60 matched causal pilot | 右改善、左undershoot、retention/fall悪化。STOP_AT_1M |
| exact resume failure | checkpoint欠損 | state-complete harness | payload bit-exact化。ただしbatched rollout差が残る |
| rollout divergence | donation/RNG/reset | fixed action、fresh input、batch ladder | batch-only MJX scatter-add divergenceを特定 |
| resume distribution shift | serialization/process境界 | H/L/C/F 80-trial decomposition | stable boundary effectは再現せず、uninterrupted運用へ |
| production crash | checkpoint/file I/O | B0–B3 ablation | telemetry有効だけ100k crash |
| telemetry crash | JSONLまたはbuffer保持 | T0–T4 isolation | 小さなdevice_get自体と関連。per-update GPU telemetry禁止 |
| fresh Adam instability | aggregate evaluation noise | batch-1 exact paired contexts | aggregate collapseなし、command-specific instabilityあり |
| C04/C09 tradeoff | LR過大 | LRを50%へ | C09改善、C04悪化、variance再配分。inconclusive |
| C04弱化 | command exposure不足 | sampler support監査 | C04 exposureはC09より多く、否定 |
| C04/C09 conflict | gradient競合 | fixed-batch cosine | 全layer正、直接競合を否定 |
| C04 signal imbalance | group normalization | regime mean/std normalization | C09を過増幅。offline FAIL |
| C04 magnitude不足 | bounded RMS scaling | scale-only 0.75–1.50 | C04/C09同時飽和、mean shift。offline FAIL |
| C04 negative dominance | negative contribution cap | C04 negativeのみ0.65 | offline成功、closed-loop失敗 |

## 7. Why the Local Interventions Did Not Transfer

固定batch監査には価値があるが、次の限界があった。

1. **On-policy distribution shift**  
   一回のactor updateがactionを変え、そのactionがcontact、support、termination、次observationを変える。次のbatchはもはや固定batchと同じ分布ではない。

2. **Rare but decisive transitions**  
   平均gradientやtrackingが改善しても、support transition直前の少数stateでactionがずれるとfallとなる。fall後を含むfixed-horizon metricには大きな影響が出る。

3. **Teacher/residual coupling**  
   actorだけのgradientを局所修正しても、periodic teacher、delay、motor clipping、head state、phaseとの合成結果は線形ではない。

4. **Shared parameters across regimes**  
   C04とC09 total gradient cosineが正でも、state-dependent Jacobian、joint-output感度、以後のtrajectory分岐まで一致するわけではない。

5. **Backend trajectory variance**  
   GPU batched MJXは最初の`crb` scatter-addから微小差を生み、長いon-policy学習ではtrajectoryとgradientの差へ増幅する。paired batch-1評価は評価varianceを除去できるが、training trajectory varianceは残る。

6. **Fresh optimizer transition**  
   pretrained actorへfresh Adamを適用すること自体が新しいoptimization phaseであり、元学習のmoment/critic/on-policy distributionを継続するものではない。

## 8. Scientific Conclusions

### 8.1 Supported

- legacy evaluation mismatchは、v59 failureの一部を偽っていた。
- corrected GPU MJXでもlinear undershoot、yaw overshoot、drift、fallは残った。
- v59 yaw overshootは少なくとも一部objective-inducedだった。
- yaw objective一項目の修正だけではretentionを維持できなかった。
- GPU MJX batch 2以上は現runtimeでbit-exactではない。
- checkpoint payload自体はbit-exactに保存可能。
- per-update GPU telemetry transferはproduction-size CUDA crashと強く関連した。
- C04 exposure不足とC04/C09直接gradient conflictは支持されなかった。
- C04 actor gradientはnegative-advantage contributionに支配されていた。
- offline局所gradient修正はclosed-loop改善を保証しなかった。

### 8.2 Not supported

- v59を実機投入できるという主張。
- v60 bounded yawを採用できるという主張。
- learning rate半減だけでC04/C09 tradeoffが解けるという主張。
- group advantage normalizationまたはbounded scalingの採用。
- C04 negative capの採用。
- 単一omnidirectional policy研究を同じ介入系列で継続する合理性。

### 8.3 Remaining unknowns

- v52のformal 20-seed/30-second practical capability envelope。
- 実機のIMU bias、loaded voltage sag、delay、backlash、velocity limit。
- independent right-lateral expertやsmall bounded residual headの有効性。
- optimizer moment reconstruction等、pretrained continuation専用transitionの有効性。
- 別JAX/MJX/CUDA versionでscatter-add divergenceとdevice_get crashがどう変わるか。

## 9. Final Decision

```text
exp_003 omnidirectional single-policy: CLOSED_NO_GO

reason:
local/offline gradient interventions did not translate into
stable closed-loop improvement

adopted simulation parent: v52 hybrid controller package
v59: diagnostic_not_qualified
v60 and later pilots: not adopted
real-hardware deployment: prohibited
```

PPO actor-lossのadvantage操作系列も終了する。同じsingle-policyへさらに局所loss tweakを重ねない。

現実的な次の選択肢は、いずれか一項目だけを独立研究として扱うことである。

1. independent right-lateral expert
2. C04専用small bounded residual
3. omnidirectional single-policyを終了し、v52 simulation parentを維持

後退についても、実用化が必要なら独立expertとして別contractを持たせる。実機試験へ進む前には、14関節低速単独駆動、IMU軸/bias、電源負荷、吊り下げpolicy、実測delay/backlash/speedのbring-up Gateを完了する必要がある。

## 10. Public Communication Contract

公開動画・投稿では次を明示する。

- 映像はMuJoCo simulation。
- v52はsimulation parentであり、実機qualifiedではない。
- v59はdiagnostic policyで、not qualified / not deployed。
- exp_003の成果はomnidir policyの完成ではなく、negative resultと診断方法。
- 実機歩行動画であるような表現を使用しない。

LinkedIn用動画はこのcontractに従い、画面内にsimulation-onlyと各versionのstatusを常時表示する。

## 11. Provenance Notes

- Report-generation workspace branch: `exp/openduckmini-statistical-resume`
- Report-generation HEAD: `8bf5dd71f9046019b1c1a1b284ba9e52d5394012`
- Existing worktree contained unrelated user modifications and untracked research artifacts; they were not altered.
- No new training, optimizer update, real-hardware evaluation, push, tag, or policy promotion was performed for this closure/video task.


# exp_011 Go2 fresh-process counterfactual replay

## Conclusion

Classification: **COUNTERFACTUAL_REPLAY_CONTRACT_PASS_SLIP_NOT_CONTROLLABLE**

Pilot readiness: **PILOT2_NOT_READY**

Next: **close scalar slip-reward tuning route and diagnose gait / actuator / contact-model compatibility**

## Same-lifecycle failure

同一Isaac lifecycleのordinary reset前後でmanager、heading controller、
contact sensor、robot/actuator public state、Python/NumPy/torch RNGを監査した。
一致しない公開fieldは
action.current, action.previous, contact.age, contact.history_speed, contact.normal_force, environment.common_step_counter, heading.command, heading.raw, heading.reference, reward.raw_slip_score, robot.applied_effort, robot.joint_position。
PhysX solver/contact warm-start cacheは`UNEXPOSED_PHYSX_INTERNAL_STATE`であり、
reset実装は変更していない。

post-reset同士ではPython/NumPy/torch CPU/CUDA RNG hash、root/joint初期値、
action/controller reset値は一致した。一方、contact boolean/history/normal
force、environment common-step counter、reset buffer、applied effort、
last applied joint targetは残留または差異を示した。contact solver warm-start
等の内部cacheはAPIから取得不能であり、`UNEXPOSED_PHYSX_INTERNAL_STATE`
として記録した。

## Fresh-process contract

1 OS process = 1 Isaac application lifecycle = 1 environment creation =
1 reset = 1 episode = at most 1 action variant。process concurrencyは1。
baseline preflightは75/75 runs、
gateは**PASS**。
formal branch eligibilityは
100.0%、
variant validityは
100.0%。
process manifestは75 baseline + 2,400 primary + 960 linearity =
3,435 fresh processesを記録し、全runのstatusは`COMPLETE`。各runは
1 process / 1 Isaac lifecycle / 1 environment / 1 episode / 1 variantで、
serial concurrency=1である。

Canonical traceはfixed dtype raw bytesからSHA-256を作成し、root/joint、
48D observation、policy mean action、contact/contact age、heading controller、
termination stepを比較した。state setter、snapshot restore、physical state copyは0。

## Counterfactual and controllability

Formal branches: 100。
Primary perturbationは12 dimensions × ±0.02、horizon 1/2/4/8 steps。
20% branchで±0.01/±0.04を追加した。

Local classification: **SLIP_NOT_LOCALLY_CONTROLLABLE**。
Overall improving branch rate:
5.0%。
Trade-off: **SLIP_REDUCTION_WITHOUT_CAPABILITY_TRADEOFF**。

| speed (m/s) | branches | improving | rate |
|---:|---:|---:|---:|
| 0.2 | 20 | 4 | 20.0% |
| 0.4 | 20 | 0 | 0.0% |
| 0.6 | 20 | 0 | 0.0% |
| 1.2 | 20 | 0 | 0.0% |
| 2.0 | 20 | 1 | 5.0% |

安全な改善branchは0.2 m/sで4/20、2.0 m/sで1/20のみであり、
0.4/0.6/1.2 m/sでは0/20。全体5/100は事前gateの10%未満である。
17 variantsはslipを20%以上下げたが、新規contact loss、fall、
saturation、speed/heading hard trade-offは支配的でなかった。問題は
trade-offよりも、安全基準を満たす局所改善方向の希少性である。

| joint | improving variant rate | mean slip reduction | Δspeed error | Δheading |
|---|---:|---:|---:|---:|
| FR_thigh | 1.5% | -0.037 | -0.0000 | -0.0000 |
| FR_calf | 1.5% | -0.009 | -0.0000 | 0.0000 |
| FR_hip | 1.0% | -0.051 | -0.0000 | -0.0000 |
| RL_hip | 1.0% | -0.020 | -0.0000 | 0.0000 |
| FL_calf | 1.0% | -0.009 | -0.0000 | -0.0000 |
| FL_hip | 0.5% | -0.010 | -0.0000 | -0.0000 |

20% branchの±0.01/±0.04監査は960/960 valid。response sign consistency
90.4%、
magnitude monotonicity
85.6%で、
±0.02が一律に極端な非線形領域にある証拠はなかった。

## Gradient agreement

Stage 12の|g_slip|/|g_base|は0.001454、base/slip cosineは-0.329、
minibatch consistencyはPASS。fresh-process central finite differenceとの
median cosineは0.164、
mean sign agreementは60.0%、
gateは**FAIL**。
speed別cosineは0.2=-0.172、0.4=0.321、0.6=0.164、
1.2=0.459、2.0=-0.043で一貫せず、平均sign agreementは60%だった。

Gradient-calibrated weight計算値は0.769418。current以上、4倍以下、
0.5以下の事前制約に対する判定はFalseである。
局所可制御性gateとgradient agreementがともに不合格であり、
提案weightも上限外なので、単純なscalar weight増加の根拠はない。

## Protection

Stage 1〜12、公式/Stage 4/Stage 7/Stage 11 checkpoint、Stage 10
controller、両評価protocol、capability manifest、production artifact、
Isaac Lab coreは変更していない。production PPO update=0、reward
optimization=0、state injection=0、remote push=false。

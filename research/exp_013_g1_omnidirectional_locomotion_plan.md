# exp_013 Unitree G1 single-policy omnidirectional locomotion plan

## Status

```text
EXPERIMENT: exp_013_unitree_g1_single_policy_omnidirectional_locomotion
STATUS: ACTIVE
CURRENT STAGE: Stage 0 complete
```

## Research contract

最終runtimeは `one checkpoint / one actor / one Gaussian policy head` とする。actorへの外部入力は、胴体座標系の次の4変数だけである。

```text
c_t = (vx_cmd, vy_cmd, yaw_rate_cmd, gait_cmd)
gait_cmd = 0: WALK
gait_cmd = 1: RUN
```

関節actionは常に一つのactorから生成する。expert router、方向別checkpoint、歩行・走行checkpoint切替、action blending、scripted movement controller、hard-coded joint action、transition/residual expertを最終runtimeへ持ち込まない。

## Stage 0 decision

Stage 0の主分類:

```text
EXP013_PARENT_HAS_PARTIAL_DIRECTIONAL_GENERALIZATION
```

command contractとanchorは正常である。現在の親方策は前方近傍を中心に、複数の斜め・旋回commandへ部分汎化した。ただし後退、横、後方斜め、高速RUN旋回では追従誤差とslipが大きい。

次に実施する方式は一つだけ:

```text
Phase W1:
retain supported directions and train missing WALK sectors
```

## Phase W1 — all-direction WALK specialist

前後、左右、全斜め、その場旋回、移動しながら旋回を一つのWALK specialistへ学習させる。Stage 0でPARTIALLY_SUPPORTED以上だったsectorをretention anchorとして扱い、missing sectorを重点化する。速度上限は方向別に安全側へ設定してよい。runtime契約へrouterを追加しない。

Exit:

- 16方向、複数速度、複数yaw rateで固定matrixを完走
- fall、slip、impact、saturation gateを満たす
- translation directionとbody yawの独立性を確認

## Phase W2 — WALK dynamics

方向変更、急反転、任意方向への発進、減速、実用的停止、60秒continuous random WALK commandを追加する。command rampはminimum-jerkを基本とするが、actor actionをscriptで補わない。

## Phase R1 — forward RUN extension

既知の前進RUNを保持しながら、走行中の左右旋回と前方斜めRUNを追加する。WALK能力とWALK↔RUN anchorを各更新で監視する。

## Phase R2 — remaining RUN sectors

後退RUN、横方向RUN、後方斜めRUNを追加する。方向別に安全な速度上限を設定してよい。WALKより厳しいimpact/slip/saturation guardを適用する。

## Phase I1 — single actor integration

WALK specialist、RUN specialist、transition dataを一つのcommand-conditioned actorへ統合する。

```text
Final runtime:
one checkpoint
one actor
one Gaussian policy head
no router
```

専門方策は学習・data sourceとしてのみ使用でき、final runtimeでaction sourceにならない。

## Phase I2 — omnidirectional gait switching

全方向でWALK↔RUNを切り替える。速度と方向、yaw rate、gaitを同時に変化させたtransitionを評価する。

## Phase E1 — fixed evaluation

16方向、複数速度、複数旋回速度の固定matrixを、deterministic episodeで評価する。body-frame vector tracking、yaw tracking、gait、safety、左右対称性を同一契約で保存する。

## Phase E2 — continuous evaluation

60秒random continuous commandを100 episode実施する。command interval、速度、方向、yaw、gait switchをseed付きで保存し、failure modeを再現可能にする。

## Phase D1 — gamepad teleoperation

gamepadは4変数commandだけを生成する。関節action、方向別motion、transition actionを生成しない。最終checkpointを更新せずにruntime確認を行う。

## Persistent protection rules

- exp_005〜exp_012、exp_012 closure、既存checkpoint/optimizerを変更しない
- Isaac Lab/RSL-RL core、G1 asset、joint order、action scale、control/physics/PD/frictionを変更しない
- phase開始前に親hash、観測index、action contractを再監査する
- production policy更新は正式gate通過後に別承認する

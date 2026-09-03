# exp_013 Stage 0 parent directional baseline report

## 結論

主分類は `EXP013_PARENT_HAS_PARTIAL_DIRECTIONAL_GENERALIZATION`。command pipeline と anchor は正常で、親方策には前方近傍を中心とする複数の斜め・旋回条件への部分汎化がある。一方、360度を一様に追従する方策ではなく、後退・横・後方斜め・RUN旋回では大きい速度誤差、yaw誤差、dangerous slip が支配的である。

## Contract と parent

- command index (zero-based): `vx=9, vy=10, yaw=11, gait=123`
- frame: robot body
- scale/normalization: 1.0 / none
- history: none; previous actionは `86..122`
- selected parent: Stage 2Q
- SHA-256: `66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698`
- architecture: `124 -> 256 -> 128 -> 128 -> 37`

## Anchor (100 deterministic episodes/condition)

| condition | gait success | fall | vector MAE |
|---|---:|---:|---:|
| WALK 0.6 | 100% | 0% | 0.113 |
| WALK 1.2 | 99% | 0% | 0.175 |
| RUN 1.2 | 99% | 0% | 0.137 |
| RUN 2.4 | 98% | 0% | 0.210 |
| WALK->RUN | 100% | 0% | 0.073 |
| RUN->WALK | 100% | 0% | 0.090 |
| practical STOP | 100% | 0% | 0.006 |

RUN 2.4の2 episodeとRUN 1.2の1 episodeに分類/impact外れがあったが、転倒は0%でanchor regression gate（大きな崩れ）には該当しない。

## Translation

WALK 64条件中、SUPPORTED/PARTIALLY_SUPPORTEDは `22` 条件。最良は `WALK_S0.6_D337.5` (MAE 0.110 m/s)、最悪は `WALK_S1.2_D135.0` (MAE 0.704 m/s)。RUNでは同分類が `12` / 64条件で、最良 `RUN_S1.2_D000.0` (0.115)、最悪 `RUN_S2.4_D202.5` (2.080)。後退・横・後方斜めの高速RUNが最も弱い。

## Yaw / combined control

純旋回12条件の平均yaw-rate MAEは `0.537` rad/s。符号反応は複数条件で存在するが、WALK/RUNとも目標rateの定量追従は弱い。translation+yaw 160条件の分類内訳は `{'SUPPORTED': 11, 'PARTIALLY_SUPPORTED': 34, 'UNSAFE': 104, 'SAFE_BUT_UNTRACKED': 11}`。前方・前方斜めの曲線は部分的に成立するが、strafe/backward turnの独立制御は不均一で、指定6条件でも両軸を安定して満たす一貫性はない。

## Transitions / random

WALK方向sequenceは転倒 `0%`、vector MAE `0.109`。RUN方向/gait sequenceは転倒 `0%`、vector MAE `0.165`だが dangerous slip率は `100%`。

60秒WALK randomは転倒 `0%`、vector MAE `0.317`、yaw MAE `0.655`。RUN-capable randomは転倒 `0%`、vector MAE `0.786`、yaw MAE `1.124`。支配的failureはyaw-rate未追従、後方/横方向速度不足、slip、RUN gait retention低下。

## Safety

全Stage 0評価 `5380` episodeの集計で fall `0.06%`、excessive tilt `53.38%`、dangerous slip `22.96%`、impact `0.84%`、long-dwell saturation `0.00%`。平均absolute roll/pitchは `0.110` / `0.111` rad。左右対称性はmirror方向間の平均absolute vector-MAE差 `0.077` m/sで、完全対称ではない。

## Next

次に選択する方式は一つだけ: **Phase W1 — supported directionsを保持し、missing WALK sectorsを学習するall-direction WALK specialist**。Stage 0では実行していない。

# exp_011 Go2 tangential-slip reward directionality diagnosis

## 結論

Stage 12の主要分類は **SLIP_REWARD_DIRECTIONALITY_INCONCLUSIVE**、Pilot readinessは
**PILOT2_NOT_READY** である。固定rollout上ではrewardの順位付け、短い時間
lag、学習可能なslip value、弱いが一貫したactor gradientを確認した。しかし
禁止されたstate injectionを使わない同一process replayではpre-branch状態が
一致せず、局所action方向と能力trade-offを物理的に検証できなかった。この
validity failureをprecedence 1としてfail-closedした。

## Reward ranking

同じspeed・schedule phase・4足contact pattern内で0.2秒segmentを順位付けした。
lowest-slip decileのraw score平均は 0.1570、
highest-slip decileは 0.6965 だった。対応する
speed MAE平均は 0.2422 / 0.2941
m/s、base reward平均は 0.03312 /
0.02903。主要5速度のsafe low-slip segment比率は
0.2: 19.8%, 0.4: 20.0%, 0.6: 20.0%, 1.2: 19.9%, 2: 19.7%で、
**LOW_SLIP_BEHAVIOR_EXISTS** と判定した。

## Temporal structure

action-rate変化との最大相関はlag 0 step
（0.00s）、相関
0.199で、
**IMMEDIATE_OR_SHORT_LAG**。固定trajectoryからStage 11と同じ
discountでslip-only returnを計算した。

## Advantage and gradient

48→128→128→1 ELUの診断V_slipはepisode/seed 70/15/15 splitでtest R²
0.726、MAE 0.001229。診断GAEの
A_slip stdは 0.000754、A_base stdは
0.543。初期actor固定batch上で
|g_slip|=0.002674、
|g_base|=1.839659、
q_g=0.001454（SLIP_GRADIENT_TOO_WEAK）。
base/slip cosineは -0.329
（GRADIENT_CONFLICT）。100 fixed permutationsの
pairwise slip-gradient cosine中央値は
0.398
（SLIP_GRADIENT_CONSISTENT）。

Stage 11 checkpoint軌跡ではiteration 1/10/25の初期方策差がslip方向と負、
後半の一部が正だったが、validation precedenceを改善せずiteration 0が選択
された。これは「gradientが弱くbase側に支配された」ことと整合するが、
counterfactual failureのためunderweightingを因果分類には昇格しない。

## Local controllability

0.2m/sの100 branch stateでordinary resetによるsame-seed replayを監査した。
fresh process間のbaseline prefixはbitwise一致した一方、同一PhysX lifecycle
の再resetではroot最大差 5.885、joint
9.604、previous action 4.937、
contact age 106.0となり、許容1e-5を満たすvariantは
0/2だった。残り400 stateとlinearity replayは
結果を作るために続行せず停止した。state setter、teleport、別env state copy
は0である。

したがってlocally improving state rate、joint/leg/phase別controllability、
speed/heading/contact Pareto trade-offは **NOT_EVALUABLE**。invalid branchの
perturbation結果は解析から除外した。

## Classification and next action

- Classification: **SLIP_REWARD_DIRECTIONALITY_INCONCLUSIVE**
- Pilot readiness: **PILOT2_NOT_READY**
- Next: **establish a reproducible no-state-injection counterfactual replay contract**

gradient-calibrated weightは提案しない。local controllability PASSとcapability
conflict不在を確認できていないため、weight増加は許可されない。

## Protection

Stage 1〜11、公式/Stage 4/Stage 7/Stage 11 checkpoint、Stage 10 controller、
両評価protocol、capability manifest、production artifact、Isaac Lab coreは
変更していない。production PPO update=0、reward optimization=0、remote
push=false。

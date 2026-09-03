# exp_006 Unitree G1 Command System v1 最終レポート

生成日: 2026-07-23  
生成時Git revision: `49bbff1`（working-tree上の凍結成果を本レポートと同時にcommitする前の基準revision）  
対象: Isaac Lab / Isaac Sim内のUnitree G1シミュレーション。実機性能の主張ではない。

## 1. Executive Summary

exp_006は、exp_005で得た高速走行actorを壊さずに、commandで選択できる複数技能へ拡張することを目指した。最終成果`command_system_v1`は、RUN、TURN、STAND、CROUCH_SHALLOWを正式PASS技能として統合し、同一controller family内のsequenceを再現可能にした。STOPは有用なprototypeとして保存したがformal gate未達、CROUCH_DEEP、STEP_OVER、LANDは明示的にNOT_SUPPORTEDである。

本成果は単一の万能ニューラルネットワークではない。Stage 4 running base、Stage 2 standing base、学習済みskill residual、scripted primitiveをparameter-free routerが選択するhybrid command systemである。未対応requestは既存controllerを維持したまま原子的に拒否する。

## 2. 背景

exp_005ではG1の5 m/s走行を段階的curriculumで完成した。旧ハードル実験は、走行中の飛行・踏切・接触をend-to-end PPOだけで同時に解く難しさを示したため、exp_006では「成立済みbaseを保護し、技能ごとにauthorityと制御構造を選ぶ」command skill systemへ方向転換した。running-centered設計の狙いは、高価なStage 4能力を凍結したままRUN/TURN/STOPを追加することだった。

## 3. Architecture

- actor観測: legacy 123次元 + skill command 29次元 = 152次元
- action: G1 minimal articulation順の37次元position action
- Stage 4 running base: RUN、TURN、STOP prototype
- Stage 2 standing base `model_4246`: STAND、CROUCH_SHALLOW
- learned residual: RUN path補正、STOP braking residual
- scripted primitive: CROUCH_SHALLOW `scripted_shallow_v1`
- router: trainable parameterを持たない外部controller selector
- policy one-hot: 既存6技能のまま。STAND用の7番目は追加していない

controller familyは`RUNNING_FAMILY={RUN, TURN}`、`STANDING_FAMILY={STAND, CROUCH_SHALLOW}`、`PROTOTYPE={STOP}`である。

## 4. RUN

RUNはStage 4 baseへpath-localなlookahead、lateral error、curvature等を与え、RUN専用residualでcourse保持を補った。command encoder/state adapter/headをskill-local化し、後続技能の更新がRUN routeへ漏れないようにした。baseline gateから継承された独立RUN成功率は100%、fall 0%、speed error 0.1205 m/s、heading error 0.0122 rad、path lateral error p95 0.1272 mである。command counterfactualとaction差によりcommand sensitivityを確認し、後続checkpointでもtensor hashと代表state actionのbitwise一致を保った。

## 5. TURN

TURN専用residualを大きく学習する必要はなかった。凍結Stage 4 actorが既存123次元観測のlegacy yaw-rate commandへ既に安定応答したため、command systemは目標角度からbounded yaw-rateを生成し、その能力を再利用した。左右45°/90°の全カテゴリーが正式評価で100%、straight recoveryも100%、平均最終角度誤差は約0.0132 radだった。TURN後のRUN routeもbitwise不変である。

## 6. STOP

初期baselineは即時target speed=0で停止できず、成功0%、position error 0.9516 m、final speed 1.3144 m/sだった。評価器のfall帰属をSTOP windowへ修正後、STOP entryで目標位置を固定し、残距離からbraking target speedを滑らかに下げるdense braking residualを学習した。model_31は位置・速度・holdを大幅改善したがheading driftと転倒が残った。

固定heading feedbackを重ねた最良formal 50 episodeは成功82%、fall 10%、saturation failure 6%、heading mean/p95 0.1429/0.3286 rad、position error 0.1609 m、hold end speed 0.0760 m/sだった。Stage A in-rangeでは成功90%、fall 2.5%だが、正式混合条件のgateを満たさない。追加gain探索は停止し、checkpointとoverlayを再現可能なPROTOTYPEとして保存した。PASSへ格上げしていない。

## 7. STAND

Stage 4 running actorはzero commandでも周期接触切替が残り、安定STAND/CROUCH baseとして不適切だった。既存候補を比較し、Stage 2 `model_4246`をstanding baseに選定した。正式50 episodeではsettle 98%、8秒hold 98%、fall 2%、speed mean/p95 0.0067/0.0133 m/s、pelvis-height range mean 0.0018 m、flight 0%、saturation 0%、final double support 98%でgate PASS。STANDはstanding familyのhome stateであり、新しいpolicy skillではない。

## 8. CROUCH

Stage 4 running base上のresidual方式ではdepth trackingだけが改善し、周期歩行、contact loss、ankle saturation、RETURN失敗が支配した。Stage 2 baseへ切り替えた後も旧residual authority内の最大安定dropは0.02948 mで、最低要求0.08 mへ届かなかった。分類は`ACTION_AUTHORITY_INSUFFICIENT`であり、PPO iteration不足ではない。

安全なpose lookupとminimum-jerk DOWN/RETURNを用いる`scripted_shallow_v1`へ変更し、0.08–0.10 mを正式supported rangeとした。formal 50 episodeは成功、hold、return、final standすべて100%、depth error mean/p95 0.00060/0.00134 m、fall/contact/saturation 0%。0.10 m超は安全RETURNが未解決なため`CROUCH_DEEP: NOT_SUPPORTED / DEEP_CROUCH_RETURN_UNRESOLVED`として、clampせず拒否する。

## 9. STEP_OVER

三方式を分離して監査した。

1. static pose chain: 346 pose中91安定、1,024 weight-shift候補中19有効だったが、必要placement約0.36 mに対し安全候補のforward displacementは最大0.0382 mで完全chainは0。
2. Stage 2 single-step option: 144条件中厳格成功4件（全てleft lead）、厳格最大reach約0.0351 m、最大toe clearance 0.0253 m。phase selectionとreachが不足した。
3. whole-body trajectory optimization: kinematic waypointは構成できたが、限定solverではcontact-force/dynamic feasibilityとIsaac replayへ到達できなかった。

最終分類は`OPTIMIZATION_FAILURE`である。これはG1の物理的不可能性を証明したものではなく、採用したsolver/position-control replayで完全な動的軌道を証明できなかったことを意味する。production routeはfail-closedのまま保持した。

## 10. LAND

Stage 2 standing baseのbaseline dropでは0.02 mを10/10で安全に処理した一方、0.04 mは成功80%、0.06 mは成功0%だった。scripted pre-flex/absorption/recoveryは0.02 mでも成功80%へ悪化し、0.04 mで70%、0.06 mで0%。高drop tailではankle torque saturationが支配し、position offset controllerでは接触impulseを十分制御できなかった。

したがって`observed_standing_drop_tolerance_m=0.02`はpassive robustnessの観測値でありLAND技能ではない。LAND supported rangeはnone。より高帯域のimpedance/torque-aware low-level controllerが必要である。

## 11. command_system_v1

正式transition graphは次のとおり。

```text
RUNNING_FAMILY: RUN <-> TURN
STANDING_FAMILY: STAND <-> CROUCH_SHALLOW
PROTOTYPE: STOP
```

RUN–TURN–RUNとSTAND–CROUCH–STANDのみを正式sequenceとして扱う。cross-family transitionは`CROSS_BASE_FAMILY_TRANSITION_UNRESOLVED`で拒否し、STOPを暗黙挿入しない。拒否時はcurrent controller、base option、command、actionを変更せず、unsafe primitiveを開始しない。55件のunsupported requestでaction bitwise不変、最大action discontinuity 0を確認した。machine-readable capability manifestがskill status、range、transition、provenanceのsource of truthである。

## 12. Formal Results

| 対象 | episode | 成功/hold | fall | その他 |
|---|---:|---:|---:|---|
| STAND | 50 | settle 98%、hold 98% | 2% | speed mean/p95 0.0067/0.0133 m/s、flight 0%、saturation 0% |
| RUN–TURN–RUN | 50 | sequence 92%、pre-RUN/TURN/recovery各100% | 0% | final heading 0.0115 rad、path lateral p95 0.0714 m |
| STAND–CROUCH–STAND | 50 | sequence/CROUCH/return/final hold各100% | 0% | contact/saturation 0%、depth error mean 0.00060 m |
| unsupported requests | 55 | safe rejection 55/55 | ― | action bitwise不変、最大不連続0 |
| protected route | 58 tensors | hash一致 | ― | RUN、TURN L/R 45/90、STOP、CROUCH action bitwise一致 |

## 13. Failure Taxonomy

- skill failure: 目標位置、heading、hold等の契約未達
- base mismatch: running baseをstanding/contact-rich技能へ流用した構造競合
- action authority不足: 安全なresidual上限内で目標poseへ届かない
- phase-selection failure: gaitのlead/support位相を決定論的に選べない
- dynamic/contact control不足: position offsetだけではimpact、摩擦、支持力を制御できない
- optimization failure: 緩和kinematicsは解けても完全dynamic problem/replayを証明できない

## 14. 得られた設計原則

1. 単一baseは万能ではなく、runningとstandingを分ける。
2. residualが常に正解ではなく、既知のmotion primitiveが安全で再現性に優れる場合がある。
3. supported rangeとNOT_SUPPORTEDを明示し、clampで成功を偽装しない。
4. PPOの前にaction authorityとmappingを監査する。
5. contact-rich skillはposition offsetだけでは解けない場合がある。
6. trained skillをtensor hashと代表actionのbitwise比較で保護する。
7. fail-closed routingにより未解決transitionを安全に隔離する。

## 15. Limitations

- Isaac Simのみ。実機、actuator delay、通信遅延は未評価
- rough terrain、外乱、認識誤差へのrobustnessは未解決
- STOP formal gate未達
- cross-family transition未解決
- STEP_OVER/LAND未対応、CROUCH_DEEP未対応
- 単一統合NNではなくhybrid controller system

## 16. exp_007へのhandoff

exp_006の主要な残課題はrunning/standing family間のbridgeである。次実験では中心状態を`STAND <-> WALK <-> RUN`とし、Stage 2 actorがzero commandでSTAND、低速commandでWALKを生成できる性質を利用する。RUNから直接STANDへ切り替える代わりに`RUN -> WALK -> STAND`と段階的に速度と接触周期を落とせば、転倒、saturation、heading driftを減らし、初期接触位相と速度差へ汎化できるという仮説を検証する。これは旧STOPを再学習する計画ではなく、停止をwalk-centered transitionとして再構成する計画である。

## 17. Reproduction

主要checkpoint:

- Stage 2 standing/walking: `logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt`
- RUN/TURN/STOP parent: `logs/rsl_rl/physical_ai_g1_command_skills/2026-07-20_14-34-35_pilot_stop_stage_a_braking/model_31.pt`
- artifact: `artifacts/exp_006_unitree_g1_command_skills/command_system_v1`

```powershell
# SHA verification
$root = ".\artifacts\exp_006_unitree_g1_command_skills\command_system_v1"
Get-Content "$root\SHA256SUMS" | ForEach-Object {
  $p = $_ -split '  ', 2
  if ((Get-FileHash "$root\$($p[1])" -Algorithm SHA256).Hash.ToLower() -ne $p[0]) { throw "SHA mismatch: $($p[1])" }
}

# Static freeze verification
python .\experiments\isaaclab\exp_006_unitree_g1_command_skills\scripts\verify_command_system_freeze.py

# GUI selector
.\experiments\isaaclab\exp_006_unitree_g1_command_skills\scripts\show_command_system_demos.ps1

# Formal reproduction commands
.\artifacts\exp_006_unitree_g1_command_skills\command_system_v1\reproduction_commands.ps1
```

期待出力は`results/exp_006_unitree_g1_command_skills/command_system_v1/`の各summaryと、artifactの`formal_sequence_results.json`である。正式数値のsource of truthはartifactであり、本レポート生成時に再学習・再探索は行っていない。

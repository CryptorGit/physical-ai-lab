# OpenDuckMini hardware-safe gait experts

この実験は、OpenDuckMini v2 の歩行エキスパートを**シミュレーションで学習するための安全契約とsource overlay**です。現時点の成果物は実機モデルではありません。`contract.json` が `PROHIBITED` の間、実機へのロード、トルク有効化、歩行試験を禁止します。「完璧」は主張せず、各ゲートの測定結果だけで昇格を判断します。

## 固定した契約

- 14軸のencoder zero、rad offset、model-space directionを `contract.json` に固定しました。負方向は実測どおり `left_knee` と `left_ankle` のみです。
- 脚の `SAFE_JOINT_LIMITS` と、公式homeを `0.34395760233918127` 倍した `SAFE_INIT_POS` を固定しました。
- 全脚のdesired targetは物理SAFE境界の内側`0.050 rad`へ固定clipし、現在targetから`2.0 rad/s`（20 msで最大`0.04 rad`）で移します。診断sweepではこの点がqpos違反0と歩容維持の両方を満たし、0.060 radでは歩容が低下しました。head targetは常に0です。これはtargetの余裕であり、実qposのSAFE違反許容値ではありません。
- reset noiseが0ならexact `SAFE_INIT_POS`を保ちます。noiseありの場合だけ全脚reset qposを物理SAFE境界の内側`0.005 rad`へclipし、headは0です。この5 mradはreset専用で、0.050 radのtarget envelopeへteleportしません。特にleft kneeの`upper - 0.005 == SAFE_INIT (0.470534)`を保ちます。各resetの初期qpos auditが0違反でなければsuiteは不合格です。
- reset後の各control tickは必ず`observe -> route -> policy/profile -> final guard（1回だけ） -> data.ctrl -> physics -> post-step metrics/audit`です。初回もrouted policy targetを唯一のguard callとして使い、home-only prechargeとの二重slewは禁止します。physics先行は0 step、guardは1回だけが合格で、各resetのcontrol-first startup auditをJSONへ残します。
- head 4軸はcommand、action residual、qpos/reset noise、初期値をすべて厳密な0にします。実機head torqueも無効のままです。
- floor geomは `floor`、root bodyは `trunk_assembly` という**名前**でCPU MuJoCo modelから解決します。MJX側で `0` / `1` のようなIDを仮定しません。
- mass randomizationはzero massを乗算後もzeroに保ち、massless rootにはpayload massもCOM jitterも加えません。
- vx/vy/yaw rewardはcommand中心の対称Gaussianです。値域は `[0, 1]`、一意な大域argmaxは `actual == command` です。速度とのdot productは使いません。
- straight reverseの正式physical commandとrouterのvx下限は`-0.050 m/s`です。locked-runtime 5×15の遷移実測に対して`-0.075 m/s`はsigned-progress余裕が小さかったため再較正しました。旧`-0.075` artifactは履歴証跡であり、現行endpointの採用証拠には使いません。

機械可読な全数値、provenance hash、実機昇格ゲートは [contract.json](contract.json) にあります。読み込み時にoffset式、14軸集合、safe limit、target margin/slew、reset margin、control-first startup順序、head lock、backward-exit recovery値、実機禁止を `safe_gait_experts.contract` が再検証します。現行statusは`ADOPTED_SIMULATION_ONLY`です。profile bank、phase-entry `7/4/4`、profile cap extra `0.0125 rad` / upper `0.413034 rad`、exit-only recovery extra `0.0225 rad` / upper `0.403034 rad` / 13 ticksの組合せを、新規exact 20×30 H3 adoption runと、独立した新規exact 20×30 release-qualification runで確認しました。formal release allowlistはrelease evidence `95819b5b…`だけを含み、新規portable packageを構築済みです。これはsimulation-only releaseであり、hardwareは引き続き`PROHIBITED`です。

## Overlayの利用

実験ディレクトリを `PYTHONPATH` に加え、MuJoCo CPU modelを持つ環境の構築後にrandomizerを作ります。

```python
import jax.numpy as jp
from safe_gait_experts import (
    bounded_symmetric_tracking,
    make_domain_randomizer,
)

# Names and joint addresses are resolved here, before conversion/use in MJX.
randomization_fn = make_domain_randomizer(env._mj_model)

# Brax PPO's randomization_fn receives (mjx_model, batched_rng).
train_fn(..., randomization_fn=randomization_fn)

# Use the same 14-axis vector for episode qpos observation/reset noise.
env._qpos_noise_scale = jp.asarray(randomization_fn.qpos_noise_scale)

# command/actual order is [vx, vy, yaw_rate].
tracking = bounded_symmetric_tracking(command[:3], actual[:3], xp=jp)
```

`qpos_noise_scale` がheadを0にするだけではhead lockは完成しません。環境側でpolicy actionのhead 4 indexを0にmaskし、default actuator targetも0に固定してください。学習開始前にそのmaskをtrace/testへ残すことが必須です。

## Routed / transition正式評価

`scripts/evaluate_routed_transitions.py` は次を1つのJSONへ保存します。

- resetを分けた7 primitives（stand、前後、左右横、左右yaw）
- formal evaluationへ載せる6つの固定compound endpoint（未検証のtraining-only anchorは含めない）
- expertまたは符号を切り替えるたびにstandを挟む、resetなしのtransition sequence
- fall、local velocity、yaw、orthogonal drift、height、upright、contact metrics。contactはcontrol endpointではなく、各`mj_step`直後の全physics substepで左右footを採取します。`contact_sample_count == physics_substep sample_count`を必須にし、control endpoint contact rateは診断表示だけに残します。
- raw policy head action、mask後head action、head target、実head qpos
- pre-clip target、0.050 rad margin適用後desired target、実applied target、target slew、実joint qposそれぞれの監査（実qposの`SAFE_JOINT_LIMITS`違反sampleは0件だけが合格）
- reset stateから最初のcommand-policy targetをguard exactly 1回だけ通し、`data.ctrl`へ適用してから最初のphysics stepへ進んだことを示すcontrol-first startup監査（home-only prechargeは禁止）
- exact generated manifest/scene/model/reference、constructorがoverride前に読む`optimized_backward_gait.json`、sceneから再帰的に到達する全XML/mesh/hfield dependency closure、および全ONNX policyのSHA-256
- moving segmentのsingle-support rate `>=0.05`と、standを含む全segmentのflight rate `<=0.05`

generated rootはrepo内の`artifacts/generated_playground`に固定し、manifestの自己申告hashだけでなく、manifest/scene/model/reference/eager gaitの期待hashと、再帰MJCF dependency closure 29件（model XML 1、mesh 28）の個別hash・closure root hashをコード側allowlistへ固定します。別root、欠落・追加dependency、hash違いは実行前にhard errorです。target marginの`0.050 rad`とslewの`2.0 rad/s`もformal modeでは差し替えられません。GPU学習を停止した後、CPU評価として実行してください。

評価意味論もruntime provenanceとして固定します。dynamic import前と全suite終了後に、external source 4件（exp003 evaluator、playground package init 2件、polynomial reference loader）をexact allowlist/rootで再検証します。exp004の実行sourceと`contract.json`、8 policy role、generated assets/MJCF closure/eager gait、選択profile、H3 candidate-selection evidence `f040…`、H3 safety-only component `090e…`、H3 adoption evidence `1aea…`、superseded H2 selection/component/adoption lineage `6f65…` / `bfaf…` / `bd7e…`、依存証跡、phase/recovery実行値はpre/post snapshot rootが一致しなければ失敗します。formal WSL runtimeはCPython `3.12.3`、NumPy `2.5.1`、MuJoCo `3.11.0`、ONNX Runtime `1.28.0`とnative binary 5件のhashをhard gateし、ORT build commit `45de2a8b06`、各sessionのactual providerが`CPUExecutionProvider`だけであることをJSONへ記録します。release qualificationは20 episodes × 30 s、transition 30 s、stand 5 s、warmup 1.5 s、noise 1.0、initial speed 0.10に加え、master seedを厳密に`20260808`へ固定します。

安全性と歩容を比較するsimulation-only sweepでは、`--diagnostic-noncontract-safety`を明示した場合だけpositiveなmargin/slewを変更できます。このmodeは値にかかわらず`adoption_contract.passed=false`、`simulation_acceptance_passed=false`、hardware `PROHIBITED`です。

flagなしのH3評価は3つの固定adopted-simulation-only profileだけを読みます。straightはH1 phase7/rate1.05由来のprofile `0a3c0849…`、leftは`b36f14dc…`、rightは`e2229527…`で、それぞれのpathとSHA-256を実行前に検査します。全profileの`composition.left_knee_extra_upper_margin_rad`は`0.0125`、profile upper targetは`0.413034 rad`のままです。straight componentは20×30で20/20、reverse transition prefixは20 seeds・100/100 segmentsを通過しました（SHA-256 `ff9412da…` / `5cfff9e9…`）。

旧bundle `af7f14c2…` / phase `6/4/4` / recovery `.0125`は5×15で190/190を通過しましたが、続くformal 20×30（SHA-256 `e975a078…`）でstraight reverse fall 1件とtransition qpos failure 5件、合計6 acceptance failuresを出したため**REJECTED**です。旧5×15 SHA-256 `8fe375ce…`は旧lineageの履歴であり、現H3 candidate selection allowlistには入りません。

phase-entry mapping `7/4/4`はH3 no-flag adopted defaultです。router switch要求ではなく`effective vx < -0.02`がfalseからtrueになる、backward feedforwardが最初に適用されるtickでincrement前に一度だけ設定します。backward family内でfeedforwardが連続activeならexpertが変わってもresetしません。eventにはprevious/current expert、effective command、reset前global phase、pre-increment値、profile phase rate、最初に実際使用したphaseを保存します。H3 adoption evidence `1aea…`では合計120 events（各40）を通過しました。旧H2 `bd7e…`はsuperseded lineageです。

backward-exit recoveryもno-flag defaultでenabledです。profile capとは独立したexit-only capとして、backward feedforwardがactiveからinactiveへ変わる最初のtickからleft knee desired targetを`0.403034 rad`以下（SAFE upper `0.475534` − base margin `0.050` − recovery extra `0.0225`）へ13 control ticks / `0.26 s`保持し、その後即時解除します。backwardへ再進入した場合は残りholdをcancelし、final target guardは各tickで引き続き1回だけです。安全専用evidence `h2_aggressive_short_transition_recovery0225_hold13_20seed_v1.json`（SHA-256 `090e09cc…`）は20 seeds、500/500 safety segments、370,000/370,000 physics/contact samples、fall/qpos/target/slew/route/nonfinite 0、left-knee最小余裕`0.0018947227635599528 rad`でした。一方、短尺motion gateは489/500で11 failuresのため、artifact自体は`passed=false` / `DIAGNOSTIC_FAIL`のままです。この11件を隠さずpinし、combined 5×15と20×30のadoption evidenceには使用しません。旧H2 `bfaf…` / `6f65…` / `bd7e…`はsuperseded lineageとして保持します。

現行candidate-selection evidenceは`artifacts/h3_combined_candidate_5x15_seed20260808_v1.json`（SHA-256 `f040a9c6f9783b7d50dd5590389d3c81411e8f3a7fa9dd155e8ac78175d5ff56`）です。primitives、compounds、transitionsの3 suiteはすべてtrueで、各5 episodes、合計15 episodes / 190 segments / 190 acceptancesを通過しました。1,100,000 physics substeps / 1,100,000 contact samples、110,000 control samples、11,000,000 leg samplesを監査し、fall、qpos、nonfinite、applied-target limit、desired-target margin、slew、unauthorized applied-margin、routeは0です。applied-marginの40 samplesはstartup margin transitionの40 samplesと完全一致する許可済み過渡で、pre-clip margin 110,403 samplesはguard前の情報値です。最小heightは`0.17911993 m`、最小uprightは`0.9785479266972336`、phase-entryは30 events（straight/left/right各10）で`7/4/4`、recoveryは15 exits / 195 active ticksでした。これは5×15 candidate selectionの通過証跡であり、adoption、formal simulation acceptance、releaseの証跡ではありません。

現行adoption evidenceは`artifacts/h3_formal_candidate_pending_20x30_seed20260808_v1.json`（18,597,453 bytes、SHA-256 `1aea58904598cfba8ea4ef572f9473bba647eacc695f7fce3fcaa1b8646391aa`）です。新規exact 20×30 runで3 suite、60 episodes / 760 segments / 760 acceptances、8,150,000 physics/contact samples、815,000 controls、81,500,000 leg samplesを完了しました。fall、qpos、nonfinite、target、desired-margin、unauthorized-margin、slew、routeは0、許可済みstartup marginは147/147、最小height `0.17911993 m`、最小upright `0.9777608163890137`、left-knee最大`0.4736497298325716 rad`（SAFE余裕`0.0018842701674284257 rad`）です。phase `7/4/4`は各40 events、recoveryは60 exits / 780 active ticksです。このhashだけをsimulation adoption allowlistへ追加し、3 profile、phase、recovery、6 reverse CommandCaseへbindします。

現行release evidenceは`artifacts/h3_formal_release_20x30_seed20260808_v1.json`（18,611,839 bytes、SHA-256 `95819b5bc1d0827a5ad779542a6f98c4aaebacf5f55a8303c0b5a14fba501674`）です。adoption後に別実行したexact 20×30で、同じ60 episodes / 760 segments / 28,120 explicit checks、8,150,000 physics/contact samples、815,000 controls、81,500,000 leg samplesを再完了し、全safety violationは0でした。許可済みstartup marginは147/147、phaseは各40、recoveryは60 exits / 780 ticks、left-knee最大とSAFE余裕はadoption runと同値です。全760 segmentのmetrics、physics-substep audit、target-safety auditもadoption runとbit-exactでした。`f040…`のselection、`090e…`のsafety、`1aea…`のadoption、`95819…`のreleaseは相互代用しない別証跡です。

`--diagnostic-unadopted-*`、`--diagnostic-noncontract-safety`、`--diagnostic-unadopted-policy`、`--policy-command-diagnostic-suite`のいずれかを指定するとH3 adopted defaultは無効になり、履歴再現用のdiagnostic pathに入ります。このmodeは結果にかかわらずadoption/simulation acceptance false、hardware `PROHIBITED`です。

reverse adoptionはhard-coded booleanではありません。straight/left/rightの各roleについて、allowlisted profile hash、allowlisted independent adoption evidence hash、blocked/rejected/pending/diagnosticではないstatusのANDから導出します。H3では3 profile hashとadoption evidence `1aea…`を許可し、`090e…`は6 reverse CommandCaseのsafety-component allowlistにだけpinします。3 profile、6 reverse CommandCase、phase-entry、backward-exit recoveryは`ADOPTED_SIMULATION_ONLY`です。package releaseはさらに独立したrelease evidence `95819…`を要求し、他hash、改ざんJSON、scale/seed/profile/policy/assets/provenance不一致はoutput directory生成前にfail-closedで拒否します。

non-reverseのformal mappingはphysical command（router・metrics用）とpolicy observation commandを分離します。primitiveはforward `.05→.10`、lateral `±.06→±.10`、yaw-left `+.30→[vy=-.06,yaw=+.60]`、yaw-right `-.30→-.80`です。compoundはforward-left `(.04,0,+.30)→(.08,0,+.30)`、forward-right `(.04,0,-.22)→(.08,0,-.45)`、forward-lateral-left `(.04,+.05,+.17)→(.06,+.05,+.20)`、forward-lateral-right `(.04,-.03,-.15)→(.06,-.05,-.35)`です。明示overrideは最終policy commandなので、positive compoundへlateral補正を二重適用しません。standは変更せず、reverse 3 routeは固定H3 adopted profileを使用します。このmappingはprimitive/compound/transition definitionと各segmentのJSONへ明記されます。

低いphysical endpointと従来のpolicy励起commandを分離して比較する場合は`--policy-command-diagnostic-suite`を指定します。各caseは独立resetの固定1 episode × 5秒です。forward `(physical .05 / policy .10)`、lateral `(±.06 / ±.10)`、yaw-left `(+.30 / [vy=-.06,yaw=+.60])`に加え、yaw-right候補A `(-.25/-.60)`、B `(-.25/-.70)`、C `(-.30/-.80)`を比較します。既存`(-.30/-.90)`は転倒した既知のrejected evidenceとして残します。両commandはmetrics/JSONへ別々に記録され、このsuiteも常に非採用です。

```powershell
python experiments/mujoco/exp_004_openduckmini_safe_gait_experts/scripts/evaluate_routed_transitions.py `
  --policy stand=PATH_TO_BASE_V22.onnx `
  --policy forward=PATH_TO_BASE_V22.onnx `
  --policy reverse=PATH_TO_BASE_V22.onnx `
  --policy lateral_left=PATH_TO_BASE_V22.onnx `
  --policy lateral_right=PATH_TO_BASE_V22.onnx `
  --policy yaw_left=PATH_TO_BASE_V22.onnx `
  --policy yaw_right=PATH_TO_BASE_V22.onnx `
  --policy compound=PATH_TO_BASE_V22.onnx `
  --seed 20260808 --episodes 20 --seconds 30 `
  --transition-seconds 30 --transition-stand-seconds 5 `
  --warmup-seconds 1.5 --initial-joint-noise-scale 1.0 `
  --initial-base-speed 0.10 `
  --backward-residual-scale 0
```

formal modeでは8 roleすべてのONNX SHA-256がbase-v22の固定hash `f7a27313…`でなければ実行前にrejectします。任意policyを比較できるのは`--diagnostic-unadopted-policy`を明示した場合だけで、そのmodeは常にadoption falseです。reverse left/right turnはrouter上の専用labelですが、`compound` roleへ明示的にaliasされます。8 roleの暗黙fallbackはありません。simulation acceptanceが全件passしても、出力JSONの`hardware_gate.status`は必ず`PROHIBITED`です。

flagなしの上記commandはH3 adopted-simulation-onlyの3 profile、phase `7/4/4`、profile cap `.0125`、exit-only recovery cap `.0225`をdefaultで実行します。昇格前のH3 unique配線smoke（`h3_fast_exit_safety_candidate_no_flag_smoke_seed20260808_1x2s_v1.json`、SHA-256 `b7612cac…`）は38/38 segments、31,500/31,500 physics/contact、fall/qpos/target/slew/route/nonfinite 0、phase events 6、recovery 3 exits / 39 ticks、CPU-only、55件のruntime-data closureとpre/post unchangedを確認しました。短尺motion gateはreverse-turn-rightのsigned progress 1件がfailしました。これは履歴上のwiring smokeのみで、現行adoption `1aea…`やrelease qualificationには使用しません。

採用後のunique no-flag smokeは`artifacts/h3_adopted_simulation_only_no_flag_smoke_seed20260808_1x2s_v1.json`（SHA-256 `708e2fc2d4758fe211bf62ca4a0ccbf973d4bb6c2e3d7420e02b76bf8d75fc53`）です。`evaluation_mode=ADOPTED_SIMULATION_ONLY`、3 profile/phase/recovery/6 reverse casesの`1aea…` adoption binding、`090e…` safety binding、CPU-only 8 sessions、source 9 / external 4 / binary 5 / runtime-data 57件のpre/post一致を確認しました。38 segments中37 passで、2秒短尺の`reverse_turn_right` signed-linear-progressだけがfailしました。fall/qpos/nonfinite/target/desired-margin/unauthorized-margin/slew/routeは0、31,500 physics/contact、3,150 controls、315,000 leg samples、phase events 6、recovery 3 exits / 39 ticksです。screening scaleのためadoption/release evidenceには使いません。hardwareは`PROHIBITED`のままです。

3つのreverse候補を旧フラグ意味論で履歴再現する場合のみ、上の8 policy指定に次を追加します。

```powershell
  --diagnostic-unadopted-reverse-profile experiments/mujoco/exp_004_openduckmini_safe_gait_experts/artifacts/optimized_reverse_margin050_slew200_candidate_v3.json `
  --diagnostic-unadopted-reverse-left-profile experiments/mujoco/exp_004_openduckmini_safe_gait_experts/artifacts/reverse_turn_candidates_v1/optimized_reverse_turn_left_margin050_slew200_candidate_v1.json `
  --diagnostic-unadopted-reverse-right-profile experiments/mujoco/exp_004_openduckmini_safe_gait_experts/artifacts/reverse_turn_candidates_v1/optimized_reverse_turn_right_margin050_slew200_candidate_v1.json `
  --diagnostic-unadopted-reverse-entry-phase-index 6.0 `
  --diagnostic-unadopted-reverse-left-entry-phase-index 4.0 `
  --diagnostic-unadopted-reverse-right-entry-phase-index 4.0 `
  --diagnostic-unadopted-backward-exit-recovery
```

この明示診断はH3 adopted defaultと区別され、終了コード`1`、`simulation_acceptance_passed=false`、hardware `PROHIBITED`が正しい挙動です。

## 実機禁止と解除条件

このoverlayで学習が成功しても実機許可にはなりません。最低でも次が必要です。

H3 combined 5×15 selection `f040…`、exact 20×30 adoption `1aea…`、独立exact 20×30 release `95819…`は通過済みです。旧H2 `6f65…` / `bfaf…` / `bd7e…`はsuperseded lineage、H3 `090e…`はsafety-only componentです。adoption allowlistは`1aea…`だけ、formal release evidence allowlistは`95819…`だけを含みます。builder CLIは`--formal-evidence PATH`を必須とし、selection/safety/adoption/旧evidenceや改ざんrelease evidenceをoutput directory作成前にrejectします。portable packageは`artifacts/router_packages/exp004-safe-gait-router-h3-release-20260808-v1`です。manifest SHA-256は`44b64ad794d83c518ec7deeac14b18d6afcbcccafa235359c3d8075cd25334fd`、statusは`ADOPTED_SIMULATION_ONLY`、hardwareは`PROHIBITED`です。

1. `SAFE_INIT_POS` / `SAFE_JOINT_LIMITS` と一致するsceneを再生成しhashを固定する。
2. runtime HWIの `left_knee` / `left_ankle` velocity observationにも `JOINT_DIRECTIONS` を適用し、simulationとの符号parityを証明する。
3. CPUとbatch-1 MJXのparity、全primitive、複合旋回、command transition、外乱、転倒、全target limitのacceptanceを通す。
4. torque-off、低トルク+tether、emergency stop、温度・電流・encoder境界試験を実施する。
5. 証跡を保存し、人が明示的に実機昇格を承認する。

## Test

Windows側はJAX/MuJoCoなしでも純粋関数とCPU name resolution contractを検証できます。

```powershell
python -m pytest experiments/mujoco/exp_004_openduckmini_safe_gait_experts/tests -q
```

MJX randomizer本体はJAXを遅延importするため、実学習環境では同じtestに加えて1 batchのrandomizer smoke testを実施してください。

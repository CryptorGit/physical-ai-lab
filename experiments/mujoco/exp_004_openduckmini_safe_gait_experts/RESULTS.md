# exp_004 最終研究結果

最終更新: 2026-08-08（H3 independent exact 20×30 simulation-only release passed）

## 結論

現行H3は、profile bank、phase-entry `7/4/4`、profile cap `0.0125 rad` / upper `0.413034 rad`、backward-exit recovery extra `0.0225 rad` / upper `0.403034 rad` / hold 13 ticksの組合せを、新規exact 20×30で採用し、別の新規exact 20×30でrelease qualificationを通過しました。statusは`ADOPTED_SIMULATION_ONLY`、adoption/simulation/package releaseはtrueです。formal release evidence allowlistは`95819b5b…`だけを含み、新規portable packageを構築・検証済みです。hardwareは`PROHIBITED`です。

**実機利用はPROHIBITEDです。** 本結果はMuJoCo exact-safe scene上のsimulation evidenceであり、実機安全性、実機性能、または未評価のcommand間補間を保証しません。

## H3 selection、safety、adoption、releaseの独立証跡

straight reverseの正式physical commandとrouterのvx下限は`[-0.050, 0.0, 0.0]`です。straight profile `0a3c0849…`、left/right turn profile `b36f14dc…` / `e2229527…`、base-v22 policy `f7a27313…`は変更していません。profile composition capは全routeでextra `0.0125 rad` / upper `0.413034 rad`です。

現行candidate-selection evidenceは`artifacts/h3_combined_candidate_5x15_seed20260808_v1.json`（SHA-256 `f040a9c6f9783b7d50dd5590389d3c81411e8f3a7fa9dd155e8ac78175d5ff56`）です。primitives、compounds、transitionsの3 suiteはすべてtrueで、各5 episodes、合計15 episodes / 190 segments / 190 acceptancesを通過しました。1,100,000 physics substeps / 1,100,000 contact samples、110,000 control samples、11,000,000 leg samplesを監査し、fall、qpos、nonfinite、applied-target limit、desired-target margin、slew、unauthorized applied-margin、routeは0です。

applied-marginの40 samplesはstartup margin transitionの40 samplesと完全一致する許可済み過渡です。pre-clip margin 110,403 samplesはguard前の情報値であり、安全違反として数えません。最小heightは`0.17911993 m`、最小uprightは`0.9785479266972336`でした。phase-entryは30 events（straight/left/right各10）でmapping `7/4/4`、recoveryは15 exits / 195 active ticksです。この5×15 passはcandidate selectionだけを昇格し、formal simulation acceptance、adoption、releaseは昇格しません。

現行adoption evidenceは`artifacts/h3_formal_candidate_pending_20x30_seed20260808_v1.json`（18,597,453 bytes、SHA-256 `1aea58904598cfba8ea4ef572f9473bba647eacc695f7fce3fcaa1b8646391aa`）です。primitives、compounds、transitionsはすべてtrueで、60 episodes / 760 segments / 760 acceptances、8,150,000 physics/contact samples、815,000 controls、81,500,000 leg-qpos samplesを完了しました。fall/qpos/nonfinite/target/desired-margin/unauthorized-margin/slew/routeは0、許可済みstartup marginは147/147、pre-clip marginは819,203です。

性能extremaはlinear progress比最小`0.3595826926676137`、yaw progress比最小`0.4936553773470118`、primary velocity error最大`0.028009207275213346 m/s`、orthogonal velocity最大`0.024010891338336268 m/s`、yaw-only planar最大`0.02668671634496424 m/s`、yaw-rate error最大`0.10269293300510374 rad/s`、uncommanded yaw最大`0.1395255079553353 rad/s`、stop drift最大`0.04004149774890449 m`、moving single-support最小`0.11593333333333333`、flight最大`0.005666666666666667`です。最小heightは`0.17911993 m`、最小uprightは`0.9777608163890137`、left-knee最大は`0.4736497298325716 rad`（SAFE余裕`0.0018842701674284257 rad`）でした。phaseは`7/4/4`各40 events、recoveryは60 exits / 780 active ticks、reset/startup/recovery auditsは各280 passです。

`f040…`はselection、`090e…`はsafety-only、`1aea…`はadoptionで、相互代用しません。`1aea…`だけを3 profile、phase、recovery、6 reverse CommandCaseのsimulation adoptionへbindします。

release evidenceは`artifacts/h3_formal_release_20x30_seed20260808_v1.json`（18,611,839 bytes、SHA-256 `95819b5bc1d0827a5ad779542a6f98c4aaebacf5f55a8303c0b5a14fba501674`）です。adoption後の独立runとして同じfrozen scale/master seedで60 episodes / 760 segments / 760 acceptances / 28,120 explicit checks、8,150,000 physics/contact、815,000 controls、81,500,000 leg samplesを再完了しました。fall/qpos/nonfinite/target/desired-margin/unauthorized-margin/slew/routeは0、許可済みstartup marginは147/147、phaseは`7/4/4`各40、recoveryは60 exits / 780 active ticksです。left-knee最大`0.4736497298325716 rad`、SAFE余裕`0.0018842701674284257 rad`でした。全760 segmentのmetrics、physics-substep audit、target-safety auditはadoption runとbit-exactです。

formal release allowlistは`95819…`だけを含み、selection `f040…`、safety `090e…`、adoption `1aea…`、superseded H2を受理しません。strict loaderはduplicate/nonfinite JSON、scale/seed、全nested acceptance/safety/phase/recovery audit、policy/profile/assets/provenance、hardware `PROHIBITED`を再検証します。改ざんまたは旧証跡ではpackage outputを生成しません。

portable packageは`artifacts/router_packages/exp004-safe-gait-router-h3-release-20260808-v1`です。manifest SHA-256は`44b64ad794d83c518ec7deeac14b18d6afcbcccafa235359c3d8075cd25334fd`、39 declared files / 40 actual files（manifestを含む）で、欠落、未宣言、hash不一致は0です。全declared fileの正規化closure SHA-256は`40b5e15461c0685cf18119d64e3d113b300392db3d0ff7e37caccdce4a5c7837`です。実行可能modelはbase-v22だけ、10 routesはclosed、dynamic lookupは無効です。package statusは`ADOPTED_SIMULATION_ONLY`、hardwareは`PROHIBITED`です。

H3 safety-only evidenceは`h2_aggressive_short_transition_recovery0225_hold13_20seed_v1.json`（SHA-256 `090e09cc2d82c1f42112a5f30a85cd93d940213956d6ec902fb4089875fb855a`）です。20 seeds、500/500 safety segments、370,000/370,000 physics/contact samplesを完了し、fall、qpos、target、slew、route、nonfinite violationsはすべて0でした。phase-entry eventsは60、recoveryは60 exits / 780 active ticks、left-knee最小SAFE余裕は`0.0018947227635599528 rad`です。

ただし同じ短尺artifactのcentral motion gateは489/500で、11件（straight reverse 2、reverse-left yaw 1、reverse-right 8）のmotion failureを記録しています。そのためsource artifactは`passed=false` / `DIAGNOSTIC_FAIL`のままです。中央validatorはこの11件をexact pinし、安全subsetだけを`090e…`として許可します。`090e…`はadoption/release evidenceではありません。

昇格前のH3 no-flag wiring smokeは`h3_fast_exit_safety_candidate_no_flag_smoke_seed20260808_1x2s_v1.json`（SHA-256 `b7612cac84b9b2d79e9ece887425e95b92131696cfa160c6b4cb51f09e971158`）です。38/38 segments、31,500/31,500 physics/contact samples、fall/qpos/target/slew/route/nonfinite 0、phase events 6、recovery 3 exits / 39 ticks、CPU-only、runtime-data closure 55件とsource/data pre/post一致を確認しました。短尺motion gateはreverse-turn-right 1件がfailしました。これは履歴上のwiring smokeであり、現行adoption `1aea…`やrelease qualificationの代替ではありません。

採用後のunique no-flag smokeは`h3_adopted_simulation_only_no_flag_smoke_seed20260808_1x2s_v1.json`（SHA-256 `708e2fc2d4758fe211bf62ca4a0ccbf973d4bb6c2e3d7420e02b76bf8d75fc53`）です。profile/phase/recovery/6 reverse CommandCaseのadoption/safety binding、8 policyのCPU-only provider、source 9 / external 4 / binary 5 / runtime-data 57件のpre/post一致を確認しました。primitives 7/7、compounds 5/6、transitions 25/25で、短尺`reverse_turn_right` signed progressだけがfailしました。安全監査は31,500 physics/contact、3,150 controls、315,000 leg samplesでfall/qpos/nonfinite/target/desired-margin/unauthorized-margin/slew/route 0、phase 6 events、recovery 3 exits / 39 ticksです。短尺screeningなのでexit code `1`とsimulation acceptance falseは正しく、`1aea…`をreleaseへ再利用しません。

## Superseded H2 lineage

旧H2 component `bfaf0522…`、combined selection `6f65bef5…`、初回20×30 `bd7e8a79…`は、それぞれ20/20・180/180、190/190、760/760を通過した履歴証跡です。しかしrecovery `.0175/.408034`の旧bundleに対する結果なので、H3のadoption evidenceには使いません。`bd7e…`のvalidator statusは`SUPERSEDED_H2_ADOPTION_LINEAGE`で、adoption/simulation/releaseはfalseです。

旧`af7f14c2…` / phase `6/4/4` / recovery extra `0.0125` bundleの5×15 artifact（SHA-256 `8fe375ce…`）は190/190を通過しましたが、続くformal 20×30（SHA-256 `e975a078…`）でstraight reverse fall 1件とtransition qpos failure 5件、合計6 acceptance failuresを出したため**REJECTED**です。旧5×15は旧bundle lineageの履歴であり、現H2 selection evidenceには使用しません。

## 旧runtimeのsimulation routes（履歴・現行採用には不使用）

| Route | 旧評価構成・command | Historical result（現行採用ではない） | Evidence |
| --- | --- | --- | --- |
| straight reverse | `v22` + `optimized_reverse_exact_safe_v1`、residual scale `0.0`、旧command `[-0.075, 0.0, 0.0]` | 20 episodes × 30 s、旧runtimeではpass、現行endpoint証拠としてはsuperseded | `artifacts/exp_004_openduckmini_safe_experts/v22_reverse_v1_safe_cap_m0075_20x30s.json` |
| yaw right | `v22`、policy observationのyaw command offset `-0.30`、requested command `[0.0, 0.0, -0.60]` | 20 episodes × 30 s、pass | `artifacts/exp_004_openduckmini_safe_experts/yaw_right_offset_m030_20x30s.json` |
| forward compound, positive yaw | `[0.08, 0.0, 0.30]`と`[0.06, 0.05, 0.20]` | 各20 episodes × 30 s、2 commandともpass | `artifacts/exp_004_openduckmini_safe_experts/compound_positive_yaw_2x20x30s.json` |
| forward compound, negative yaw | policy observationのyaw command offset `-0.15`、`[0.08, 0.0, -0.30]`と`[0.06, -0.05, -0.20]` | 各20 episodes × 30 s、2 commandともpass | `artifacts/exp_004_openduckmini_safe_experts/compound_negative_yaw_offset_m015_2x20x30s.json` |
| reverse turn left | `v22` + optimized left profile、`[-0.03, 0.0, 0.20]` | 20 episodes × 30 s、pass | `artifacts/exp_004_openduckmini_safe_experts/reverse_turn_left_opt_v1_m003_p020_20x30s.json` |
| reverse turn right | `v22` + legacy right profile、`[-0.04, 0.0, -0.20]` | 20 episodes × 30 s、pass | `artifacts/exp_004_openduckmini_safe_experts/reverse_turn_right_legacy_m004_m020_20x30s.json` |

上表の旧runtime passing routesはすべて、fall count `0`、head target peak `0.0`、joint target limit violations `0`でした。これは履歴上の列挙した評価点だけの結果で、現行locked runtimeの採用を意味しません。reverse turn endpointsはatomic maneuverとして扱い、未評価の補間点をpass扱いしません。

## Primitive baseline

`artifacts/exp_004_openduckmini_safe_experts/v22_exact_safe_v1_baseline_7x20x30s.json`は、補正前の`v22`を7 commands、各20 episodes × 30 sで評価した基準記録です。旧基準では5/7 commandsがpassし、straight reverseとyaw rightはfailでした。現行H3 candidate selectionの判断には`f040…`を使用し、`090e…`はその安全専用componentとしてpinします。この旧baseline自体をpassまたはadoption evidenceとは記録しません。

## 1M learned reverse candidate: REJECTED

Run IDは`pilot_reverse_safe_v1_res012_seed20260807_1m_b`です。学習は1,000,000 environment interactions、1,600 optimizer updatesまで完了しました。training curve最終行のepisode lengthは`73.94` steps（configured horizon `1000`）です。

5 episodes × 15 sのexact-safe evaluationは次のとおりです。

| Candidate setting | Mean vx | Mean vy | Falls | Formal status |
| --- | ---: | ---: | ---: | --- |
| learned residual scale `0.12` | `-0.0163345328` | `0.0463892970` | `0` | REJECTED; primary velocity error、orthogonal velocity、episode count checks fail |
| residual scale `0.0` baseline | `-0.0484201243` | `0.0265629138` | `0` | 5-run artifactはprimary velocity error、episode count checks fail |

Evidence:

- `artifacts/exp_004_openduckmini_safe_experts/reverse_expert_1m_5x15s.json`
- `artifacts/exp_004_openduckmini_safe_experts/reverse_expert_1m_res000_5x15s.json`
- `experiments/mujoco/exp_004_openduckmini_safe_gait_experts/artifacts/training_runs/reverse/pilot_reverse_safe_v1_res012_seed20260807_1m_b/`
- `experiments/mujoco/exp_004_openduckmini_safe_gait_experts/artifacts/models/reverse_safe_v1_res012_seed20260807_1m_b.onnx`

このlearned ONNXは研究証跡として保持しますが、実行はdisabled、routing/package adoptionは不可、実機利用はPROHIBITEDです。run directory内の`REJECTED.md`と`rejection_manifest.json`が同じ判断を固定します。

## Artifact integrity

すべてworkspace root相対です。

| Artifact | SHA-256 |
| --- | --- |
| `.openduck_runtime_source_review/calibrated_hybrid_policy_v22.onnx` | `f7a2731330cd3be52858989b021423a5f363cc4a8f9850512281da745a7617c0` |
| `experiments/mujoco/exp_004_openduckmini_safe_gait_experts/artifacts/optimized_reverse_margin050_slew200_h1_phase7_rate105_candidate_v1.json` | `0a3c0849124b397ca1cb60ae0b5f5783a2e545f1a03108846fa8c60cd5d8bb5b` |
| `experiments/mujoco/exp_004_openduckmini_safe_gait_experts/artifacts/h2_integrated_phase744_rate105_recovery0175_hold13_transition20x9_v1.json` | `bfaf052235e15262c34a794896e2c63a62bd1bd934998a77b7f6ea6c54009133` |
| `experiments/mujoco/exp_004_openduckmini_safe_gait_experts/artifacts/h2_combined_candidate_5x15_seed20260808_v1.json`（superseded H2 selection lineage） | `6f65bef5053da5962442eca3bf46b855a36691aa9bbad84496c9892b36ee0de4` |
| `experiments/mujoco/exp_004_openduckmini_safe_gait_experts/artifacts/h2_formal_candidate_pending_20x30_seed20260808_v1.json`（superseded H2 lineage） | `bd7e8a79b32880fa63e54570854682b5b8912f1cdafeed8e80273501dc6ef611` |
| `experiments/mujoco/exp_004_openduckmini_safe_gait_experts/artifacts/h2_aggressive_short_transition_recovery0225_hold13_20seed_v1.json`（H3 safety-only component） | `090e09cc2d82c1f42112a5f30a85cd93d940213956d6ec902fb4089875fb855a` |
| `experiments/mujoco/exp_004_openduckmini_safe_gait_experts/artifacts/h3_combined_candidate_5x15_seed20260808_v1.json`（current H3 candidate selection） | `f040a9c6f9783b7d50dd5590389d3c81411e8f3a7fa9dd155e8ac78175d5ff56` |
| `experiments/mujoco/exp_004_openduckmini_safe_gait_experts/artifacts/h3_formal_candidate_pending_20x30_seed20260808_v1.json`（current H3 simulation adoption only） | `1aea58904598cfba8ea4ef572f9473bba647eacc695f7fce3fcaa1b8646391aa` |
| `experiments/mujoco/exp_004_openduckmini_safe_gait_experts/artifacts/h3_formal_release_20x30_seed20260808_v1.json`（current H3 simulation-only release evidence） | `95819b5bc1d0827a5ad779542a6f98c4aaebacf5f55a8303c0b5a14fba501674` |
| `experiments/mujoco/exp_004_openduckmini_safe_gait_experts/artifacts/router_packages/exp004-safe-gait-router-h3-release-20260808-v1/package_manifest.json` | `44b64ad794d83c518ec7deeac14b18d6afcbcccafa235359c3d8075cd25334fd` |
| `experiments/mujoco/exp_004_openduckmini_safe_gait_experts/artifacts/h3_fast_exit_safety_candidate_no_flag_smoke_seed20260808_1x2s_v1.json`（H3 wiring smoke only） | `b7612cac84b9b2d79e9ece887425e95b92131696cfa160c6b4cb51f09e971158` |
| `experiments/mujoco/exp_004_openduckmini_safe_gait_experts/artifacts/h3_adopted_simulation_only_no_flag_smoke_seed20260808_1x2s_v1.json`（post-adoption no-flag smoke only） | `708e2fc2d4758fe211bf62ca4a0ccbf973d4bb6c2e3d7420e02b76bf8d75fc53` |
| `experiments/mujoco/exp_004_openduckmini_safe_gait_experts/artifacts/h1_phase57_rate105_formal20x30s_v1.json` | `ff9412da4a6813151b82894553e789231cc20717ab377dd3fb0c24a1d2da2a5e` |
| `experiments/mujoco/exp_004_openduckmini_safe_gait_experts/artifacts/h1_phase7_rate105_formal_transition_reverse_prefix_20seed_v1.json` | `5cfff9e96d363797433ec50f8e4f18af25469597b0bdb2623a28ecdbfbc42f19` |
| `experiments/mujoco/exp_004_openduckmini_safe_gait_experts/artifacts/formal_candidate_pending_20x30_seed20260808_v1.json`（旧H1、REJECTED） | `e975a078f452bdfe215d136b015b16d8b6b89f69f8777874fef80b6836efaead` |
| `experiments/mujoco/exp_004_openduckmini_safe_gait_experts/artifacts/stage_a_h2_components_no_flag_smoke_seed20260808_1x2s_v1.json`（wiring smokeのみ） | `983cec1f0b31f179730c89e1097e6fcd27fa045e52184870413239007169c5a3` |
| `experiments/mujoco/exp_004_openduckmini_safe_gait_experts/artifacts/stage_a_h2_5x15_promoted_no_flag_smoke_seed20260808_1x2s_v1.json`（現wiring smokeのみ） | `4c5a83a9bd75fb7bb7fc0339cf71ae51f5fe39da5e8f2c5fd694f1fc8d783644` |
| `experiments/mujoco/exp_004_openduckmini_safe_gait_experts/artifacts/stage_b_adopted_no_flag_smoke_seed20260808_1x2s_v1.json`（Stage B wiring smokeのみ） | `53dec6e92249b21f96040dff1ea34c1c048efdcfe9e873a10954b329054bcd5f` |
| `experiments/mujoco/exp_004_openduckmini_safe_gait_experts/artifacts/optimized_reverse_margin050_slew200_candidate_v3.json` | `af7f14c2c4877a088b9320d59625bd37e41677ddc3a3802761df1e982179373e` |
| `experiments/mujoco/exp_004_openduckmini_safe_gait_experts/artifacts/reverse_turn_candidates_v1/optimized_reverse_turn_left_margin050_slew200_candidate_v1.json` | `b36f14dc1bbacfbf998adc00f6e6fe62d1f14a4a8de034b1b0b18ae5bccb8703` |
| `experiments/mujoco/exp_004_openduckmini_safe_gait_experts/artifacts/reverse_turn_candidates_v1/optimized_reverse_turn_right_margin050_slew200_candidate_v1.json` | `e2229527d435d03636c091ca7b435ed3be483b0e74293d28a2ff927995bea16b` |
| `experiments/mujoco/exp_004_openduckmini_safe_gait_experts/artifacts/routed_combined_reverse_bank_phase644_recovery13_endpointm050_control_first_5x15_v1.json` | `8fe375ce044d86987364909df3b7122a9108ef58316d294a5e6e3f82ed30b51c` |
| `experiments/mujoco/exp_004_openduckmini_safe_gait_experts/artifacts/optimized_reverse_exact_safe_v1.json` | `fd2f3a6c129ed0c37a9014dbad1813764cca35cdb54dc0b63d39f82b925e2306` |
| `artifacts/exp_004_openduckmini_safe_experts/v22_exact_safe_v1_baseline_7x20x30s.json` | `ce5fc1385ed006519bcb7a0419380167763f6357abe2db732685de4456eb5aba` |
| `artifacts/exp_004_openduckmini_safe_experts/v22_reverse_v1_safe_cap_m0075_20x30s.json` | `b51315ceee913b08bf1dbd863eb505042d67a6e9578bff38691acc5bd975b1fe` |
| `artifacts/exp_004_openduckmini_safe_experts/yaw_right_offset_m030_20x30s.json` | `03e9435e6eebfb34c18cd83cd883c99c009e1d3557cbab8f30bad7ada409d6b1` |
| `artifacts/exp_004_openduckmini_safe_experts/compound_positive_yaw_2x20x30s.json` | `63cec8c6b71b7a6bb9907bd7cffa7d4bfe2c636427272c6eb97d1aad8f237456` |
| `artifacts/exp_004_openduckmini_safe_experts/compound_negative_yaw_offset_m015_2x20x30s.json` | `e0c0a816e042838c0521350d59e05719f1ff9dc282b1f2420b89c32d0bb0f6bb` |
| `artifacts/exp_004_openduckmini_safe_experts/reverse_turn_left_opt_v1_m003_p020_20x30s.json` | `07fe325a3f3ef1f631d643c5d7645917765b0ad139789328520b7e8436341cc3` |
| `artifacts/exp_004_openduckmini_safe_experts/reverse_turn_right_legacy_m004_m020_20x30s.json` | `b5929f871c4f4279cb6e4425ddd85874ef375fdf4ccaa5f53afea89be6af97e2` |
| learned `final_params.pkl` | `5cd4a278d7996445455aeb01507d8044908e7a2a8f259a0b3cc92a28f4e5b6cf` |
| learned reverse ONNX | `a7a01d005729add0dc7492c29fc86b8d1fa8d5935e4d3e84e1518b4fbd3ecbc1` |

## Scope boundary

本書が記録する評価範囲は、上記のexact-safe scene、評価条件、H3の離散command/profile、およびそのsimulation-only portable packageに限定されます。この範囲のsimulation adoption/acceptance/releaseだけをpassとします。hardware promotion、実機試験、未評価command、連続補間、外乱条件への一般化は範囲外であり、passとはみなしません。hardwareは`PROHIBITED`です。

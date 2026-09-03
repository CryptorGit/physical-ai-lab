# exp_013 Phase W2-P1-D4: held-out exact-zero generalization diagnosis

## 結論

主分類は `EXACT_ZERO_SUBSAMPLE_PREVALENCE_INSTABILITY` である。R2 step 37,000 の
held-out failure は、validationからheld-outへの真のexact-zero状態一般化低下ではない。
各start episodeに1点だけ存在するexact-zero boundaryが、復元抽出されたsampleへ何点入るかで
mean MSE gateがほぼ決まる。

既存の `EXP013_W2_P1_R2_VALIDATION_SELECTED_HELDOUT_FAIL` は保存する。Checkpoint再選択、
student学習、formal closed-loop、DAgger、dataset/label/split/gate変更、canonical promotionは行っていない。

## Metric sensitivityとsampling contract

全held-out populationでexact-zero MSEは`0.06681709`、nonzero MSEは`0.0000172390`。
Gate `0.001`を通る最大exact-zero比率は`1.4712%`、10,000 sample換算で最大147点である。
既存held-outは171点（1.71%）でFAIL、既存validationは5,000点中72点（1.44%）でPASSした。

Splitはseed `20276021`でcondition-stratified episode splitを作る。Metricはsplit内episodeを一様・
復元抽出し、そのepisode内timestepを一様・復元抽出する。明示的なepisode/condition/timestep再weightは
ない。R2 validation selectionはseed `20276100`で各group 5,000点、held-out authorizationは
group共通generator seed `20276023`で各group 10,000点を抽出した。Exact-zero点はstratifyされない。

## Full-population metrics

| split | samples | episodes | zero prevalence | mean MSE | zero MSE | nonzero MSE |
|---|---:|---:|---:|---:|---:|---:|
| train | 104,115 | 1,893 | 1.81818% | 0.00123176 | 0.06681319 | 0.0000172926 |
| validation | 13,200 | 240 | 1.81818% | 0.00123279 | 0.06687429 | 0.0000172072 |
| held-out | 13,200 | 240 | 1.81818% | 0.00123178 | 0.06681709 | 0.0000172390 |

全episodeは55 sampleでexact-zeroを1点持つ。Episode-balancedとcondition-balanced MSEはsample meanと
実質一致し、validation/held-out full-population MSE差は約`1.01e-6`だけだった。

## Resamplingとcheckpoint robustness

各splitから現行方式で10,000点を5,000回抽出した。Validation PASS確率は`0.34%`、held-outは
`0.52%`。Zero件数平均は181.716、181.895で、PASS最大zero件数は147、FAIL最小は147--148だった。
Mean MSEはzero件数・zero mean・nonzero meanから誤差`1e-10`以内で再構成できた。

元validation結果は10,000再抽出分布の下位`0.12%`相当、元held-outは`22.06%` percentileだった。
元validation PASSが稀な低zero-prevalence sampleであり、held-out FAILはnatural prevalenceに近い。

事前登録20 checkpointをvalidationだけで各1,000回再抽出した。Step 37,000はnominal順位4位、robust
順位4位、resampled PASS確率`0.30%`。順位相関`0.9895`、pairwise reversal 6件で、step 37,000固有の
強いselection overfitは支持されない。Held-outでは固定step 37,000以外を評価していない。

## Split shift

Validation/held-out exact-zero stateのsplit classifierはlinear AUROC `0.5326`、small nonlinear AUROC
`0.4980`。Exact-zero MSEも`0.06687429`対`0.06681709`でheld-outの方が僅かに低い。
Direction/yaw構成とprevalenceも一致し、`NO_MEANINGFUL_EXACT_ZERO_STATE_SHIFT`と判定した。

## Label semanticsとone-step physical diagnostic

Boundaryではphysical/actor/observation commandがbitwise zero、ramp progressも0。LabelはW1B-R2
start actor、previous action/stateはexp_012 teacherが作ったstop stateである。Current commandだけなら
stop維持が自然だが、protocol phaseとしては次step以降のnonzero commandを先取りするlabelである。

Matched physical stateを4 branchへ複製し、boundaryの1 control stepだけactionを変えた19,200 episode
診断では以下を得た。その後は全branchでstep 37,000 studentを使用した。

| branch | endpoint | acquisition | median acquisition | fall | slip | impact |
|---|---:|---:|---:|---:|---:|---:|
| student | 89.60% | 88.42% | 0.68s | 9.29% | 8.42% | 0% |
| W1B label | 93.10% | 89.56% | 0.36s | 6.15% | 5.65% | 0% |
| stop teacher | 89.83% | 88.46% | 0.68s | 9.08% | 7.92% | 0% |
| canonical parent | 93.31% | 89.50% | 0.36s | 5.92% | 5.54% | 0% |

これはdiagnostic-onlyでformal gateには使用しない。W1B labelの1 stepはendpoint/acquisitionを改善したため、
単純な無意味labelとは扱えない。

## Joint contribution

Exact-zero student-vs-W1B errorのheld-out寄与は上半身`96.33%`、下半身`3.48%`、waist`0.19%`。
Hand/wristが`85.37%`、shoulderが`10.25%`を占める。ただしlegs-only MSEも`0.00717`で、
one-step physical差も観測された。上肢支配は副次的事実だがphysical relevanceがゼロとはしない。

## AuthorizationとNext

現行random-subsample contractでは自然prevalenceの同一母集団でもPASS確率が約0.3--0.5%であり、
validation PASS/held-out FAILを能力差と解釈できない。Closed-loop authorizationは不許可、promotionなし。

次は一方式のみとする。

```text
deterministic start-retention authorization contract preflight

compare:
full-split natural-prevalence metric
episode-balanced metric
preregistered exact-zero/nonzero stratified reporting

do not change physical closed-loop gates
```

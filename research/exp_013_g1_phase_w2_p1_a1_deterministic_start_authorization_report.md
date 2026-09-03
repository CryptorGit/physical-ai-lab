# exp_013 Phase W2-P1-A1 deterministic start authorization preflight

## 結論

主分類は `EXACT_ZERO_PHYSICAL_NONINFERIORITY_FAIL` である。Legacy random subsampleは非決定的であり、
C1 full-populationとC2 episode-balancedは決定的だが、自然exact-zero prevalence `1/55`のもとで既存
MSE threshold `0.001`を超える。C3 stratified contractはnonzero startと他static groupを正しく認可するが、
事前登録されたexact-zero one-step physical gateを満たさなかった。

既存 `EXP013_W2_P1_R2_VALIDATION_SELECTED_HELDOUT_FAIL` は保存した。Step 37,000の再選択、学習、
formal closed-loop、DAgger、dataset/label/split/gate変更、promotionは行っていない。

## Legacy contract

Validationはseed `20276100`で5,000 samples/group、held-outはseed `20276023`で10,000
samples/group。Episodeとそのtimestepを二段階で復元抽出し、exact-zeroをstratifyしない。
D4でのPASS確率はvalidation `0.34%`、held-out `0.52%`で、authorization contractとして非決定的である。

## Candidate C1/C2

| Split | C1 full MSE | C1 cosine | C2 episode MSE | Result |
|---|---:|---:|---:|---|
| Train | 0.00123176 | 0.99971256 | 0.00123176 | FAIL |
| Validation | 0.00123279 | 0.99971353 | 0.00123279 | FAIL |
| Held-out | 0.00123178 | 0.99971292 | 0.00123178 | FAIL |

全episodeが55 samplesで同長かつexact-zeroを1点持つため、episode balancingはaggregate prevalenceを
変えない。C1/C2は全splitで決定的かつ一貫してFAILした。Thresholdは変更していない。

## Candidate C3

Nonzero startは全splitでMSE約`1.72e-5`、cosine約`0.9999966`でPASSした。Stop recovery、steady stop、
全moving-retention subgroupも既存`0.001/0.98` gateをPASSした。

D4の24条件×200 matched-state trajectoryをそのまま再利用したone-step physical gateは以下だった。

| Check | Result |
|---|---:|
| Aggregate endpoint差 | -3.71pp, PASS |
| Aggregate acquisition差 | -1.08pp, PASS |
| Student fall | 9.29%, FAIL (`<=5%`) |
| Dangerous slip | 8.42%, PASS |
| Impact | 0%, PASS |
| Max condition endpoint差 | 26.0pp, FAIL |
| Max condition acquisition差 | 23.5pp, FAIL |
| Condition endpoint failures | 5 / 24 |
| Condition acquisition failures | 2 / 24 |

従ってC3 combined authorizationは全splitでFAILする。Fresh rolloutは追加していない。

## Candidate discriminationとnegative controls

- Canonical parent: moving/nonzero startはPASSするが、stop recovery MSE `0.03998`、steady stop
  `0.06876`で拒否。
- exp_012 stop teacher: steady stopは強いが、nonzero start MSE `0.08312`、worst moving MSE
  `0.08564`で拒否。
- Old W2-P1 step20k: static groupsは成立するがexact-zero physical authorizationが未確立のため拒否。
- R2 step37,000: stop/start/moving staticは成立するがexact-zero physical gateで拒否。

Synthetic N1--N5は全てC3 FAIL、false PASSは0だった。N1、N2、N5はnonzero imitationで拒否。
N3/N4はin-memory hybrid actionであり、独立physical evidenceがないためfail-closedとした。

## Determinismとleakage

C1/C2/C3 reportはsame process 2回、fresh process 2回、serialized replay 2回でhash
`95fc805bfc1d650c93e12eedc1aa94b851ba637c63dbdbe9f3c908068f3e3c49`へ完全一致した。
Metric最大差は0、sample countとPASS/FAILも一致した。Full-split contractはlegacy comparison用seedを
5種類変更しても不変だった。

5pp/10ppとsafety thresholdはA1実行前にユーザーが事前登録した。Held-outによるthreshold tuning、
checkpoint変更、新規trainingは0である。

## Semantic and physical separation

Exact-zero境界ではcurrent commandがbitwise zeroだが、labelはfuture W1B start actionである。
Student actionはstop teacher actionに近く、W1B actionとはMSE `0.06682`離れる。このstratumをnonzero
startと分離するC3の構造は意味論上妥当だが、今回のphysical evidenceでは認可できない。

Static imitationはformal closed-loopへ進むためのsurrogateにすぎない。Practical stop、physical start、
moving endpoint、safetyのformal gateは別であり、A1はそれらのPASSを宣言しない。

## Next

```text
start-boundary physical capability diagnosis

do not authorize closed-loop integration
```

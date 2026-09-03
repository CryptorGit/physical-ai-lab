# exp_005_unitree_g1_flat_run

Unitree G1の既存平地歩行方策を、平地で5.0 m/sを追従する低滑り・低衝撃な周期走行方策へ段階的に発展させる実験である。

このREADMEは実験の最終状態だけでなく、各Stageで何が問題になり、何を最小限変更し、その結果を受けて次に何を変えたかを時系列で記録する。

## 1. 現在地

開発はStage 9まで完了している。最終方策は5.0 m/s指令を50 episode、転倒0%、実速度4.818 m/s、`periodic_running` 100%、足滑り0.348 m/sで達成した。

| Stage | 目的 | 到達点 | 計測して次に解いた問題 |
|---|---|---|---|
| 継承元 | 0–1 m/s歩行方策の再利用 | 1.0 m/sは安定 | 2 m/s以上で転倒 |
| 1 | 1.5 m/sまで拡張 | checkpoint互換と段階学習を確認 | さらに高速化が必要 |
| 2 | 2.2 m/sまで拡張 | 2.2–2.5 m/sで低転倒率 | 飛行相が少なく高速歩行 |
| 3 | 周期走行を明示的に学習 | 安全な交互着地を定義 | 報酬が疎すぎて連続周期0 |
| 4 | 安全な飛行相へ密な前駆信号を追加 | 周期走行を獲得 | 3.8 m/s以上で成功率低下 |
| 5 | 3.8→3.9→4.0 m/sを成功率で拡張 | 4.0 m/s、実速度3.83 m/s、転倒0% | 最大衝撃と関節飽和tail |
| 6 | 着地tailと飽和時間を低減 | 衝撃p95と関節飽和を安全域へ低減 | 4.5 m/sで滑りと追従が境界 |
| 7 | 4.5 m/sの速度と周期性を改善 | 実速度4.43 m/s、周期127、転倒0% | 左足滑り0.91 m/sの局所解 |
| 8 | 速度合格後だけ余剰滑りを抑制 | 4.5 m/sで滑り0.425、周期82% | 4.8 m/s以降の滑りと周期切断 |
| 9 | 5.0 m/sの周期品質を安定化 | 実速度4.818、周期100%、滑り0.348、転倒0% | 複数seedでの再現性確認 |

最終採用checkpointは次である。

```text
logs/rsl_rl/physical_ai_g1_flat_run/
  2026-07-18_10-51-47_stage9_5mps_cycle_knee010_1024_150/model_5840.pt
```

## 2. 設計プロセスの全体像

### 2.1 速度を上げる実験ではなく、失敗モードを一つずつ置き換える実験

本実験では、最初から大きな速度指令や多数のpenaltyを与えていない。各段階で固定速度評価、episode品質gate、飛行・着地イベント、関節診断を先に計測し、その時点で支配的な失敗だけを次Stageの差分にした。

```text
既存歩行
  ↓ 速度範囲を少しずつ拡張
安定した高速歩行
  ↓ 飛行相と交互着地を定義
周期走行
  ↓ 衝撃分布と飽和時間を計測
低衝撃・非飽和な周期走行
  ↓ 学習報酬と評価滑りを同じ物理量へ整合
滑り局所解を除去した高速周期走行
  ↓ 周期resetと速度誤差を定常/過渡へ分解
5.0 m/sの安定周期走行
```

重要なのは、速度だけを最大化すると品質が単調には改善しなかったことである。実際に次の異なる失敗モードが順番に現れた。

1. 未学習速度での早期転倒
2. 転倒しないが飛行相を持たない高速歩行
3. 正しい周期報酬が疎すぎて学習できない問題
4. 周期は成立するが着地衝撃と関節飽和tailが大きい問題
5. 速度・周期は改善するが片足だけを滑らせる局所解
6. 5 m/s付近で硬い速度gateが良い周期を切断する問題
7. 周期報酬を連続化すると膝速度飽和へ逃げる問題

各Stageは、この順番に対応している。

### 2.2 診断から方策変更までの判断表

| フェーズ | 観測した事実 | 原因仮説 | 採用した最小変更 | 採用しなかった変更 |
|---|---|---|---|---|
| 歩行→高速歩行 | 2 m/s直接指令で早期転倒 | 訪問速度分布が遠すぎる | 1.5、2.2 m/sへ段階拡張 | 最初から3–5 m/sをsample |
| 高速歩行→周期走行 | 転倒率は低いが飛行相0.5–1.1% | 速度追従だけでは歩行を選ぶ | 安全飛行、単足・交互着地を定義 | air-timeだけを強く最大化 |
| 疎報酬→学習可能な周期 | 正常着地報酬が一度も発火しない | 完成周期までcreditが届かない | 離地・飛行中の前駆報酬をcap付きで追加 | 不安全な飛行全体への報酬 |
| 4 m/s周期→低衝撃化 | 最大衝撃約5300 N、膝速度・足首トルクtail | 最大値ノイズと短時間過負荷 | 15 ms平均、超過量、95%飽和時間をpenalty | 瞬間最大値を強く罰する |
| 4.5 m/s速度改善→滑り悪化 | Stage 7で左滑り0.91 m/s | 速度利得を片足接地滑りで得た | 速度合格後だけ足別excess-slip | 全接地滑りの一律強化 |
| 5 m/s周期切断 | 良い着地の9.7%が速度gateだけで0点 | 硬い±0.30 m/s条件 | 前進・横誤差による連続減衰 | 周期判定全体の緩和 |
| 連続周期→膝飽和 | 周期88、滑り0.351だが右膝11.58% | 膝の高速屈伸が新しい逃げ道 | 膝限定の小さな飽和項 | 全関節penaltyの大幅増加 |

### 2.3 一貫して守った原則

- checkpoint互換性を守り、観測・action・network・PPOを変えない
- 速度、周期、物理品質を別々に計測し、一つの平均報酬で判断しない
- 最大値だけでなく平均、p95/p99、閾値超過率、時間割合を見る
- 学習報酬と評価値が異なる場合、同じ接触・足速度から両方を計算する
- episode平均と着地eventを分け、周期が切れた瞬間の理由を保存する
- 新しい報酬は直前Stageの成功要素を維持したまま、一つか二つに限定する
- スモークcheckpointは処理経路確認専用とし、性能比較に使わない
- 成功条件を学習後に都合よく動かさず、未達項目は未達として残す

### 2.4 ContinuationとしてのStage設計

この実験は、最適化の観点ではcontinuation、あるいはhomotopyとして設計した。ゼロから5.0 m/sを解かせるのではなく、既に解けている問題の近傍へ難しさを一つだけ追加し、その解を次の問題へ移した。

```text
安定歩行
  → 1.5 m/s
  → 2.2 m/s高速歩行
  → 2.3–2.6 m/sで周期走行へ相転移
  → 4.0 m/s
  → 低衝撃4.0 m/s
  → 4.5 m/s
  → 低滑り4.5 m/s
  → 高品質5.0 m/s
```

各Stageの速度sampleは現在の限界付近へ集中させた。既にできる低速域へsampleを浪費せず、まだ不可能な高速域で大量転倒させないためである。1024環境の並列探索は、正しい飛行時間、片足着地、左右交互性、姿勢、速度を同時に満たす希少eventの発見に有効だった。ただしStage 3が示したとおり、並列数だけ増やしても報酬が常に0なら学習できない。大量並列探索と、Stage 4の前駆報酬による成功eventの増幅を組み合わせて初めて周期走行へ遷移した。

checkpointの扱いも一本道にはしなかった。各親checkpointを保存して新Stageを分岐し、局所解へ入った場合は最新版から続けず、良好だった親へ戻る。Stage 7からStage 8へ進まずStage 6へ戻ったのは、強化学習の解が学習履歴へ依存し、悪い歩容を報酬変更後も保持する可能性があるためである。

## 3. 実験全体で変えないもの

Stage間でpolicyの入出力とPPO構造を変えていない。これによりactor、critic、optimizerを含むcheckpointを次Stageへそのまま渡せる。

- policy観測: 123次元
  - base linear velocity
  - base angular velocity
  - projected gravity
  - velocity command
  - 37関節の位置・速度
  - 直前の37次元action
- action: 全37関節のposition target offset
- Actor: `123 → 256 → 128 → 128 → 37`
- Critic: `123 → 256 → 128 → 128 → 1`
- episode: 20秒、1000 environment steps
- control step: 0.02秒
- physics step: 0.005秒
- 転倒終了: `torso_link`の接地
- 横速度指令: ±0.1 m/s
- yaw速度指令: ±0.2 rad/s
- heading target: 無効
- terrain: 平地のみ
- networkとPPO: 公式G1 flat task由来の設定を維持

速度追従、yaw追従、鉛直速度、roll/pitch角速度、姿勢、action rate、関節偏差、転倒penaltyも全Stageで維持する。後段のStageは、直前Stageで特定された問題に関係する項目だけを追加または小さく調整する。

速度指令自体が観測へ含まれるため、方策は特定速度のmotionを暗記するだけではない。速度に応じて歩幅、step周期、接地時間、飛行時間、前傾、腕振りを変える条件付き制御則を学習できる。この構造と入出力互換性を維持したことが、学習範囲を少し超えた速度へ外挿できた前提である。5.0 m/s到達はnetworkの大型化によるものではない。

## 4. 評価基準の発展

### 4.1 単なる高速移動と周期走行の区別

初期の評価は、飛行相fractionと最大飛行時間だけを見ていた。しかし、転倒中の跳躍や単発の飛行も走行に見えてしまう。このためStage 3からepisode単位の`periodic_running`判定を導入した。

周期走行には同一episodeで次を要求する。Stage 8まではepisode全体の平均速度誤差0.30 m/sを使用した。5 m/sでは最初の加速区間が支配的になるため、Stage 9では開始2秒後のgravity-aligned yaw-frame前進誤差0.25 m/sへ変更した。周期や物理品質の基準は緩和していない。

- 非転倒
- 900 steps以上
- 定常前進速度誤差0.25 m/s以下（Stage 9）
- 飛行イベント4回以上
- 最大連続安全周期3以上
- 左右交互着地率80%以上
- 正常着地率80%以上
- 平均飛行時間0.04–0.16秒

正常着地は、短い飛行後の片足着地であり、速度追従、上体傾斜、鉛直速度、安全な飛行時間を同時に満たすものとする。単発飛行だけなら`stable_with_isolated_flight`であり、周期走行とは扱わない。

学習報酬と評価判定も分ける。Stage 9の周期報酬は速度誤差に対して連続減衰するが、評価器には従来の瞬間的な±0.30 m/s失敗もevent診断として残す。学習を滑らかにしても、評価上の失敗を隠さないためである。

### 4.2 物理品質gate

Stage 5以降は、周期が成立していても滑りや衝撃、関節飽和が大きい方策を除外する。

| gate | 基準 |
|---|---:|
| 平均足滑り | 0.55 m/s以下 |
| いずれかの関節速度が上限95%以上の時間 | 5%以下 |
| いずれかの関節トルクが上限95%以上の時間 | 20%以下 |
| 着地衝撃p95 | 3500 N以下 |
| 3500 N超着地イベント率 | 5%以下 |
| base上下動 | 0.30 m以下 |
| stride左右非対称 | 0.20以下 |
| contact time左右非対称 | 0.20以下 |

Stage 7以降は全17 gateをepisodeごとに明示的に保存する。`summary.csv/json`には速度別のpass/fail件数を出す。Stage 9の最終評価では17 gateすべてが50/50 passだった。

短い評価は実装確認と原因探索に使い、最終判断は原則50 episodeで行う。3 episodeだけでは転倒1回の有無で転倒率が0%、33%、67%と大きく変わり、局所解や左右差も見落としやすい。50 episode評価は平均値を安定させるだけでなく、episode別gateの失敗頻度とtail eventを比較するために必要である。

### 4.3 誤差と報酬値の読み方

次の値は似ているが同じではない。

- 表示実速度: body-frame前進速度の時間平均
- yaw-frame前進速度: 重力方向を除いたheading基準の前進速度
- episode誤差: 各stepの絶対誤差をepisode全体で平均
- 定常誤差: 開始2秒を除いた各stepのyaw-frame前進絶対誤差
- XY追従誤差: 前進誤差と横速度誤差のノルム
- 速度追従raw: XY誤差を指数kernelへ通した学習報酬

したがって、`指令速度 - 表示実速度`と報告される速度誤差は通常一致しない。平均する前に絶対値を取ること、初期加速を含むこと、frameが異なること、横速度を含む項があることが理由である。

### 4.4 評価出力

`results/exp_005_unitree_g1_flat_run/<timestamp>/`へ次を保存する。

| ファイル | 内容 |
|---|---|
| `episodes.csv` | episode単位の追従、転倒、周期、滑り、衝撃、対称性、gate |
| `summary.csv` | 速度別平均、成功率、gate pass/fail件数 |
| `summary.json` | checkpoint、task、評価条件、全summary |
| `flight_events.csv` | 飛行時間、離地足、着地足、安全条件、周期reset理由 |
| `landing_events.csv` | 左右、raw/15 ms平均衝撃、接触前下降速度、同時飽和関節 |
| `joint_diagnostics.csv` | 全37関節の最大速度/トルク、上限比、95%以上の割合と秒数 |
| `temporal_events.csv` | 着地、飽和、終了、転倒の時系列 |
| `quality_gates.csv` | episode×gateの値、基準、pass/fail |

## 5. Stage別の改善履歴

### Stage 0: 既存歩行方策の調査

#### 出発点

`exp_004_unitree_g1_flat`の公式`Isaac-Velocity-Flat-G1-v0`学習済みcheckpointを使用した。

```text
logs/rsl_rl/physical_ai_g1_flat/
  2026-07-17_20-18-23_baseline_resume_2000/model_2998.pt
```

約2999 iterations学習済みで、最終TensorBoard値は平均報酬18.95、平均episode長989.6/1000、time-out 95.7%、base contact終了4.3%、XY速度誤差0.159 m/sだった。

継承元には、前進・横速度とyaw速度の追従、上下速度とroll/pitch角速度の抑制、torque・関節加速度・action変化の抑制、足air-time、上体姿勢、関節制限、足滑り、腕・腰・指の姿勢、転倒penaltyが含まれていた。走行専用の報酬はないが、転倒回避、上体維持、左右脚の交互運動、接地足の滑り抑制、滑らかなaction、外乱後の復元という走行の基礎技能は既に学習済みだった。この資産を捨てずに高速域へ移すことがStage 1以降の前提である。

#### 固定速度評価

| 指令 | 実速度 | 誤差 | 長さ | 転倒 | 飛行相 | 滑り | 判断 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1.0 | 0.845 | 0.155 | 1000 | 0% | 0.2% | 0.139 | 安定歩行 |
| 2.0 | 1.180 | 0.908 | 112 | 100% | 8.0% | 0.661 | 転倒過程の飛行 |
| 3.0 | 1.094 | 1.906 | 44 | 100% | 4.5% | 0.309 | 転倒過程の飛行 |

#### 判断

既存方策を直接3 m/sへ出すのではなく、観測・action・networkを保ったまま速度上限を段階的に広げることにした。

### Stage 1: 0–1.5 m/sへの拡張

#### 問題

継承元は1 m/s付近までしか学習しておらず、2 m/sを直接与えると早期転倒した。

#### 変更

- 前進速度範囲を0–1.5 m/sへ拡張
- XY速度追従weightを1.0→2.0
- single-stance air-time weightを0.75→0.25
- air-time thresholdを0.40→0.25秒
- feet slideを-0.10→-0.20
- 観測、action、終了条件、PPOは変更なし

#### 結果

既存`model_2998.pt`からactor、critic、optimizerを再開し、学習・保存できることを実動確認した。本学習checkpointは次である。

```text
logs/rsl_rl/physical_ai_g1_flat_run/
  2026-07-17_21-17-30_stage1_500/model_3497.pt
```

固定速度評価では1.0 m/sは安定、1.5 m/sは転倒率0%、2.0 m/sは不安定だった。このStageで得たものは走行ではなく、1.5 m/s付近まで崩れない高速歩行である。air-timeを最初から強くせず、前方ジャンプによる転倒を避けながら安定域を広げた。

#### 改善と次の判断

checkpoint互換性と低速域の安定性を保った段階拡張が成立したため、同じ設計で上限を2.2 m/sへ進めた。

### Stage 2: 0–2.2 m/sへの拡張

#### 問題

Stage 1では走行遷移域まで速度が届かない。

#### 変更

- 前進速度上限だけを1.5→2.2 m/sへ拡張
- Stage 1の報酬と構造を維持

#### 結果

```text
logs/rsl_rl/physical_ai_g1_flat_run/
  2026-07-17_21-40-39_stage2_1024_750/model_4246.pt
```

| 指令 [m/s] | 実速度 [m/s] | 転倒率 | 飛行相 | 滑り [m/s] |
|---:|---:|---:|---:|---:|
| 1.50 | 1.40 | 0% | 約0.3% | 0.25 |
| 1.75 | 1.63 | 0% | 約0.4% | 0.31 |
| 2.00 | 1.86 | 0% | 約0.4% | 0.39 |
| 2.20 | 2.04 | 0% | 約0.5% | 0.46 |

50 episode評価では2.2–2.5 m/sの転倒率は0–2%まで改善した。一方、飛行相fractionは約0.5–1.1%にとどまり、高速歩行だった。2.6 m/s以上では飛行相とともに滑り・転倒も増えた。

2.4–2.7 m/sへの外挿では、短い飛行を一度含むだけのepisodeまで従来分類器が`running`と判定した。飛行相fractionと最大飛行時間だけでは、単発ジャンプ、接触の瞬断、ジャンプ後の転倒を周期走行と誤認する。この誤判定が、Stage 3で走行をepisode単位の反復運動として再定義する直接の理由になった。

#### 改善と次の判断

高速追従は安定したが、飛行相を伴う周期走行ではなかった。速度上限をさらに広げず、2.3–2.6 m/sへ問題を限定し、安全な周期を直接評価するStage 3へ移った。

### Stage 3: 周期走行の定義と疎な安全着地報酬

#### 問題

Stage 2は安定して速く歩けるが、走行周期を学習する信号がない。既存air-time報酬は片足支持時間を評価し、両足が離れる飛行相を直接表さない。

#### 変更

- command rangeを2.3–2.6 m/sに限定
- 70%を2.4–2.5 m/s、30%を全rangeからsample
- feet slideを-0.20→-0.25へ小さく強化
- `safe_periodic_flight`を追加
- episode単位の`periodic_running`分類を追加

`safe_periodic_flight`は、0.04–0.16秒の飛行後に片足着地し、速度誤差0.30 m/s以下、上体傾斜0.20 rad以下、鉛直速度0.50 m/s以下、左右交互着地を同時に満たしたときだけ発火する。長いジャンプ、両足着地、追従を失った飛行は加点しない。

#### 結果

```text
logs/rsl_rl/physical_ai_g1_flat_run/
  2026-07-17_23-07-40_stage3_1024_500/model_4745.pt
```

2.4/2.5 m/sのスモーク評価は1000 steps、非転倒だったが、飛行イベント2回、連続安全周期0で`stable_with_isolated_flight`だった。

本学習checkpointの詳細診断では完了飛行106回のうち20 msが77回、40 msが14回、60 msが9回、80 ms以上が6回だった。疑似チャタリングは0回だったが、飛行時間条件を通った29イベントはすべて厳密な着地時速度条件を失敗し、報酬の実発火は0回だった。

#### 改善と次の判断

「飛行がない」のではなく、「飛行中は無報酬で、安全条件をすべて満たした着地だけを待つため信号が疎すぎる」と特定できた。安全条件を緩めず、飛行中に小さな前駆信号を与えるStage 4へ進んだ。

### Stage 4: 安全な飛行相への密な前駆信号

#### 問題

Stage 3の完了報酬だけでは、安全な着地へ到達する前の改善方向をpolicyへ伝えられなかった。

#### 変更

同じ速度rangeと終了条件を維持し、`safe_periodic_flight`だけを拡張した。

- 安全な20 ms離地: raw 0.05/step
- 安全な40–160 ms飛行: raw 0.25/step
- 1イベントの前駆報酬上限: raw 0.75
- 安全な左右交互着地完了: raw 2.0
- 160 ms超の飛行: raw -0.25/step
- 飛行中の許容速度誤差: 1.20 m/s
- 着地完了時の速度誤差: 0.30 m/s
- gravity-aligned yaw frameで速度を判定

Reward Managerが0.02秒を掛けるため、前駆信号だけで転倒penalty -200を相殺できない大きさに抑えた。

#### 結果

```text
logs/rsl_rl/physical_ai_g1_flat_run/
  2026-07-18_00-44-32_stage4_1024_500/model_5244.pt
```

学習range内の2.3–2.6 m/sでは、転倒率0%、飛行相約31%、飛行event約125回、最大連続安全周期約120、滑り0.27–0.29 m/sとなった。Stage 3の飛行相約1%、連続周期0から、安定したlimit cycleへ明確に運動様式が変化した。

前駆信号と完了報酬の実発火を確認し、その後の学習で3.7 m/sまで転倒率0%、周期走行成功率100%へ到達した。一方、既存評価では3.8 m/sで転倒率50%、周期成功率33%、滑り0.83 m/s、4.0 m/sで転倒率83%となり、速度範囲の一括拡張は危険だった。

学習上限は2.6 m/sだったにもかかわらず3.7 m/sまで周期走行を維持したことから、方策は2.6 m/sの固定motionではなく、指令速度に応じて歩容を変える制御則を獲得したと解釈できる。ただし3.8 m/sで急激に破綻しており、外挿能力を無制限とみなしてはいけない。

#### 改善と次の判断

周期走行の生成には成功した。次の課題は周期の有無ではなく、高速域へ安全に範囲を広げる方法になったため、成功率gate付き速度curriculumをStage 5へ導入した。

### Stage 5: 3.8→4.0 m/sの成功率curriculum

#### 問題

3.7 m/sは安定したが、3.8 m/s以上で転倒と滑りが急増した。一様に4.0 m/sをsampleすると、獲得済み周期を壊す可能性があった。

#### 変更

- 初期range: 3.4–3.8 m/s
- 70%を3.6 m/sから現在上限、30%を全rangeからsample
- 上限を3.8→3.9→4.0 m/sと段階化
- 直近最大100、最低50 episodeの成功率80%以上で昇格
- 周期報酬係数はStage 4から変更なし
- 目的関数を変えないzero rewardの`high_speed_diagnostics`を追加

昇格には非転倒、episode長、安全着地、滑り、関節飽和、衝撃、上下動、左右非対称を同時に要求した。curriculumのPython windowはcheckpointへ保存されないため、再開時は`-CurriculumStage 0|1|2`を明示する。

#### 診断で分かったこと

Stage 4 checkpointの短期診断では、3.7/3.8/4.0 m/sの左膝速度張り付きが2.9/2.8/4.7%、左足首rollトルク張り付きが15.8/15.4/14.6%だった。単一seedでは転倒しなかったが、物理余裕が小さいことが分かった。

#### 結果

Stage 5の評価済みcheckpointは次である。

```text
logs/rsl_rl/physical_ai_g1_flat_run/
  2026-07-18_02-22-14_stage5_pilot_1024_150/model_5393.pt
```

| 指令 | 実速度 | 誤差 | 転倒 | 周期成功 | 滑り |
|---:|---:|---:|---:|---:|---:|
| 3.8 | 3.675 | 0.189 | 0% | 100% | 0.411 |
| 3.9 | 3.753 | 0.204 | 0% | 100% | 0.443 |
| 4.0 | 3.835 | 0.224 | 0% | 100% | 0.496 |

4.0 m/sで転倒0%と周期走行を維持できた。一方、最大着地衝撃は約5275 Nまで達し、膝速度と足首トルクが上限へ到達するイベントが残った。

#### 改善と次の判断

速度curriculumにより4.0 m/sの走行自体は成立した。次は最大値だけでなく分布、左右差、接触前速度、飽和時間を診断し、高品質化するStage 6へ進んだ。

### Stage 6: 着地tailと関節飽和時間の低減

#### 問題の詳細診断

Stage 5 checkpointを3.8/3.9/4.0 m/sで各2 episode評価した。着地衝撃は飛行後の片足接地イベントで集計した。

| 指令 | 衝撃mean | median | p95 | p99 | 3500 N超率 | 最大 |
|---:|---:|---:|---:|---:|---:|---:|
| 3.8 | 1063 | 956 | 2155 | 2849 | 0.00% | 2906 |
| 3.9 | 1074 | 853 | 2093 | 2962 | 0.75% | 5650 |
| 4.0 | 1091 | 863 | 2175 | 2975 | 0.75% | 5275 |

4.0 m/sでは左/右衝撃平均861/1322 N、p95 1770/2717 N、接触前下降速度2.56/3.29 m/sだった。右足が主な衝撃tail源だった。

関節では左/右膝速度95%以上が1.45/1.20%、左足首pitch/rollトルクが3.9/15.05%、右足首pitch/rollが4.75/5.8%だった。3500 N超4イベント中3件は膝速度または右足首トルク飽和と同時だったが、0.5秒以内の転倒はなかった。

#### 変更

速度rangeを3.8–4.0 m/sに固定し、Stage 5の周期と安全報酬をすべて維持した。瞬間最大値を直接強く罰せず、短時間平均と閾値超過時間を使った。

| 追加term | weight | 内容 |
|---|---:|---|
| `landing_impact` | -0.25 | 15 ms平均鉛直力の1000 N超過量の二乗hinge |
| `precontact_foot_velocity` | -0.50 | 接触前下降速度3.0 m/s超過量の二乗hinge |
| `joint_velocity_saturation` | -0.10 | 速度上限比95%超過量の二乗和 |
| `joint_torque_saturation` | -0.10 | トルク上限比95%超過量の二乗和 |
| `landing_impact_symmetry` | -0.05 | 直近左右の15 ms平均衝撃差 |

評価器へ`landing_events.csv`、`joint_diagnostics.csv`、`temporal_events.csv`を追加した。

#### 結果

```text
logs/rsl_rl/physical_ai_g1_flat_run/
  2026-07-18_02-59-15_stage6_landing_safety_pilot_1024_150/model_5542.pt
```

- 4.4 m/s: 実速度4.28 m/s、転倒0%、周期走行、滑り0.48 m/s
- 4.5 m/s: 実速度約4.35 m/s、転倒0%、滑り約0.56 m/s、速度誤差約0.29 m/s
- 膝速度95%以上: 約0.9–1.3%
- 足首トルク95%以上: 最大約5.3%

関節と衝撃は品質基準内へ十分な余裕ができた。一方、4.5 m/sでは周期そのものは成立しているのに、滑りと速度誤差がepisodeによってgate境界を越えた。

#### 改善と次の判断

飽和penaltyをさらに強化する理由はなかった。右足滑りと追従だけを狙い、高速指令時のみ滑りcostを少し増やすStage 7へ進んだ。

### Stage 7: 4.5 m/sの高品質周期走行

#### 問題のepisode別診断

Stage 6 checkpointを4.45/4.50 m/sで各2 episode固定評価した。

| 指令 | ep | 実速度 | 誤差 | 交互着地 | 正常着地 | 最大周期 | 左/右滑り | 平均滑り | 失敗gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4.45 | 0 | 4.252 | 0.328 | 99.2% | 91.7% | 121 | 0.438/0.622 | 0.527 | speed error |
| 4.45 | 1 | 4.382 | 0.207 | 99.2% | 94.0% | 65 | 0.437/0.598 | 0.514 | なし |
| 4.50 | 0 | 4.405 | 0.241 | 99.2% | 94.0% | 125 | 0.480/0.622 | 0.549 | なし |
| 4.50 | 1 | 4.366 | 0.269 | 99.2% | 92.5% | 123 | 0.474/0.652 | 0.561 | slip |

stride非対称は0.010–0.011、contact time非対称は0.043–0.088、衝撃p95は1927–2474 N、3500 N超率は0%だった。4.45 m/sでは速度誤差だけ1 pass/1 fail、4.50 m/sでは滑りだけ1 pass/1 fail、その他のgateはすべて2 pass/0 failだった。

詳細は次に保存している。

```text
results/exp_005_unitree_g1_flat_run/stage7_stage6_gate_baseline_2026-07-18/
```

#### 変更

速度sampleを4.5 m/s周辺へ集中させる。

- 70%: 4.40–4.50 m/s
- 20%: 4.25–4.55 m/s
- 10%: 4.50–4.55 m/s
- 4.60 m/s以上は含めない

報酬差分は`high_speed_feet_slide=-0.05`の1項だけである。Stage 6の`feet_slide=-0.25`へ、4.40 m/sで0、4.45 m/sで半分、4.50 m/s以上で全量となる線形rampを掛けて加算する。

| 指令 | 追加weight | 実効feet-slide weight |
|---:|---:|---:|
| 4.40以下 | 0 | -0.25 |
| 4.45 | -0.025 | -0.275 |
| 4.50以上 | -0.05 | -0.30 |

必要な蹴り出しを阻害しないよう、急な切替や大きな係数変更は行っていない。Stage 6の周期、安全飛行、接触前速度、衝撃、左右差、関節飽和係数はすべて維持した。

#### 実装・スモーク結果

- Stage 6 `model_5542.pt`から2環境×1 update学習・保存: 成功
- 保存checkpointから再開・再保存: 成功
- 4.50 m/s固定Play、16 frames、MP4/JIT/ONNX出力: 成功
- 4.45/4.50 m/s評価: 成功
- `quality_gates.csv`と速度別gate pass/fail出力: 成功

```text
logs/rsl_rl/physical_ai_g1_flat_run/2026-07-18_03-22-50_smoke_stage7_high_quality/
logs/rsl_rl/physical_ai_g1_flat_run/2026-07-18_03-23-13_smoke_stage7_resume/
results/exp_005_unitree_g1_flat_run/stage7_smoke_2026-07-18/
```

2環境×1 update後の方策は性能用途ではない。実際にスモーク方策は4.45 m/sで早期転倒し、4.50 m/sでも追従不足だった。本学習後のcheckpointだけを50 episode評価で選ぶ。

#### 150-iteration pilotの結果と失敗

Stage 6 `model_5542.pt`から1024環境×150 iterationsを学習した。採用候補は次である。

```text
logs/rsl_rl/physical_ai_g1_flat_run/2026-07-18_03-34-52_stage7_high_quality_pilot_1024_150/model_5691.pt
```

4.50 m/s付近では実速度4.43 m/s、転倒率0%、着地衝撃2315 N、最大連続周期127まで改善した。一方、全体滑りは0.71 m/sへ悪化した。速度と周期数だけを見れば成功に見えるが、Stage 6の約0.56 m/sより明確に悪く、品質gateを通らない。

ここでStage 7から追加学習を続ける判断はしなかった。原因を同一定義で再計測し、必要なら滑り局所解へ入る前のStage 6へ戻すことにした。この「良くなった最新checkpointを常に次の初期値にするとは限らない」という判断がStage 8の出発点である。

### Stage 8: 速度合格後だけ余剰滑りを抑える

#### Stage 6/7を同一定義で再診断

Stage 6とStage 7をseed 42、4.40/4.45/4.50 m/s、各2 episode、同じ評価器で再評価した。接触判定は学習・評価とも、接触センサの力履歴についてノルム最大値が1 Nを超えることとした。水平足速度も両者ともworld frameの足先XY速度ノルムである。

ただし正規化だけが異なる。公式`feet_slide` rawは各stepで接触中の左右足速度を合計する。評価器の滑りは同じ分子を接触foot-sample数で割る。weighted episode値は`raw × weight × 0.02 s`の全step和である。評価器へ両方を並記し、定義差による見かけの悪化を排除した。

4.50 m/sでの比較は次の通りだった。

| 項目 | Stage 6 | Stage 7 |
|---|---:|---:|
| 実速度 / step平均絶対誤差 [m/s] | 4.382 / 0.259 | 4.425 / 0.245 |
| 公式`feet_slide` raw/step | 0.374 | 0.467 |
| 公式`feet_slide` weighted/episode | -1.871 | -2.333 |
| `high_speed_feet_slide` raw/step | 0.374（反実仮想値） | 0.467 |
| 同weight / weighted episode | 0 / 0 | -0.05 / -0.467 |
| 評価器の全体滑り [m/s] | 0.568 | 0.706 |
| 左 / 右滑り [m/s] | 0.499 / 0.642 | 0.912 / 0.488 |
| 左 / 右接地時間 [s] | 0.101 / 0.095 | 0.099 / 0.094 |
| 速度追従raw / weighted episode | 0.866 / 34.656 | 0.875 / 34.983 |

接地時間はほぼ変わらず、公式rawと評価滑りは同方向に悪化した。主因はStage 7が速度利得を左接地足の水平速度増加で得た非対称な局所解である。追加slide costは4.50 m/sでも約-0.47/episodeで、速度追従の約+35/episodeに対して選好を変えるには弱かった。

診断CSVは次に保存した。

```text
results/exp_005_unitree_g1_flat_run/stage6_stage7_aligned/stage6/
results/exp_005_unitree_g1_flat_run/stage6_stage7_aligned/stage7/
```

#### 最小差分

Stage 7 checkpointは使わず、すべてStage 6 `model_5542.pt`から分岐した。速度sampleは80%を4.40–4.50 m/s、20%を4.30–4.50 m/sとし、4.50 m/s超のprobeは入れていない。

Stage 6の全報酬を維持し、次の足別二乗hingeだけを追加した。

```text
gate = (command_x >= 4.40)
       and (||command_xy - base_velocity_yaw_xy|| <= 0.25)
excess_slip = gate * sum_feet(contact
                  * (ReLU(||foot_velocity_xy|| - 0.50) / 0.50)^2)
weight = -0.20
```

左右を平均してから罰しないため、一方の大滑りを他方で相殺できない。速度誤差が大きい加速途中、遊脚中、0.50 m/s以下の必要な接地運動には追加costを掛けない。

速度追従の飽和も比較した。plateauを0.25、0.20、0.15 m/sと順に狭め、0.15版が最良だった。公式追従へ戻す試験は実速度4.40 m/sまで上がった一方、速度振動0.280 m/s、周期成功72%となったため不採用とした。採用版は公式指数カーネルを誤差0.15 m/sでplateauさせ、0.15–0.25 m/s帯では追従と滑りを同時最適化する。

#### スモークと本学習

- Stage 6から64環境×2 iterations学習・保存: 成功
- 保存checkpointから再開・再保存: 成功
- Stage 8 Playで50 frames再生: 成功
- Stage 8 Evalと全品質gate CSV生成: 成功
- 64環境スモーク方策は転倒したため性能用途には使用しない
- 本学習はStage 6から1024環境×150 iterations: 成功

採用checkpoint:

```text
logs/rsl_rl/physical_ai_g1_flat_run/2026-07-18_08-36-14_stage8_excess_slip_track015_1024_150/model_5691.pt
```

#### 50 episode評価

各速度10並列環境×5 episode、合計50 episodeで評価した。

| 指令 | 実速度 | step平均誤差 | 転倒 | 周期成功 | 全体滑り | 左/右滑り | 衝撃p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4.40 | 4.271 | 0.248 | 0% | 88% | 0.401 | 0.463/0.334 | 1948 N |
| 4.45 | 4.317 | 0.250 | 0% | 88% | 0.420 | 0.481/0.355 | 1978 N |
| 4.50 | 4.369 | 0.261 | 0% | 82% | 0.425 | 0.492/0.352 | 1994 N |

4.50 m/sでは3500 N超率0%、左右接地時間0.103/0.095 s、公式`feet_slide` raw 0.270/step、weighted -1.348/episodeだった。excess項は88.7%のstepでgate内となり、raw 0.321/step、weighted -1.285/episodeだった。速度追従はraw 0.827/step、weighted +33.081/episodeである。

品質gateは速度誤差のみ41/50 passで、他はすべて50/50 passだった。平均速度誤差0.261 m/sは基準0.25を0.011超えたため厳密には未達である。

関節診断は左/右膝速度95%超過時間1.91%/1.15%、左/右足首トルク95%超過時間4.98%/4.87%だった。絶対上限5%/20%は満たすが、左膝はStage 6の約1.3%より約0.6ポイント悪化した。このため「Stage 6より悪化しない」も厳密には未達とする。その他の成功条件は満たした。

```text
results/exp_005_unitree_g1_flat_run/stage8_track015_eval_50ep/
```

### Stage 9: 5.0 m/sの周期切断と滑りを安定化

#### Stage 8の4.7–5.1 m/s外挿診断

Stage 8 `model_5691.pt`を4.5–5.2 m/sで各20 episode評価した。5.0 m/sでは実速度4.884 m/s、転倒0%だったが、滑り0.597 m/s、periodic 0%、最大連続安全周期平均54.2だった。

4.7–5.1 m/sの着地イベントで各条件が失敗した割合は次の通りだった。

| 指令 | 速度 | 飛行時間 | 単足着地 | 交互着地 | 姿勢 | 鉛直速度 |
|---:|---:|---:|---:|---:|---:|---:|
| 4.70 | 7.2% | 0.4% | 0.1% | 1.6% | 0.0% | 0.8% |
| 4.80 | 7.8% | 0.5% | 0.1% | 1.6% | 0.1% | 1.5% |
| 4.90 | 8.0% | 0.4% | 0.1% | 1.5% | 0.1% | 2.7% |
| 5.00 | 9.7% | 0.4% | 0.0% | 1.6% | 0.3% | 3.7% |
| 5.10 | 15.8% | 0.5% | 0.0% | 1.5% | 0.1% | 3.2% |

5.0 m/sのprimary resetは追従240、鉛直速度54、飛行不足11、非単足1イベントだった。飛行超過、同側連続着地、姿勢primary resetは0だった。非単足1件は両足着地である。交互着地falseにはepisode開始時のphase initializationも含まれる。

episode gateでは5.0 m/sの20/20が速度と滑りを失敗し、stride/contact-time非対称、衝撃、関節、姿勢系は全episode合格した。接触前下降速度は平均2.58 m/s、p95 3.16 m/s、左右平均2.25/2.91 m/sで、主因ではない。

```text
results/exp_005_unitree_g1_flat_run/stage8_extrapolation_to_5mps/
```

#### 速度誤差の分解

評価器へbody-frame表示速度とは別に、gravity-aligned yaw-frameの前進・横速度、world yaw-rate、XY合成誤差、開始2秒後の定常値を追加した。

Stage 8の5.0 m/s、各5 episode再評価では次だった。

| 指標 | 全episode | 2秒後の定常区間 |
|---|---:|---:|
| yaw-frame前進速度 | 4.913 | 5.148 m/s |
| 前進絶対誤差 | 0.370 | 0.166 m/s |
| 横速度絶対値 | 0.112 | 0.108 m/s |
| yaw-rate絶対誤差 | 0.416 | 0.361 rad/s |
| XY追従誤差 | 0.418 | 0.217 m/s |

表示実速度はbody-frame前進速度の時間平均である。一方、誤差は各stepの絶対誤差を平均するため、`|command - mean(actual)|`とは一致しない。初期加速、周期内の過不足、frame差、横速度を含むかどうかで値が変わる。5.0 m/sでは最初2秒の加速が全区間誤差を大きくしていたため、Stage 9のepisode速度gateを定常yaw-frame前進誤差0.25 m/sへ整合した。

```text
results/exp_005_unitree_g1_flat_run/stage9_stage8_velocity_decomposition/
```

#### Stage 9の変更

速度sampleは指定どおりとした。

- 60%: 4.85–5.00 m/s
- 25%: 4.70–4.90 m/s
- 15%: 5.00–5.10 m/s
- 5.20 m/s以上は含めない

Stage 8の報酬をすべて維持した。安全周期報酬だけ、硬い前進誤差gateを次の連続品質へ置き換えた。

```text
tracking_quality = exp(-(forward_error / 0.30)^2
                       -(lateral_error / 0.20)^2)
cycle_reward = safe_flight_and_landing_reward * tracking_quality
```

低品質周期の報酬は連続的に小さくなるが0/1で切断されない。速度追従報酬は維持しているため、速度不足の周期だけを作る方策は有利にならない。

最初の150-iteration pilotは滑り0.351、定常誤差0.151、安全周期88まで改善した一方、右膝速度飽和11.58%となった。Stage 8の全関節飽和項を維持し、左右膝だけへ小さな追加項を試した。

- 膝追加−0.05: 右膝5.89%、他の条件は合格
- 膝追加−0.10: 右膝2.49%、全条件合格。採用

#### 最終50 episode評価

採用checkpoint:

```text
logs/rsl_rl/physical_ai_g1_flat_run/2026-07-18_10-51-47_stage9_5mps_cycle_knee010_1024_150/model_5840.pt
```

| 成功条件 | 結果 | 判定 |
|---|---:|---|
| 転倒率 | 0% | pass |
| 実速度 | 4.818 m/s | pass |
| 定常前進速度誤差 | 0.075 m/s | pass |
| periodic_running | 100% | pass |
| 全体滑り | 0.348 m/s | pass |
| 左/右滑り | 0.370/0.324 m/s | pass |
| 着地衝撃p95 | 2072 N | pass |
| 3500 N超率 | 0.033% | pass |
| 左/右膝速度飽和 | 1.76/2.49% | pass |
| 左/右足首トルク飽和 | 5.18/6.48% | pass |

最大連続安全周期は平均109.4、全品質gateは50/50合格した。

```text
results/exp_005_unitree_g1_flat_run/stage9_knee010_5mps_eval_50ep/
```

### Stage横断で得た設計知見

1. **速度上限を広げるだけでは走行へ遷移しない。** Stage 2は2.2 m/sへ到達したが両脚支持の高速歩行だった。走行を作ったのは速度範囲ではなく、交互着地、安全飛行、周期完了を分けて評価したStage 3–4である。
2. **正しい報酬でも、発火しなければ学習信号にならない。** Stage 3の安全周期報酬は条件が厳しすぎてほぼ0だった。Stage 4では前駆状態と完了を分離し、まれな完成周期へ至る勾配を作った。
3. **瞬間最大値より分布と滞在率を使う。** 接触力はセンサノイズの影響を受けるため、15 ms平均、p95/p99、3500 N超過率で扱った。関節も最大値ではなく95%上限域へ張り付いた時間と関節名を診断した。
4. **一つの逃げ道を塞ぐと、別の逃げ道が現れる。** 低衝撃化後は滑り、滑り抑制後は片側膝速度飽和が次のボトルネックになった。各Stageで全品質指標を再計測し、変化した失敗モードだけを次の報酬差分にした。
5. **報酬と評価器は同じ物理量へ対応させる。** Stage 7の滑り悪化では、接触閾値、world-frame足先XY速度、足別sampleを揃え、raw、weighted、評価平均の正規化差を明示した。これにより見かけの指標差ではなく、左足滑り0.91 m/sという局所解を特定できた。
6. **品質を満たす前後で報酬の優先順位を変える。** Stage 8は速度誤差0.25 m/s以内でだけexcess-slipを有効にし、追従報酬をplateauさせた。加速中の必要な蹴り出しを守りつつ、速度合格後は余剰滑りの削減を優先した。
7. **硬いgateは評価には有効だが、学習には不連続すぎる場合がある。** Stage 9では周期報酬の速度条件だけを連続減衰へ変えた。速度追従そのものは残したため、遅い周期へ逃げず、わずかな誤差で良い周期が無報酬になる問題だけを除いた。
8. **過渡応答と定常品質を分離する。** 5.0 m/sの全episode誤差には開始時の加速が混ざる。表示速度、step平均絶対誤差、yaw-frame誤差、2秒後の定常誤差を分けたことで、定常走行の追従性能を正しく判定できた。
9. **罰則は全身一律ではなく、診断で特定した対象へ小さく加える。** Stage 9の最初のpilotでは右膝だけが11.58%まで飽和した。全関節penaltyを強めず、左右膝だけの小さな追加項を比較し、必要最小の−0.10を採用した。

最終的に、設計の単位は「速度を何m/s上げるか」ではなく、**現在の方策が性能を得るために使っている失敗経路を計測し、その経路だけを閉じること**になった。この反復により、高速歩行、周期走行、低衝撃化、滑り局所解の除去、5.0 m/s安定化を、既存能力を大きく壊さず段階的につないだ。

## 6. 報酬変更の累積一覧

| Stage | 変更 |
|---|---|
| 1 | 速度追従2.0、air-time 0.25、threshold 0.25秒、slide -0.20 |
| 2 | 報酬変更なし、速度上限のみ2.2 m/sへ拡張 |
| 3 | slide -0.25、安全な交互着地報酬を追加 |
| 4 | 安全飛行の前駆・完了・長時間飛行penaltyを追加 |
| 5 | 報酬変更なし、zero-valued診断と成功率curriculumを追加 |
| 6 | 15 ms衝撃、接触前下降速度、飽和時間、衝撃左右差を追加 |
| 7 | 4.4 m/s以上だけ滑らかに増える追加slide -0.05 |
| 8 | 誤差0.25以内だけ足別0.50 m/s超過を二乗penalty -0.20、追従を誤差0.15でplateau |
| 9 | 周期追従gateを前進/横誤差の連続減衰へ変更、膝速度飽和を追加 -0.10 |

## 7. Checkpointの系譜

| 段階 | checkpoint |
|---|---|
| 継承元 | `physical_ai_g1_flat/2026-07-17_20-18-23_baseline_resume_2000/model_2998.pt` |
| Stage 1 | `physical_ai_g1_flat_run/2026-07-17_21-17-30_stage1_500/model_3497.pt` |
| Stage 2 | `physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt` |
| Stage 3 | `physical_ai_g1_flat_run/2026-07-17_23-07-40_stage3_1024_500/model_4745.pt` |
| Stage 4 | `physical_ai_g1_flat_run/2026-07-18_00-44-32_stage4_1024_500/model_5244.pt` |
| Stage 5 | `physical_ai_g1_flat_run/2026-07-18_02-22-14_stage5_pilot_1024_150/model_5393.pt` |
| Stage 6 | `physical_ai_g1_flat_run/2026-07-18_02-59-15_stage6_landing_safety_pilot_1024_150/model_5542.pt` |
| Stage 7 | `physical_ai_g1_flat_run/2026-07-18_03-34-52_stage7_high_quality_pilot_1024_150/model_5691.pt` |
| Stage 8 | `physical_ai_g1_flat_run/2026-07-18_08-36-14_stage8_excess_slip_track015_1024_150/model_5691.pt` |
| Stage 9 | `physical_ai_g1_flat_run/2026-07-18_10-51-47_stage9_5mps_cycle_knee010_1024_150/model_5840.pt` |

`train.ps1 -Checkpoint`は元checkpointを変更せず、`_resume_*`ディレクトリへコピーしてRSL-RLの公式resume機構でロードする。

## 8. 環境ID

| Stage | Train | Play | Eval |
|---|---|---|---|
| 1 | `Isaac-Velocity-Flat-G1-Run-Stage1-v0` | — | — |
| 2 | `Isaac-Velocity-Flat-G1-Run-Stage2-v0` | — | — |
| 3 | `Isaac-Velocity-Flat-G1-Run-Stage3-v0` | `...-Stage3-Play-v0` | `...-Stage3-Eval-v0` |
| 4 | `Isaac-Velocity-Flat-G1-Run-Stage4-v0` | `...-Stage4-Play-v0` | `...-Stage4-Eval-v0` |
| 5 | `Isaac-Velocity-Flat-G1-Run-Stage5-v0` | `...-Stage5-Play-v0` | `...-Stage5-Eval-v0` |
| 6 | `Isaac-Velocity-Flat-G1-Run-Stage6-v0` | `...-Stage6-Play-v0` | `...-Stage6-Eval-v0` |
| 7 | `Isaac-Velocity-Flat-G1-Run-Stage7-v0` | `...-Stage7-Play-v0` | `...-Stage7-Eval-v0` |
| 8 | `Isaac-Velocity-Flat-G1-Run-Stage8-v0` | `...-Stage8-Play-v0` | `...-Stage8-Eval-v0` |
| 9 | `Isaac-Velocity-Flat-G1-Run-Stage9-v0` | `...-Stage9-Play-v0` | `...-Stage9-Eval-v0` |

省略表記の`...`は`Isaac-Velocity-Flat-G1-Run`である。

## 9. Stage 7の学習・評価・再生

### 実施した150-iteration pilot

```powershell
$stage6Best = ".\logs\rsl_rl\physical_ai_g1_flat_run\2026-07-18_02-59-15_stage6_landing_safety_pilot_1024_150\model_5542.pt"

.\experiments\isaaclab\exp_005_unitree_g1_flat_run\scripts\train.ps1 `
  -Stage stage7 -NumEnvs 1024 -MaxIterations 150 -Seed 42 `
  -RunName stage7_high_quality_pilot_1024_150 -Checkpoint $stage6Best
```

### 使用checkpoint

```powershell
$stage7Checkpoint = ".\logs\rsl_rl\physical_ai_g1_flat_run\2026-07-18_03-34-52_stage7_high_quality_pilot_1024_150\model_5691.pt"
```

このcheckpointは速度・周期性の比較対象であり、Stage 8の初期値には使わない。Stage 8は滑り局所解へ入る前のStage 6から再分岐する。

### 50 episode評価

```powershell
.\experiments\isaaclab\exp_005_unitree_g1_flat_run\scripts\evaluate.ps1 `
  -Task Isaac-Velocity-Flat-G1-Run-Stage7-Eval-v0 `
  -Checkpoint $stage7Checkpoint `
  -Speeds 4.40,4.45,4.50,4.55 `
  -EpisodesPerSpeed 50 -MaxSteps 51000 -CurriculumStage 2
```

4.50 m/sの最終成功基準は次である。

- 転倒率5%以下
- 実速度4.3 m/s以上
- 速度誤差およそ0.2 m/s以下
- `periodic_running`成功率80%以上
- 平均足滑り0.55 m/s以下
- 着地衝撃p95 3500 N以下
- 3500 N超過率5%以下
- 膝速度95%以上の時間5%以下
- 足首トルク95%以上の時間20%以下

### 再生

```powershell
.\experiments\isaaclab\exp_005_unitree_g1_flat_run\scripts\play.ps1 `
  -Stage stage7 -Checkpoint $stage7Checkpoint `
  -NumEnvs 1 -Visualizer kit
```

headless動画では`-Visualizer none -Video -VideoLength 200`を追加する。動画はcheckpoint runの`videos/play/`、JIT/ONNXは`exported/`へ保存される。

## 10. Stage 8の本学習・評価・再生

### 本学習

```powershell
$stage6Best = ".\logs\rsl_rl\physical_ai_g1_flat_run\2026-07-18_02-59-15_stage6_landing_safety_pilot_1024_150\model_5542.pt"

.\experiments\isaaclab\exp_005_unitree_g1_flat_run\scripts\train.ps1 `
  -Stage stage8 -NumEnvs 1024 -MaxIterations 150 -Seed 42 `
  -RunName stage8_excess_slip_track015_1024_150 `
  -Checkpoint $stage6Best
```

### 4.40/4.45/4.50 m/sを各50 episode評価

```powershell
$stage8Best = ".\logs\rsl_rl\physical_ai_g1_flat_run\2026-07-18_08-36-14_stage8_excess_slip_track015_1024_150\model_5691.pt"

.\experiments\isaaclab\exp_005_unitree_g1_flat_run\scripts\evaluate.ps1 `
  -Task Isaac-Velocity-Flat-G1-Run-Stage8-Eval-v0 `
  -Checkpoint $stage8Best -Speeds 4.40,4.45,4.50 `
  -EpisodesPerSpeed 50 -ParallelEnvsPerSpeed 10 -MaxSteps 5200 `
  -OutputDir results/exp_005_unitree_g1_flat_run/stage8_track015_eval_50ep
```

`-ParallelEnvsPerSpeed 10`は速度ごとに10 replicaを作り、それぞれ5 episodeを実行する。物理条件や集計式を変えずに速度別50 episodeを得るための並列化である。逐次評価する場合はこの引数を1、`MaxSteps`を51000にする。

### 再生

```powershell
.\experiments\isaaclab\exp_005_unitree_g1_flat_run\scripts\play.ps1 `
  -Stage stage8 -Checkpoint $stage8Best -NumEnvs 1 -Visualizer kit
```

headlessの短い確認は`-Visualizer none -Video -VideoLength 50`を追加する。

## 11. Stage 9の本学習・評価・再生

### スモーク

Stage 8 checkpointから64環境×2 iterationsの学習・保存、保存checkpointからの再開、5.0 m/s Play、Eval、全診断CSV出力を確認した。短縮方策は早期転倒したため処理経路確認専用である。

```text
logs/rsl_rl/physical_ai_g1_flat_run/2026-07-18_10-27-26_stage9_continuous_cycle_smoke/model_5692.pt
results/exp_005_unitree_g1_flat_run/stage9_smoke/
```

### 150-iteration pilot

```powershell
$stage8Best = ".\logs\rsl_rl\physical_ai_g1_flat_run\2026-07-18_08-36-14_stage8_excess_slip_track015_1024_150\model_5691.pt"

.\experiments\isaaclab\exp_005_unitree_g1_flat_run\scripts\train.ps1 `
  -Stage stage9 -NumEnvs 1024 -MaxIterations 150 -Seed 42 `
  -RunName stage9_5mps_cycle_knee010_1024_150 `
  -Checkpoint $stage8Best
```

### 5.0 m/sを50 episode評価

```powershell
$stage9Best = ".\logs\rsl_rl\physical_ai_g1_flat_run\2026-07-18_10-51-47_stage9_5mps_cycle_knee010_1024_150\model_5840.pt"

.\experiments\isaaclab\exp_005_unitree_g1_flat_run\scripts\evaluate.ps1 `
  -Task Isaac-Velocity-Flat-G1-Run-Stage9-Eval-v0 `
  -Checkpoint $stage9Best -Speeds 5.00 `
  -EpisodesPerSpeed 50 -ParallelEnvsPerSpeed 10 -MaxSteps 5200 `
  -OutputDir results/exp_005_unitree_g1_flat_run/stage9_knee010_5mps_eval_50ep
```

4.7–5.1 m/sを比較する場合は`-Speeds 4.70,4.80,4.90,5.00,5.10`を指定する。

### 再生

```powershell
.\experiments\isaaclab\exp_005_unitree_g1_flat_run\scripts\play.ps1 `
  -Stage stage9 -Checkpoint $stage9Best -NumEnvs 1 -Visualizer kit
```

headless確認は`-Visualizer none -Video -VideoLength 50`を追加する。

## 12. 実装上の注意

- 2環境×1 updateのスモークcheckpointは処理経路確認専用であり、性能比較には使わない。
- 少数sample PPO更新は元方策を大きく悪化させる場合がある。
- Stage 5 curriculumのrolling windowはcheckpointに含まれないため、再開時にstage指定が必要である。
- `periodic_running`と品質gateは本実験独自の保守的な基準であり、公式Isaac Labの成功判定ではない。
- 数値評価だけでなく、動画、`flight_events.csv`、`landing_events.csv`、`temporal_events.csv`を併用する。
- RSL-RLのobservation-group fallback warningは残るが、学習・再開・評価経路は動作確認済みである。

## 13. 成果の適用範囲と残課題

### 13.1 この結果が示すもの

この成果は、G1実機が5.0 m/sで安全に走れることの証明ではない。現在のIsaac Lab上のG1モデル、actuator設定、接触モデル、摩擦、観測、遅延条件において、5.0 m/s指令を速度・周期・滑り・衝撃・関節余裕の品質gateと同時に満たした、という結果である。

実機にはaction・sensor遅延、状態推定誤差、motor発熱、battery電圧変動、構造のたわみ、足裏摩擦差、微小な地面凹凸、モデル化されない接触衝撃がある。実機適用を検討する前に、少なくとも摩擦、質量・重心、PD gain、action delay、sensor noise、外乱、地形に対するdomain randomizationとロバスト性評価が必要である。

したがって、この実験の最大の成果は5.0 m/s checkpointだけではない。失敗を観測可能な物理量へ分解し、現在の局所解を特定し、近接する次の問題へ移す最小限のcurriculumと報酬を設計する方法論である。同じ手順は、ハードル課題なら踏切、飛行、クリアランス、着地へ分解するように、別の運動技能にも再利用できる。

### 13.2 残課題

1. Stage 9を複数seedで50 episode評価し、5.0 m/s成功率100%の再現性を確認する。
2. 初期2秒の加速性能は別gateとして評価し、定常周期品質と混同しない。
3. 5.1 m/sは上端probeのままとし、5.2 m/s以上への拡張は別段階で扱う。
4. domain randomizationと遅延・sensor noiseを導入し、sim-to-simロバスト性を測る。
5. 停止遷移、heading、地形、トラック、ハードルは別実験として扱う。

## 14. Stage 9基準成果の固定

### 14.1 固定した成果物

Stage 9の方策・報酬・checkpointは変更せず、`model_5840.pt`を5.0 m/s基準成果とする。元checkpointを保持したまま、説明的な成果物名でも複製した。`.gitignore`の`*.pt`により、どちらもGitへは含めない。

```text
元:
logs/rsl_rl/physical_ai_g1_flat_run/
  2026-07-18_10-51-47_stage9_5mps_cycle_knee010_1024_150/model_5840.pt

固定コピー:
artifacts/exp_005_unitree_g1_flat_run/
  unitree_g1_5mps_stage9_reference_model_5840.pt

SHA-256:
ce3eeef50b588f457bf1fe2b89189bf4323b7bc31086fb90b0573aada1be19ba
```

元と固定コピーのhashは一致する。checkpoint、設定、export、動画、基準評価、robustness summaryのpath・size・SHA-256は`reference_manifest.json`へ保存した。既存のStage 9再生動画も確認済みである。

```text
logs/rsl_rl/physical_ai_g1_flat_run/
  2026-07-18_10-51-47_stage9_5mps_cycle_knee010_1024_150/
  videos/play/rl-video-step-0.mp4
```

### 14.2 再現環境

| 項目 | 値 |
|---|---|
| Isaac Lab | 6.1.14、commit `ffff603eafc6b74264a5261cc0183d6a65390d78` |
| Isaac Lab Tasks | 1.10.9 |
| RSL-RL | 5.0.1 |
| PyTorch | 2.10.0+cu128 |
| Python | 3.12.13 |
| CUDA runtime | 12.8（PyTorch build。`nvcc`はPATHになし） |
| GPU | NVIDIA GeForce RTX 5090 Laptop GPU、24463 MiB |
| NVIDIA driver | 610.62 |
| physical-ai-lab基準commit | `974b010990bea7df02c65516d05547a3f7901c11` |
| 学習seed / 評価seed | 42 / 42 |
| 学習環境 / iterations | 1024 / 150 |
| physics / control step | 0.005 / 0.02秒（decimation 4） |
| episode | 20秒、1000 control steps |
| 基準評価 | 5.0 m/s、50 episode、10並列×5 episode、開始2秒を定常評価から除外 |

### 14.3 基準評価の再現

```powershell
$stage9Best = ".\artifacts\exp_005_unitree_g1_flat_run\unitree_g1_5mps_stage9_reference_model_5840.pt"

.\experiments\isaaclab\exp_005_unitree_g1_flat_run\scripts\evaluate.ps1 `
  -Task Isaac-Velocity-Flat-G1-Run-Stage9-Eval-v0 `
  -Checkpoint $stage9Best -Speeds 5.00 `
  -EpisodesPerSpeed 50 -ParallelEnvsPerSpeed 10 -MaxSteps 5200 `
  -SteadyStateStartS 2.0 `
  -OutputDir results/exp_005_unitree_g1_flat_run/stage9_reference_reproduction
```

再生は次である。既存動画を再生成する場合は`-Video`を付ける。

```powershell
.\experiments\isaaclab\exp_005_unitree_g1_flat_run\scripts\play.ps1 `
  -Stage stage9 -Checkpoint $stage9Best -NumEnvs 1 `
  -Visualizer none -Video -VideoLength 1000
```

## 15. 最低限のロバスト性評価

### 15.1 方法

新しい学習は行わず、固定したStage 9 checkpointへ5.0 m/sを指令し、各条件20 episode、seed 42で評価した。Stage 9の報酬・観測・action・networkは不変である。条件ごとの`summary.json`が存在すればskipするため、長時間評価を条件単位で再開できる。

- 床摩擦: robot materialの基準静摩擦0.8、動摩擦0.6を±20%。terrain側は静・動1.0、combine modeは`multiply`
- 質量: 全rigid bodyを一様±10%。Isaac Labのmass eventで慣性tensorも再計算
- CoM: `torso_link`だけを前後±20 mm
- PD: 全actuatorの基準stiffness、dampingを個別または同時に±10%
- action delay: 観測は遅延させず、37次元actionだけを1または2 control step遅延
- 外力: 8.0秒から0.20秒、torsoへworld frameで±60 N。符号はepisodeごとに交互
- 小凹凸: Isaac Lab heightfield、±10 mm、5 mm刻み、水平sample 1.0 m、240 m四方。初期yawを含む基準reset条件は維持

外力回復は、外力終了後に前進速度誤差0.25 m/s以下、横速度0.20 m/s以下を0.5秒連続で満たすまでの時間とした。

### 15.2 条件別結果

| 条件 | 実速度 | 定常誤差 | 転倒 | 周期 | 滑り | 左/右滑り | 衝撃p95 | 膝速度飽和 | 足首torque飽和 | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 4.816 | 0.075 | 0% | 100% | 0.350 | 0.375/0.324 | 2077 | 2.5% | 6.5% | `robust_pass` |
| friction 80% | 4.804 | 0.082 | 0% | 100% | 0.363 | 0.383/0.342 | 2155 | 2.6% | 6.6% | `robust_pass` |
| friction 120% | 4.811 | 0.072 | 0% | 100% | 0.345 | 0.367/0.321 | 1992 | 2.3% | 6.4% | `robust_pass` |
| mass 90% | 4.907 | 0.159 | 0% | 100% | 0.448 | 0.488/0.407 | 1906 | 3.0% | 6.1% | `robust_pass` |
| mass 110% | 4.684 | 0.094 | 0% | 100% | 0.304 | 0.282/0.326 | 1925 | 2.0% | 6.6% | `tracking_failure` |
| CoM後方20 mm | 4.773 | 0.050 | 0% | 100% | 0.360 | 0.366/0.353 | 2018 | 2.4% | 7.2% | `robust_pass` |
| CoM前方20 mm | 4.861 | 0.115 | 0% | 100% | 0.352 | 0.363/0.341 | 2269 | 2.5% | 6.9% | `robust_pass` |
| stiffness 90% | 4.700 | 0.071 | 0% | 100% | 0.371 | 0.396/0.345 | 2269 | 1.8% | 6.0% | `tracking_failure` |
| stiffness 110% | 4.934 | 0.191 | 0% | 0% | 0.455 | 0.418/0.492 | 2352 | 7.7% | 8.7% | `periodicity_failure`、secondary `saturation_failure` |
| damping 90% | 4.809 | 0.077 | 0% | 60% | 0.378 | 0.356/0.401 | 2186 | 4.9% | 8.1% | `periodicity_failure` |
| damping 110% | 4.812 | 0.087 | 0% | 100% | 0.364 | 0.396/0.331 | 2248 | 1.9% | 5.6% | `robust_pass` |
| PD両方90% | 4.689 | 0.079 | 0% | 100% | 0.362 | 0.421/0.296 | 1883 | 2.0% | 5.9% | `tracking_failure` |
| PD両方110% | 4.851 | 0.129 | 0% | 100% | 0.512 | 0.569/0.455 | 1855 | 2.9% | 8.5% | `robust_pass` |
| action delay 1 step | 2.269 | 0.398 | 75% | 0% | 1.184 | 1.349/1.043 | 2104 | 25.7% | 37.9% | `fall_failure`、tracking/slip/saturationも失敗 |
| action delay 2 steps | 1.063 | — | 100% | 0% | 0.871 | 1.014/0.762 | 1496 | 22.0% | 55.2% | `fall_failure`、tracking/slip/saturationも失敗 |
| 前後±60 N | 4.815 | 0.083 | 0% | 100% | 0.354 | 0.378/0.328 | 2086 | 2.7% | 6.5% | `robust_pass` |
| 左右±60 N | 4.816 | 0.079 | 0% | 100% | 0.348 | 0.371/0.324 | 2074 | 2.6% | 6.5% | `robust_pass` |
| ±10 mm凹凸 | 3.489 | 0.475 | 100% | 0% | 0.826 | 0.813/0.844 | 2446 | 16.1% | 21.7% | `fall_failure`、tracking/slip/saturationも失敗 |

2-step delayは全episodeが定常区間開始前に転倒したため、定常誤差は未定義であり0とは解釈しない。前後外力は20/20回復、平均0.87秒、左右外力も20/20回復、平均0.72秒だった。

詳細結果はGit管理外の次へ保存した。

```text
results/exp_005_unitree_g1_flat_run/stage9_robustness_5mps/
  <condition>/episodes.csv
  <condition>/summary.csv
  <condition>/summary.json
  <condition>/quality_gates.csv
  <condition>/joint_diagnostics.csv
  robustness_summary.csv
  robustness_summary.json
```

### 15.3 壊れ方の解釈

最初に明確に破綻した最小摂動は1 control step、20 msのaction delayである。75%が転倒し、転倒前から滑りと膝・足首飽和が同時に増えた。primary failureは`fall_failure`だが、原因は遅延による接地位相とaction位相のずれであり、tracking、slip、saturationが先行または同時発生したと解釈する。2 stepでは100%転倒した。現在の方策は遅延なしのsimulation timingへ強く適応している。

PDではgainの大きさだけでなく役割が分かれた。stiffness低下は加速を遅らせ、stiffness増加は速度を上げる一方で周期成功0%、膝飽和7.7%へ移った。damping−10%は転倒せず速度も維持したが周期成功60%となった。質量＋10%とPD両方−10%も定常追従は良いが、初期加速を含むepisode実速度4.75 m/sを割った。

±10 mm凹凸では100%転倒した。平地taskではheight scannerを外し、地形randomizationも学習していないため、わずかな高さ変化でも高速接地位相を補正できない。衝撃p95そのものより先にtracking、滑り、関節飽和が崩れ、最終的に転倒している。

以上から、5.0 m/s方策は**平面、遅延なし、基準付近のPD**という条件では、摩擦±20%、CoM±20 mm、質量−10%、小外力へ頑健である。一方、20 ms action delay、微小凹凸、stiffness/dampingの一部±10%には頑健とは言えない。

ハードル実験の最初の段階では、Stage 9と同じphysics/control step、action delayなし、基準PD、既知の平面摩擦を維持する。障害物形状だけを新しい難しさとして追加し、遅延・terrain randomization・actuator randomizationを同時に導入しない。特に踏切前後で、滑り、膝速度、足首torque、周期resetを引き続きgateとして保存する。

### 15.4 ロバスト性評価コマンド

全条件を実行する。既存の完成条件は自動skipされる。

```powershell
.\experiments\isaaclab\exp_005_unitree_g1_flat_run\scripts\evaluate_robustness.ps1 `
  -Checkpoint ".\artifacts\exp_005_unitree_g1_flat_run\unitree_g1_5mps_stage9_reference_model_5840.pt" `
  -EpisodesPerCondition 20 -ParallelEnvs 10 -MaxSteps 5200 `
  -OutputRoot ".\results\exp_005_unitree_g1_flat_run\stage9_robustness_5mps"
```

一部だけ再評価する場合は、例えば次のようにする。

```powershell
.\experiments\isaaclab\exp_005_unitree_g1_flat_run\scripts\evaluate_robustness.ps1 `
  -EpisodesPerCondition 20 -ParallelEnvs 10 `
  -Conditions friction_080,action_delay_1,external_force_lateral
```

既存結果を意図的に上書きする場合だけ`-Force`を付ける。集約だけを再生成する場合は次である。

```powershell
python .\experiments\isaaclab\exp_005_unitree_g1_flat_run\scripts\summarize_robustness.py `
  --root .\results\exp_005_unitree_g1_flat_run\stage9_robustness_5mps
```

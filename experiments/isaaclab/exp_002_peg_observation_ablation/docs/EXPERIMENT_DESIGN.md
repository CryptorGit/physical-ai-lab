# Experiment Design

## Long-term objective

タスクの幾何・物理条件から、
必要な観測モダリティと観測解像度を予測するための
タスク要求指標を構築する。

## First-step hypothesis

タスク要求が高くなるほど、
目標成功率を満たすための最小十分観測集合が増える。

## H1

低難度のペグ挿入では、
限定された状態観測でも十分な成功率に到達する。

## H2

クリアランスの縮小または穴位置の不確実性増加に伴い、
相対幾何情報や力覚情報の価値が増加する。

## H3

不要な観測を追加した条件では、
有限データ・ノイズ・分布外条件において、
学習効率または汎化性能が低下する場合がある。

## Phase 0: Baseline reproduction

公式環境を変更せずに学習・評価し、
checkpointとログを再現可能に保存する。

## Phase 1: Custom task registration

公式環境を継承した独自タスクを登録する。
この段階では観測・報酬・物理条件を変更しない。

## Phase 2: Observation ablation

Actor observationのみを変更する。
Critic state、報酬、制御器、学習条件は固定する。

## Phase 3: Parametric difficulty

最初はpeg-hole clearanceのみを変更する。
複数物理パラメータは同時に変更しない。
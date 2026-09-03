# exp_002_peg_observation_ablation

## Purpose

Isaac Lab の公式タスク
`Isaac-Factory-PegInsert-Direct-v0`
を基盤として、ペグ挿入条件とActorの観測構成が、
深層強化学習の学習効率・成功率・汎化性能へ与える影響を調べる。

## Current phase

Phase 0: 公式PegInsert環境の再現確認

## Research question

ペグ挿入の幾何学的・物理的条件が変化したとき、
目標成功率を満たすための最小十分観測集合は変化するか。

## Fixed conditions

- Reward function
- Action space
- Controller
- RL algorithm
- Hyperparameters
- Training steps
- Critic observations

## Experimental variables

- Actor observation set
- Peg-hole clearance
- Hole position uncertainty
- Random seed

## Baseline task

`Isaac-Factory-PegInsert-Direct-v0`
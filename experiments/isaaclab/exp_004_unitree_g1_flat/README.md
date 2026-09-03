# exp_004_unitree_g1_flat

## 目的

Unitree G1 の完成した歩行方策を得ることではなく、インストール済み Isaac Lab の公式平地速度追従タスクを使い、環境のロード、入出力の確認、短時間学習、報酬・エピソード情報、checkpoint 保存、checkpoint 再生までを PowerShell から再現できる最小実験基盤を作る。

## 採用した公式環境とアセット

- 学習: `Isaac-Velocity-Flat-G1-v0`
- 再生: `Isaac-Velocity-Flat-G1-Play-v0`
- Gym entry point: `isaaclab.envs:ManagerBasedRLEnv`
- ロボット設定: `isaaclab_assets.G1_MINIMAL_CFG`
- USD: `${ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1_minimal.usd`
- RL 実装: 公式 RSL-RL PPO 設定 `G1FlatPPORunnerCfg`

平地環境、対応する Play 環境、RSL-RL 設定がすべて正式登録されていたため、ローカル環境クラスや報酬設定は追加していない。`G1_MINIMAL_CFG` は公式 G1 と同じ関節・アクチュエータ構成で、衝突メッシュを減らした学習向けアセットである。

ローカルの `isaaclab_assets.robots.unitree` では `G1_CFG`、`G1_MINIMAL_CFG`、`G1_29DOF_CFG`、固定ベース Inspire hand 版 `G1_INSPIRE_FTP_CFG`、固定ベース Dex3 hand 版 `G129_CFG_WITH_DEX3_BASE_FIX` も確認した。Gym registry には次の歩行・速度追従環境が登録されていた。

- `Isaac-Velocity-Flat-G1-v0`
- `Isaac-Velocity-Flat-G1-Play-v0`
- `Isaac-Velocity-Rough-G1-v0`
- `Isaac-Velocity-Rough-G1-Play-v0`
- experimental package を import した場合のみ `Isaac-Velocity-Flat-G1-Warp-v0` と Play 版

benchmark script には `Isaac-Velocity-Flat-G1-v1` という文字列も残っていたが、現在の Gym registry には登録されていないため採用していない。

調査時のインストール情報は次の通り。

- Isaac Lab Python package: `6.1.14`
- `isaaclab_tasks`: `1.10.9`
- RSL-RL: `5.0.1`
- Gymnasium: `1.2.1`
- PyTorch: `2.10.0+cu128`
- Isaac Lab checkout の `git describe`: `v3.0.0-beta2.patch1`

package metadata と checkout のタグ表示は一致していない。ただし import 元は checkout 内の `source/isaaclab` と `source/isaaclab_tasks` であり、下記の動作確認はこの組み合わせで行った。

## 環境の確認

リポジトリルートから実行する。

```powershell
& "$HOME\workspace\IsaacLab\isaaclab.bat" -p `
  ".\experiments\isaaclab\exp_004_unitree_g1_flat\scripts\inspect_env.py" `
  --task Isaac-Velocity-Flat-G1-v0 `
  --num_envs 2 `
  --max_steps 128 `
  --viz none
```

このスクリプトは環境 ID、entry point、USD、全関節名、観測・行動 space、観測・行動・報酬・終了 term を表示し、zero action で有限ステップだけ進める。

## 短時間学習

```powershell
.\experiments\isaaclab\exp_004_unitree_g1_flat\scripts\train_smoke.ps1 `
  -NumEnvs 64 `
  -MaxIterations 5 `
  -Seed 42 `
  -RunName smoke
```

最小確認だけなら `-NumEnvs 2 -MaxIterations 1` でもよい。これは学習品質を評価する設定ではない。公式設定の本来の既定値は 4096 environments、1500 iterations であり、本実験では長時間学習を行わない。

ログと checkpoint は次の場所に作られる。

```text
logs/rsl_rl/physical_ai_g1_flat/<timestamp>_<run-name>/
├── events.out.tfevents.*
├── model_0.pt
├── params/
│   ├── agent.yaml
│   └── env.yaml
└── git/IsaacLab.diff
```

学習中の標準出力と TensorBoard event には `Episode_Reward/*`、`Episode_Termination/*`、`Metrics/*` が記録される。

## checkpoint の再生

GUI で再生する。checkpoint は実際に生成されたパスへ置き換える。

```powershell
.\experiments\isaaclab\exp_004_unitree_g1_flat\scripts\play_checkpoint.ps1 `
  -Checkpoint ".\logs\rsl_rl\physical_ai_g1_flat\<run>\model_0.pt" `
  -NumEnvs 1 `
  -Visualizer kit
```

GUI なしで checkpoint のロードと有限再生を検証する場合は動画記録を利用する。

```powershell
.\experiments\isaaclab\exp_004_unitree_g1_flat\scripts\play_checkpoint.ps1 `
  -Checkpoint ".\logs\rsl_rl\physical_ai_g1_flat\<run>\model_0.pt" `
  -NumEnvs 1 `
  -Visualizer none `
  -Video `
  -VideoLength 64
```

動画と export 済み policy は checkpoint と同じ run の `videos/play/`、`exported/` に保存される。

## 観測、行動、報酬、終了条件

実動確認時は 37 関節、policy 観測 123 次元、行動 37 次元だった。

観測は base linear/angular velocity、projected gravity、速度 command、37 関節の相対位置、37 関節速度、直前の 37 actions を連結する。平地設定では height scan は無効である。学習時は公式設定の observation noise が有効、Play 設定では無効になる。

行動は全 37 関節に対する position target で、scale は 0.5、default joint pose を offset として使う。

有効な報酬は XY 速度追従、yaw 速度追従、足の滞空時間と、上下速度、roll/pitch angular velocity、torque、joint acceleration、action rate、姿勢、関節制限、転倒終了、足滑り、hip/arm/finger/torso の既定姿勢からのずれに対する penalty の計 16 項目である。

終了条件は 20 秒の time-out と `torso_link` の ground contact である。

## 動作確認結果

2026-07-17、Windows/PowerShell、RTX 5090 Laptop GPU で、公式 trainer を `2 environments × 1 iteration` 実行した。

- G1 USD と平地 scene の生成: 成功
- 37 joints、123 observations、37 actions の構築: 成功
- 48 simulation steps と PPO update 1 回: 成功
- 16 reward terms と 2 termination terms の出力: 成功
- checkpoint `model_0.pt`（約 2.0 MB）の保存: 成功
- 保存した checkpoint の Play 環境へのロードと 32 frames の推論: 成功
- JIT/ONNX export と headless MP4（約 111 KB）の保存: 成功

## 残っている課題

- 1～5 iterations は配線確認専用であり、歩行性能は得られない。
- package metadata (`6.1.14`) と checkout の tag (`v3.0.0-beta2.patch1`) の不一致は、将来の更新前に環境構築履歴を確認する必要がある。
- 初回確認では Isaac Lab 内部ログの em dash が Windows CP932 へ出力できず logging error が出た。学習自体は成功しており、本実験の PowerShell ラッパーでは `PYTHONUTF8=1` と `PYTHONIOENCODING=utf-8` を設定する。
- 公式 `G1FlatPPORunnerCfg` は RSL-RL 5.0.1 の `actor`/`critic` observation group を明示しておらず、現在は `policy` group への fallback warning が出る。現バージョンでは学習・再生とも成功するが、将来の RSL-RL 更新時には公式設定側の対応状況を再確認する。
- Isaac Lab 環境に `ruff` はインストールされていなかったため lint は未実施。Python の構文確認と全スクリプトの実動確認は完了している。
- GUI 再生の見た目と長時間学習時の歩行品質は本スモーク実験の対象外である。

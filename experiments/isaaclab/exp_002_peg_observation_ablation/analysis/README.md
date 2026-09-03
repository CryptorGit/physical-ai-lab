# exp_002 TensorBoard解析

## 配置先

`analyze_tensorboard.py` を以下へ置きます。

```text
experiments/isaaclab/exp_002_peg_observation_ablation/
└── analysis/
    └── analyze_tensorboard.py
```

## 実行前確認

```powershell
& "$HOME\workspace\IsaacLab\env_isaaclab\Scripts\python.exe" `
  -c "from tensorboard.backend.event_processing import event_accumulator; import pandas, matplotlib, tabulate; print('ok')"
```

不足時:

```powershell
& "$HOME\workspace\IsaacLab\env_isaaclab\Scripts\python.exe" `
  -m pip install pandas matplotlib tensorboard tabulate
```

## 実行

```powershell
cd "$HOME\workspace\physical-ai-lab"

& "$HOME\workspace\IsaacLab\env_isaaclab\Scripts\python.exe" `
  ".\experiments\isaaclab\exp_002_peg_observation_ablation\analysis\analyze_tensorboard.py"
```

## 出力先

```text
experiments/isaaclab/exp_002_peg_observation_ablation/results/analysis/
```

生成物:

- `available_tags.csv`
- `metrics_long.csv`
- `run_summary.csv`
- `condition_summary.csv`
- `reward_curve.png`
- `success_rate_curve.png`（タグが存在する場合）
- `episode_length_curve.png`（タグが存在する場合）
- `analysis_report.md`

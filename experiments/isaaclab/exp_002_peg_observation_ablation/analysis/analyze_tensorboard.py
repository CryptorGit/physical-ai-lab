#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

RUNS = {
    "2026-07-14_01-33-22": ("baseline", 42),
    "2026-07-14_01-46-21": ("baseline", 43),
    "2026-07-14_01-59-06": ("baseline", 44),
    "2026-07-14_02-11-34": ("no_angvel", 42),
    "2026-07-14_02-24-29": ("no_angvel", 43),
    "2026-07-14_02-37-23": ("no_angvel", 44),
    "2026-07-14_02-50-19": ("no_linvel", 42),
    "2026-07-14_03-03-12": ("no_linvel", 43),
    "2026-07-14_03-15-58": ("no_linvel", 44),
    "2026-07-14_03-28-55": ("no_velocity", 42),
    "2026-07-14_03-41-53": ("no_velocity", 43),
    "2026-07-14_03-53-37": ("no_velocity", 44),
    "2026-07-14_04-06-21": ("position_only", 42),
    "2026-07-14_04-19-06": ("position_only", 43),
    "2026-07-14_04-32-01": ("position_only", 44),
}

OBS_DIMS = {
    "baseline": 19,
    "no_angvel": 16,
    "no_linvel": 16,
    "no_velocity": 13,
    "position_only": 9,
}

@dataclass(frozen=True)
class RunInfo:
    run_dir: str
    condition: str
    seed: int
    observation_dim: int
    event_file: Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--log-root",
        type=Path,
        default=Path.home() / "workspace" / "physical-ai-lab" / "logs" / "rl_games" / "Factory",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "analysis",
    )
    return p.parse_args()

def find_event_file(run_path: Path) -> Path:
    files = sorted(run_path.rglob("events.out.tfevents*"))
    if not files:
        raise FileNotFoundError(f"No TensorBoard event file under {run_path}")
    return max(files, key=lambda p: p.stat().st_mtime)

def load_runs(log_root: Path):
    runs = []
    for run_dir, (condition, seed) in RUNS.items():
        run_path = log_root / run_dir
        if not run_path.exists():
            print(f"[WARN] Missing: {run_path}")
            continue
        runs.append(
            RunInfo(
                run_dir,
                condition,
                seed,
                OBS_DIMS[condition],
                find_event_file(run_path),
            )
        )
    if not runs:
        raise RuntimeError(f"No configured runs found under {log_root}")
    return runs

def load_scalars(run: RunInfo):
    ea = event_accumulator.EventAccumulator(
        str(run.event_file),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()
    tags = sorted(ea.Tags().get("scalars", []))
    rows = []
    for tag in tags:
        for e in ea.Scalars(tag):
            rows.append(
                {
                    "run_dir": run.run_dir,
                    "condition": run.condition,
                    "seed": run.seed,
                    "observation_dim": run.observation_dim,
                    "tag": tag,
                    "step": int(e.step),
                    "wall_time": float(e.wall_time),
                    "value": float(e.value),
                }
            )
    return pd.DataFrame(rows), tags

def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

def choose_tag(tags: Iterable[str], patterns: Iterable[str]):
    tags = list(tags)
    for pat in patterns:
        matches = [t for t in tags if norm(pat) in norm(t)]
        if matches:
            return sorted(matches, key=lambda x: (len(x), x))[0]
    return None

def plot_metric(metrics, tag, output, title, ylabel):
    d = metrics[metrics["tag"] == tag].copy()
    if d.empty:
        return
    agg = (
        d.groupby(["condition", "step"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    plt.figure(figsize=(10, 6))
    for condition, g in agg.groupby("condition", sort=False):
        g = g.sort_values("step")
        plt.plot(g["step"], g["mean"], label=condition)
        std = g["std"].fillna(0)
        plt.fill_between(g["step"], g["mean"] - std, g["mean"] + std, alpha=0.15)
    plt.title(title)
    plt.xlabel("Training step")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()

def summarize_runs(metrics, reward_tag, success_tag, length_tag):
    rows = []
    keys = ["run_dir", "condition", "seed", "observation_dim"]
    for values, group in metrics.groupby(keys, sort=False):
        row = dict(zip(keys, values))
        for name, tag in [
            ("reward", reward_tag),
            ("success_rate", success_tag),
            ("episode_length", length_tag),
        ]:
            m = group[group["tag"] == tag].sort_values("step") if tag else pd.DataFrame()
            row[f"{name}_final"] = float(m.iloc[-1]["value"]) if not m.empty else math.nan
            row[f"{name}_max"] = float(m["value"].max()) if not m.empty else math.nan
            row[f"{name}_mean_last_5"] = float(m.tail(5)["value"].mean()) if not m.empty else math.nan
        rows.append(row)
    return pd.DataFrame(rows)

def summarize_conditions(run_summary):
    metric_cols = [c for c in run_summary.columns if c not in {"run_dir","condition","seed","observation_dim"}]
    rows = []
    for condition, g in run_summary.groupby("condition", sort=False):
        row = {
            "condition": condition,
            "observation_dim": int(g["observation_dim"].iloc[0]),
            "num_seeds": int(g["seed"].nunique()),
        }
        for col in metric_cols:
            vals = [float(v) for v in g[col].dropna().tolist()]
            row[f"{col}_mean"] = statistics.fmean(vals) if vals else math.nan
            row[f"{col}_std"] = statistics.stdev(vals) if len(vals) >= 2 else math.nan
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    frames = []
    tag_rows = []
    runs = load_runs(args.log_root.resolve())

    for run in runs:
        frame, tags = load_scalars(run)
        frames.append(frame)
        for tag in tags:
            tag_rows.append({"run_dir": run.run_dir, "condition": run.condition, "seed": run.seed, "tag": tag})
        print(f"[OK] {run.run_dir} {run.condition} seed={run.seed} tags={len(tags)}")

    metrics = pd.concat(frames, ignore_index=True)
    tags = sorted(metrics["tag"].unique().tolist())

    reward_tag = choose_tag(tags, ["rewards/iter", "reward"])
    success_tag = choose_tag(tags, ["success_rate", "successes", "success"])
    length_tag = choose_tag(tags, ["episode_length", "ep_length", "length"])

    run_summary = summarize_runs(metrics, reward_tag, success_tag, length_tag)
    condition_summary = summarize_conditions(run_summary)

    pd.DataFrame(tag_rows).drop_duplicates().to_csv(out / "available_tags.csv", index=False)
    metrics.to_csv(out / "metrics_long.csv", index=False)
    run_summary.to_csv(out / "run_summary.csv", index=False)
    condition_summary.to_csv(out / "condition_summary.csv", index=False)

    if reward_tag:
        plot_metric(metrics, reward_tag, out / "reward_curve.png", f"Reward ({reward_tag})", "Reward")
    if success_tag:
        plot_metric(metrics, success_tag, out / "success_rate_curve.png", f"Success rate ({success_tag})", "Success rate")
    if length_tag:
        plot_metric(metrics, length_tag, out / "episode_length_curve.png", f"Episode length ({length_tag})", "Episode length")

    report = [
        "# TensorBoard analysis report",
        "",
        f"- Runs found: {len(runs)} / {len(RUNS)}",
        f"- Reward tag: `{reward_tag}`",
        f"- Success tag: `{success_tag}`",
        f"- Episode length tag: `{length_tag}`",
        "",
        "## Condition summary",
        "",
        condition_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Available scalar tags",
        "",
        *[f"- `{t}`" for t in tags],
    ]
    (out / "analysis_report.md").write_text("\n".join(report), encoding="utf-8")

    print(f"[DONE] Output: {out}")
    print(f"Reward tag: {reward_tag}")
    print(f"Success tag: {success_tag}")
    print(f"Episode length tag: {length_tag}")

if __name__ == "__main__":
    main()

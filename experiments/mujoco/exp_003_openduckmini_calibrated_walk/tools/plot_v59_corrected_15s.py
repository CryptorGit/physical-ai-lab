#!/usr/bin/env python3
"""Small evidence plots for the corrected diagnostic; CSV remains canonical."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    plots = root / "plots"
    plots.mkdir(exist_ok=True)
    rows = list(csv.DictReader((root / "command_summary.csv").open(encoding="utf-8")))
    commands = [row["command_id"] for row in rows if row["condition"] == "D"]
    x = np.arange(len(commands))
    d = {row["command_id"]: row for row in rows if row["condition"] == "D"}
    s = {row["command_id"]: row for row in rows if row["condition"] == "S"}

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].bar(x - 0.2, [float(d[c]["vx_mean"]) for c in commands], 0.4, label="D")
    axes[0].bar(x + 0.2, [float(s[c]["vx_mean"]) for c in commands], 0.4, label="S")
    axes[0].plot(x, [float(d[c]["vx"]) for c in commands], "k.", label="command")
    axes[0].set_ylabel("vx (m/s)")
    axes[0].legend()
    axes[1].bar(x - 0.2, [float(d[c]["yaw_rate_mean"]) for c in commands], 0.4, label="D")
    axes[1].bar(x + 0.2, [float(s[c]["yaw_rate_mean"]) for c in commands], 0.4, label="S")
    axes[1].plot(x, [float(d[c]["yaw_rate"]) for c in commands], "k.", label="command")
    axes[1].set_ylabel("yaw rate (rad/s)")
    axes[1].set_xticks(x, commands, rotation=70, ha="right")
    fig.tight_layout()
    fig.savefig(plots / "command_tracking_d_vs_s.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.bar(x - 0.2, [int(d[c]["fall_count"]) for c in commands], 0.4, label="D")
    ax.bar(x + 0.2, [int(s[c]["fall_count"]) for c in commands], 0.4, label="S")
    ax.set_ylabel("falls / 5 diagnostic episodes")
    ax.set_xticks(x, commands, rotation=70, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "fall_counts_d_vs_s.png", dpi=160)
    plt.close(fig)

    counter = list(
        csv.DictReader(
            (root / "command_progress_counterfactual.csv").open(encoding="utf-8")
        )
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    for command_id in ("C05_yaw_left", "C06_yaw_right"):
        selected = [
            row
            for row in counter
            if row["command_id"] == command_id
        ]
        ax.plot(
            [float(row["yaw_tracking_ratio"]) for row in selected],
            [float(row["yaw_related_total"]) for row in selected],
            marker="o",
            label=command_id,
        )
    ax.axvline(1.0, color="black", linestyle="--", label="commanded")
    ax.set_xlabel("actual yaw / commanded yaw")
    ax.set_ylabel("yaw-related reward per step, before dt")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "yaw_counterfactual.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()

"""Aggregate every available run without selecting seeds by performance."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LABELS = {"soft": "SAC", "hard_adaptive": "TRADE"}
COLORS = {"soft": "#1f77b4", "hard_adaptive": "#d62728"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="results/SUITE/ENV directory")
    parser.add_argument("--metric", choices=["returns", "success_rates"], default="returns")
    args = parser.parse_args()
    fig, ax = plt.subplots(figsize=(8, 5))
    found = False
    for mode, label in LABELS.items():
        runs = []
        for path in sorted((args.directory / mode).glob("seed_*/curves.npz")):
            if not (path.parent / "summary.json").exists():
                continue
            with np.load(path) as data:
                x, y = data["steps"], data[args.metric]
                if len(x) and np.isfinite(y).all():
                    runs.append((x.copy(), y.copy()))
        if not runs:
            continue
        start, stop = max(x[0] for x, _ in runs), min(x[-1] for x, _ in runs)
        grid = np.unique(np.concatenate([x[(x >= start) & (x <= stop)] for x, _ in runs]))
        if not len(grid):
            continue
        values = np.stack([np.interp(grid, x, y) for x, y in runs])
        mean, std = values.mean(0), values.std(0, ddof=1 if len(runs) > 1 else 0)
        ax.plot(grid, mean, label=label, color=COLORS[mode], linewidth=2)
        ax.fill_between(grid, mean-std, mean+std, color=COLORS[mode], alpha=0.15)
        found = True
    if not found:
        raise ValueError("No completed runs with the requested metric")
    ax.set(xlabel="Steps", ylabel="Average Return" if args.metric == "returns" else "Success rate")
    if args.metric == "success_rates":
        ax.set_ylim(0, 1)
        ax.set_yticks([0, .5, 1])
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.directory / (args.metric + ".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()

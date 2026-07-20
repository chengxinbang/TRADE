import argparse
import glob
import os
import pickle
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

# Compatibility for pickles saved under NumPy versions that reference
# internal module paths like numpy._core.*.
import numpy.core as _np_core

sys.modules.setdefault("numpy._core", _np_core)
sys.modules.setdefault("numpy._core.multiarray", _np_core.multiarray)
sys.modules.setdefault("numpy._core.numeric", _np_core.numeric)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import config as cfg

SEED_PATTERN = re.compile(r"seed(\d+)")
PLOT_MAX_STEPS = 10_000_000


def _get_step_limit():
    total_steps = getattr(cfg, "TOTAL_STEPS", PLOT_MAX_STEPS)
    try:
        total_steps = int(total_steps)
    except (TypeError, ValueError):
        total_steps = PLOT_MAX_STEPS
    return min(total_steps, PLOT_MAX_STEPS)


def _truncate_by_step(steps, values, max_step):
    steps = np.asarray(steps, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if steps.size == 0 or values.size == 0:
        return steps, values

    n = min(steps.size, values.size)
    steps = steps[:n]
    values = values[:n]

    mask = steps <= max_step
    return steps[mask], values[mask]


def _interp_to_grid(xs, ys, common_grid):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if len(xs) == 0 or len(ys) == 0:
        return None

    if len(xs) != len(ys):
        n = min(len(xs), len(ys))
        xs = xs[:n]
        ys = ys[:n]

    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]

    xs_unique, unique_idx = np.unique(xs, return_index=True)
    ys_unique = ys[unique_idx]

    if len(xs_unique) == 0:
        return None

    if len(xs_unique) == 1:
        out = np.full_like(common_grid, np.nan, dtype=np.float64)
        idx = int(np.argmin(np.abs(common_grid - xs_unique[0])))
        out[idx] = ys_unique[0]
        return out

    return np.interp(common_grid, xs_unique, ys_unique, left=np.nan, right=np.nan)


def _infer_mode(data, pkl_path):
    if "is_fixed" in data:
        return "ddqn" if bool(data["is_fixed"]) else "hard_adaptive"

    name = os.path.basename(pkl_path).lower()
    if "fixed" in name:
        return "ddqn"
    if "adaptive" in name:
        return "hard_adaptive"
    return "unknown"


def _infer_seed(data, pkl_path):
    if "seed" in data:
        return int(data["seed"])

    m = SEED_PATTERN.search(os.path.basename(pkl_path))
    if m:
        return int(m.group(1))
    return None


def _resolve_curve_values(data):
    if "returns" in data and len(data["returns"]) > 0:
        return data["returns"]
    if "success_rates" in data and len(data["success_rates"]) > 0:
        return data["success_rates"]
    return []


def _score_returns(returns, metric="mean", tail_k=5):
    returns = np.asarray(returns, dtype=np.float64)
    if len(returns) == 0:
        return None

    if metric == "final":
        return float(returns[-1])

    if metric == "mean":
        return float(np.mean(returns))

    k = max(1, min(int(tail_k), len(returns)))
    return float(np.mean(returns[-k:]))


def load_runs(task_name):
    env_dir = cfg.get_env_paths(task_name)["env_dir"]
    runs = []
    step_limit = _get_step_limit()

    pattern = os.path.join(env_dir, "**", "*.pkl")
    for pkl_path in sorted(glob.glob(pattern, recursive=True)):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        mode = _infer_mode(data, pkl_path)
        if mode not in ("ddqn", "hard_adaptive"):
            continue

        seed = _infer_seed(data, pkl_path)
        step_records = data.get("step_records", [])
        returns = _resolve_curve_values(data)

        steps, returns = _truncate_by_step(step_records, returns, step_limit)

        if seed is None or steps.size == 0 or returns.size == 0:
            continue

        runs.append(
            {
                "mode": mode,
                "seed": seed,
                "path": pkl_path,
                "steps": steps,
                "returns": returns,
            }
        )

    return runs


def _aggregate_by_seed(runs, mode, metric, tail_k):
    grouped = {}
    for r in runs:
        if r["mode"] != mode:
            continue

        score = _score_returns(r["returns"], metric=metric, tail_k=tail_k)
        if score is None:
            continue

        seed = int(r["seed"])
        if seed not in grouped:
            grouped[seed] = {
                "seed": seed,
                "scores": [],
                "runs": [],
            }

        grouped[seed]["scores"].append(score)
        grouped[seed]["runs"].append(r)

    aggregated = []
    for seed, g in grouped.items():
        # For one-seed-multiple-runs scenarios, rank by the mean seed score.
        seed_score = float(np.mean(g["scores"]))

        # Representative run for plotting: the run with score closest to seed mean.
        idx = int(np.argmin(np.abs(np.asarray(g["scores"], dtype=np.float64) - seed_score)))
        rep_run = g["runs"][idx]

        aggregated.append(
            {
                "mode": mode,
                "seed": seed,
                "score": seed_score,
                "path": rep_run["path"],
                "steps": rep_run["steps"],
                "returns": rep_run["returns"],
                "num_runs": len(g["runs"]),
            }
        )

    return aggregated


def select_runs(runs, mode, top_k, metric, tail_k):
    pool = _aggregate_by_seed(runs, mode=mode, metric=metric, tail_k=tail_k)

    if mode == "hard_adaptive":
        pool.sort(key=lambda x: x["score"], reverse=True)  # best first
    else:
        pool.sort(key=lambda x: x["score"], reverse=False)  # worst first

    return pool[: max(1, int(top_k))], pool


def _plot_group(common_grid, selected_runs, label, color):
    curves = []
    for r in selected_runs:
        interp = _interp_to_grid(r["steps"], r["returns"], common_grid)
        if interp is not None:
            curves.append(interp)

    if not curves:
        return False

    arr = np.array(curves)
    valid_counts = np.sum(~np.isnan(arr), axis=0)
    if not np.any(valid_counts > 0):
        return False

    marr = np.ma.masked_invalid(arr)
    mean = marr.mean(axis=0).filled(np.nan)
    std = marr.std(axis=0).filled(np.nan)
    std = np.where(valid_counts >= 2, std, 0.0)

    valid_mask = valid_counts > 0

    plt.plot(common_grid, mean, color=color, linewidth=2.0, label=label)
    plt.fill_between(
        common_grid,
        mean - std,
        mean + std,
        where=valid_mask,
        color=color,
        alpha=0.2,
        interpolate=True,
    )
    return True


def _build_common_grid(selected_runs, min_grid_points=0):
    step_arrays = []
    for r in selected_runs:
        steps = np.asarray(r.get("steps", []), dtype=np.float64)
        if len(steps) == 0:
            continue

        steps = np.unique(np.sort(steps))
        if len(steps) > 0:
            step_arrays.append(steps)

    if not step_arrays:
        return None

    native_grid = np.unique(np.concatenate(step_arrays))
    if len(native_grid) == 0:
        return None

    # Keep origin for visual alignment with other plots in this project.
    native_grid = np.unique(np.concatenate([np.array([0.0], dtype=np.float64), native_grid]))

    min_grid_points = int(min_grid_points)
    if min_grid_points <= 0 or len(native_grid) >= min_grid_points:
        return native_grid

    dense_grid = np.linspace(native_grid[0], native_grid[-1], num=min_grid_points)
    return np.unique(np.concatenate([native_grid, dense_grid]))


def _discover_env_ids_with_data():
    discovered = []
    if not os.path.isdir(cfg.BASE_RESULTS_DIR):
        return discovered

    for name in sorted(os.listdir(cfg.BASE_RESULTS_DIR)):
        env_dir = os.path.join(cfg.BASE_RESULTS_DIR, name)
        if not os.path.isdir(env_dir):
            continue
        if glob.glob(os.path.join(env_dir, "**", "*.pkl"), recursive=True):
            discovered.append(name)
    return discovered


def _default_out_dir(task):
    paths = cfg.get_env_paths(task)
    plot_root_dir = paths.get("plot_root_dir", paths["env_dir"])

    if cfg.PLOT_USE_TIMESTAMP_DIR:
        from datetime import datetime

        out_dir = os.path.join(plot_root_dir, f"plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    else:
        out_dir = os.path.join(plot_root_dir, cfg.PLOT_OUTPUT_SUBDIR)

    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def main(args):
    task = args.env

    if not task:
        discovered = _discover_env_ids_with_data()
        if not discovered:
            raise RuntimeError("No environment with pkl data found under ddqn_adaptive_results.")
        task = discovered[0]
        print(f"[Info] --env not provided, use discovered env: {task}")

    runs = load_runs(task)
    if not runs:
        raise RuntimeError(f"No valid pkl runs found for env: {task}")

    hard_selected, hard_all = select_runs(
        runs,
        mode="hard_adaptive",
        top_k=args.hard_top_k,
        metric=args.metric,
        tail_k=args.tail_k,
    )
    ddqn_selected, ddqn_all = select_runs(
        runs,
        mode="ddqn",
        top_k=args.ddqn_bottom_k,
        metric=args.metric,
        tail_k=args.tail_k,
    )

    if len(hard_selected) == 0:
        raise RuntimeError("No selectable hard_adaptive runs found.")
    if len(ddqn_selected) == 0:
        raise RuntimeError("No selectable ddqn runs found.")

    common_grid = _build_common_grid(
        hard_selected + ddqn_selected,
        min_grid_points=max(0, int(args.grid_points)),
    )
    if common_grid is None or len(common_grid) == 0:
        raise RuntimeError("Failed to build common plotting grid from selected runs.")

    save_dir = args.out_dir if args.out_dir else _default_out_dir(task)
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    ddqn_ok = _plot_group(
        common_grid,
        ddqn_selected,
        label="DDQN (Bottom Seeds)",
        color="#1f77b4",
    )
    hard_ok = _plot_group(
        common_grid,
        hard_selected,
        label="Hard Adaptive (Top Seeds)",
        color="#d62728",
    )

    if not (ddqn_ok or hard_ok):
        raise RuntimeError("Failed to create any curve from selected runs.")

    plt.xlabel("Steps")
    plt.ylabel("Average Return")
    plt.title(f"{task}")
    plt.grid(alpha=0.3)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.legend()

    figure_path = os.path.join(save_dir, f"return_comparison_{task}.png")
    plt.savefig(figure_path, bbox_inches="tight")
    plt.close()

    rank_file = os.path.join(save_dir, f"selected_seed_ranking_{task}.txt")
    with open(rank_file, "w", encoding="utf-8") as f:
        f.write(f"env={task}\n")
        f.write(f"metric={args.metric}\n")
        f.write(f"tail_k={args.tail_k}\n\n")

        f.write("[ddqn all | ascending by mean-seed-score (worst first)]\n")
        for i, r in enumerate(ddqn_all, start=1):
            f.write(
                f"{i:02d}. seed={r['seed']} score={r['score']:.6f} runs={r['num_runs']} file={os.path.basename(r['path'])}\n"
            )

        f.write("\n[hard_adaptive all | descending by mean-seed-score (best first)]\n")
        for i, r in enumerate(hard_all, start=1):
            f.write(
                f"{i:02d}. seed={r['seed']} score={r['score']:.6f} runs={r['num_runs']} file={os.path.basename(r['path'])}\n"
            )

        f.write("\n[selected ddqn bottom]\n")
        for r in ddqn_selected:
            f.write(f"seed={r['seed']} score={r['score']:.6f}\n")

        f.write("\n[selected hard_adaptive top]\n")
        for r in hard_selected:
            f.write(f"seed={r['seed']} score={r['score']:.6f}\n")

    print("[Selection] DDQN bottom seeds:")
    for r in ddqn_selected:
        print(f"  seed={r['seed']} score={r['score']:.6f}")

    print("[Selection] hard_adaptive top seeds:")
    for r in hard_selected:
        print(f"  seed={r['seed']} score={r['score']:.6f}")

    print(f"[Output] figure: {figure_path}")
    print(f"[Output] ranking: {rank_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="")
    parser.add_argument("--ddqn_bottom_k", type=int, default=3)
    parser.add_argument("--hard_top_k", type=int, default=3)
    parser.add_argument(
        "--metric",
        type=str,
        default="mean",
        choices=["mean", "final", "tail_mean"],
        help="Seed ranking metric: mean = mean of all eval returns, final = last eval return, tail_mean = mean of last K eval returns.",
    )
    parser.add_argument("--tail_k", type=int, default=5)
    parser.add_argument(
        "--grid_points",
        type=int,
        default=0,
        help="Minimum interpolation grid points. 0 = use native eval steps (denser, closer to raw curves).",
    )
    parser.add_argument("--out_dir", type=str, default="")
    main(parser.parse_args())

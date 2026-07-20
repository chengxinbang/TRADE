import argparse
import csv
import glob
import os
import pickle
import queue
import sys


def _sanitize_thread_env_vars(default_threads="1"):
    thread_vars = [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]
    for key in thread_vars:
        raw = os.environ.get(key)
        if raw is None:
            continue
        try:
            val = int(str(raw).strip())
            if val <= 0:
                raise ValueError
        except Exception:
            os.environ[key] = str(default_threads)


_sanitize_thread_env_vars(default_threads="1")

import matplotlib.pyplot as plt
import numpy as np
import torch
import multiprocessing as mp

import config


DEFAULT_ADAPTIVE_EPSILON_TOLS = [0.0, 0.1, 0.001, 0.0001, 0.00001]
SENSITIVITY_ROOT = os.path.join(config.BASE_RESULTS_DIR, "adaptive_epsilon_tol_sensitivity")


def _parse_nonnegative_float(value):
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("adaptive_epsilon_tol must be a number.") from exc
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("adaptive_epsilon_tol must be >= 0.")
    return parsed


def _parse_positive_int(value):
    text = str(value).replace("_", "")
    try:
        parsed = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("total_steps must be a positive integer.") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("total_steps must be a positive integer.")
    return parsed


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run hard_adaptive sensitivity analysis for adaptive_epsilon_tol."
    )
    parser.add_argument(
        "--adaptive-epsilon-tols",
        nargs="+",
        type=_parse_nonnegative_float,
        default=DEFAULT_ADAPTIVE_EPSILON_TOLS,
        help="Values of adaptive_epsilon_tol to run, e.g. 0 0.1 0.001 0.0001 0.00001.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(config.SEEDS),
        help="Seeds to run.",
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        default=list(config.ENV_IDS),
        help="Environment IDs to run, e.g. BreakoutNoFrameskip-v4 EnduroNoFrameskip-v4.",
    )
    parser.add_argument(
        "--total-steps",
        type=_parse_positive_int,
        default=int(config.TOTAL_STEPS),
        help=f"Total training steps for each run. Default: {config.TOTAL_STEPS}.",
    )
    return parser.parse_args()


def _resolve_runtime_gpus(config_gpu_ids):
    visible_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if visible_gpu_count <= 0:
        return []
    return [gid for gid in config_gpu_ids if 0 <= gid < visible_gpu_count]


def _unique_preserve_order(values):
    out = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        out.append(value)
        seen.add(value)
    return out


def _tau_label(tau):
    text = f"{float(tau):.12g}"
    return text


def _safe_tau_dir_name(tau):
    text = _tau_label(tau)
    return "tau_" + text.replace("+", "").replace("-", "m").replace(".", "p")


def _steps_dir_name(total_steps):
    return f"steps_{int(total_steps)}"


def _get_run_root(total_steps):
    total_steps = int(total_steps)
    if total_steps == int(config.TOTAL_STEPS):
        return SENSITIVITY_ROOT
    return os.path.join(SENSITIVITY_ROOT, _steps_dir_name(total_steps))


def _get_sensitivity_paths(env_id, tau, total_steps):
    env_dir = os.path.join(_get_run_root(total_steps), _safe_tau_dir_name(tau), env_id)
    pth_dir = os.path.join(env_dir, "checkpoints")
    os.makedirs(env_dir, exist_ok=True)
    os.makedirs(pth_dir, exist_ok=True)
    return {
        "env_dir": env_dir,
        "pth_dir": pth_dir,
        "plot_root_dir": env_dir,
    }


def _load_pickle_compat(pkl_path):
    with open(pkl_path, "rb") as f_pkl:
        try:
            return pickle.load(f_pkl)
        except ModuleNotFoundError as exc:
            if getattr(exc, "name", "") and exc.name.startswith("numpy._core"):
                sys.modules.setdefault("numpy._core", np.core)
                sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
                f_pkl.seek(0)
                return pickle.load(f_pkl)
            raise


def _score_average_return(data):
    returns = np.asarray(data.get("returns", []), dtype=np.float64)
    if returns.size == 0:
        return None
    return float(np.mean(returns))


def _collect_tau_scores(tau, seeds, envs, total_steps):
    seed_set = {int(seed) for seed in seeds}
    scores = []
    for env_id in envs:
        paths = _get_sensitivity_paths(env_id, tau, total_steps)
        pkl_files = sorted(glob.glob(os.path.join(paths["env_dir"], "adaptive_*_seed*.pkl")))
        for pkl_path in pkl_files:
            data = _load_pickle_compat(pkl_path)
            if bool(data.get("is_fixed", True)):
                continue
            seed = int(data.get("seed", -1))
            if seed not in seed_set:
                continue
            saved_total_steps = data.get("total_steps")
            if saved_total_steps is not None and int(saved_total_steps) != int(total_steps):
                continue
            score = _score_average_return(data)
            if score is None:
                continue
            scores.append(
                {
                    "tau": float(tau),
                    "total_steps": int(total_steps),
                    "env_id": data.get("env_id", env_id),
                    "seed": seed,
                    "average_return_mean": score,
                    "elapsed_time_sec": float(data.get("elapsed_time_sec", 0.0)),
                    "path": pkl_path,
                }
            )
    return scores


def _write_summary_csv(rows, output_path):
    fieldnames = [
        "tau",
        "total_steps",
        "mean_average_return",
        "std_average_return",
        "num_runs",
        "total_elapsed_time_sec",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_sensitivity(summary_rows, output_path):
    taus = [row["tau"] for row in summary_rows]
    means = [row["mean_average_return"] for row in summary_rows]
    stds = [row["std_average_return"] for row in summary_rows]
    x = np.arange(len(taus), dtype=np.float64)

    plt.figure(figsize=(8, 5))
    plt.plot(x, means, marker="o", linewidth=2.0, color="#d62728")
    plt.fill_between(
        x,
        np.asarray(means, dtype=np.float64) - np.asarray(stds, dtype=np.float64),
        np.asarray(means, dtype=np.float64) + np.asarray(stds, dtype=np.float64),
        color="#d62728",
        alpha=0.18,
    )
    plt.xticks(x, [_tau_label(tau) for tau in taus])
    plt.xlabel("τ", fontsize=12)
    plt.ylabel("Mean Average Return", fontsize=12)
    plt.title("Sensitivity of adaptive_epsilon_tol", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()


def _build_summary(tols, seeds, envs, total_steps):
    summary_rows = []
    detail_rows = []
    for tau in tols:
        scores = _collect_tau_scores(tau, seeds, envs, total_steps)
        detail_rows.extend(scores)
        values = np.asarray([row["average_return_mean"] for row in scores], dtype=np.float64)
        elapsed = float(np.sum([row["elapsed_time_sec"] for row in scores]))
        if values.size == 0:
            mean_score = float("nan")
            std_score = float("nan")
        else:
            mean_score = float(np.mean(values))
            std_score = float(np.std(values))
        summary_rows.append(
            {
                "tau": float(tau),
                "total_steps": int(total_steps),
                "mean_average_return": mean_score,
                "std_average_return": std_score,
                "num_runs": int(values.size),
                "total_elapsed_time_sec": elapsed,
            }
        )
    return summary_rows, detail_rows


def _write_detail_csv(rows, output_path):
    fieldnames = ["tau", "total_steps", "env_id", "seed", "average_return_mean", "elapsed_time_sec", "path"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def worker_wrapper(task_queue, result_queue, gpu_id):
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import train_worker

    train_worker.run_single_training(task_queue, result_queue, gpu_id)


def run_experiments(tols, seeds, envs, total_steps):
    runtime_gpus = _resolve_runtime_gpus(config.AVAILABLE_GPUS)
    runtime_max_parallel = max(1, len(runtime_gpus) * config.NUM_WORKERS_PER_GPU) if runtime_gpus else 1

    all_tasks = []
    for tau in tols:
        for env_id in envs:
            paths = _get_sensitivity_paths(env_id, tau, total_steps)
            for seed in seeds:
                all_tasks.append(
                    (
                        env_id,
                        int(seed),
                        False,
                        paths,
                        config.ADAPTIVE_PARAMS.get("adaptive_C_min"),
                        float(tau),
                        int(total_steps),
                    )
                )

    if not all_tasks:
        raise RuntimeError("No sensitivity tasks were created.")

    print("=" * 80)
    print("hard_adaptive adaptive_epsilon_tol sensitivity analysis")
    print(f"Environments: {len(envs)}")
    print(f"Environment IDs: {envs}")
    print(f"Seeds: {seeds}")
    print(f"Total steps: {total_steps}")
    print(f"adaptive_epsilon_tol values: {[_tau_label(t) for t in tols]}")
    print(f"adaptive_C_min: {config.ADAPTIVE_PARAMS.get('adaptive_C_min')}")
    print(f"Runtime GPUs: {runtime_gpus if runtime_gpus else 'CPU'}")
    print(f"Max parallel tasks: {runtime_max_parallel}")
    print(f"Output root: {_get_run_root(total_steps)}")
    print("=" * 80)

    task_queue = mp.Queue()
    result_queue = mp.Queue()
    for task in all_tasks:
        task_queue.put(task)

    worker_count = min(runtime_max_parallel, len(all_tasks))
    for _ in range(worker_count):
        task_queue.put(None)

    workers = []
    for i in range(worker_count):
        gpu_id = runtime_gpus[i % len(runtime_gpus)] if runtime_gpus else None
        p = mp.Process(target=worker_wrapper, args=(task_queue, result_queue, gpu_id))
        p.start()
        workers.append(p)

    finished_count = 0
    active_workers = len(workers)
    try:
        while active_workers > 0:
            try:
                result = result_queue.get(timeout=5)
            except queue.Empty:
                continue

            if result is None:
                active_workers -= 1
                continue

            finished_count += 1
            if isinstance(result, dict) and result.get("status") == "error":
                print(
                    f"\r[Progress] {finished_count}/{len(all_tasks)} | "
                    f"{result.get('env_id')} S{result.get('seed')} failed: {result.get('error')}"
                )
                continue

            print(
                f"\r[Progress] {finished_count}/{len(all_tasks)} | "
                f"{result['env_id']} S{result['seed']} tau={_tau_label(result.get('adaptive_epsilon_tol', np.nan))} done",
                end="",
                flush=True,
            )
    except KeyboardInterrupt:
        print("\nStopping workers...")
    finally:
        print("\nCleaning workers...")
        for p in workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)

        for q in (task_queue, result_queue):
            try:
                q.close()
                q.join_thread()
            except Exception:
                pass


def main():
    args = _parse_args()
    tols = _unique_preserve_order(args.adaptive_epsilon_tols)
    seeds = _unique_preserve_order(args.seeds)
    envs = _unique_preserve_order(args.envs)
    total_steps = int(args.total_steps)

    run_experiments(tols, seeds, envs, total_steps)

    output_root = _get_run_root(total_steps)
    os.makedirs(output_root, exist_ok=True)
    summary_rows, detail_rows = _build_summary(tols, seeds, envs, total_steps)

    summary_csv = os.path.join(output_root, "adaptive_epsilon_tol_sensitivity_summary.csv")
    detail_csv = os.path.join(output_root, "adaptive_epsilon_tol_sensitivity_details.csv")
    figure_path = os.path.join(output_root, "adaptive_epsilon_tol_sensitivity.png")

    _write_summary_csv(summary_rows, summary_csv)
    _write_detail_csv(detail_rows, detail_csv)
    _plot_sensitivity(summary_rows, figure_path)

    print("\nSensitivity summary:")
    for row in summary_rows:
        print(
            f"  tau={_tau_label(row['tau'])} | "
            f"mean_average_return={row['mean_average_return']:.6f} | "
            f"runs={row['num_runs']}"
        )
    print(f"[Output] figure: {figure_path}")
    print(f"[Output] summary: {summary_csv}")
    print(f"[Output] details: {detail_csv}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

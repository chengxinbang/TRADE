import os
import sys
import argparse


def _sanitize_thread_env_vars(default_threads="1"):
    """修正线程环境变量，避免 libgomp 因非法值报错。"""
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

import torch
import multiprocessing as mp
import queue
from collections import defaultdict

import config
from plot_utils import plot_env_results


MODE_TO_IS_FIXED = {
    "ddqn": True,
    "hard_adaptive": False,
}


def _parse_optional_positive_int(value):
    text = str(value).strip().lower()
    if text in ("none", "null"):
        return None
    try:
        parsed = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("必须是正整数，或 none/null") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return parsed


def _parse_positive_float(value):
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("必须是正浮点数") from exc
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("必须是正浮点数")
    return parsed


def _resolve_runtime_gpus(config_gpu_ids):
    """Return GPU ids that are actually visible to the current process."""
    visible_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if visible_gpu_count <= 0:
        return []
    return [gid for gid in config_gpu_ids if 0 <= gid < visible_gpu_count]


def _parse_args():
    parser = argparse.ArgumentParser(
        description="DDQN 自适应目标网络更新 - 多环境并行训练"
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        default=None,
        help=(
            "指定要训练的环境ID列表（空格分隔）。"
            "例如: --envs PongNoFrameskip-v4 BreakoutNoFrameskip-v4"
        ),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=None,
        help=(
            "指定要训练的模式列表（空格分隔）。"
            "可选: ddqn hard_adaptive。"
            "例如: --modes ddqn hard_adaptive"
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help=(
            "指定要训练的随机种子列表（空格分隔）。"
            "例如: --seeds 1 3"
        ),
    )
    parser.add_argument(
        "--adaptive-c-min",
        type=_parse_optional_positive_int,
        default=config.ADAPTIVE_PARAMS.get("adaptive_C_min"),
        help=(
            "自适应更新的最小间隔 C_min。"
            "传 none/null 表示按公式自动估计。"
            f"默认: {config.ADAPTIVE_PARAMS.get('adaptive_C_min')}"
        ),
    )
    parser.add_argument(
        "--adaptive-epsilon-tol",
        type=_parse_positive_float,
        default=float(config.ADAPTIVE_PARAMS["adaptive_epsilon_tol"]),
        help=(
            "自适应更新中的阈值 epsilon_tol（正浮点数）。"
            f"默认: {config.ADAPTIVE_PARAMS['adaptive_epsilon_tol']}"
        ),
    )
    return parser.parse_args()


def _resolve_selected_envs(cli_envs):
    if not cli_envs:
        return list(config.ENV_IDS)

    unknown = [env for env in cli_envs if env not in config.ENV_IDS]
    if unknown:
        raise ValueError(
            "以下环境不在 config.ENV_IDS 中: "
            + ", ".join(unknown)
            + "\n可选环境: "
            + ", ".join(config.ENV_IDS)
        )

    # 去重且保持输入顺序
    selected_envs = []
    seen = set()
    for env in cli_envs:
        if env not in seen:
            selected_envs.append(env)
            seen.add(env)
    return selected_envs


def _resolve_selected_modes(cli_modes):
    if not cli_modes:
        return list(MODE_TO_IS_FIXED.keys())

    normalized = [mode.lower() for mode in cli_modes]
    unknown = [mode for mode in normalized if mode not in MODE_TO_IS_FIXED]
    if unknown:
        raise ValueError(
            "以下模式不受支持: "
            + ", ".join(unknown)
            + "\n可选模式: "
            + ", ".join(MODE_TO_IS_FIXED.keys())
        )

    # 去重且保持输入顺序
    selected_modes = []
    seen = set()
    for mode in normalized:
        if mode not in seen:
            selected_modes.append(mode)
            seen.add(mode)
    return selected_modes


def _resolve_selected_seeds(cli_seeds):
    if not cli_seeds:
        return list(config.SEEDS)

    unknown = [seed for seed in cli_seeds if seed not in config.SEEDS]
    if unknown:
        raise ValueError(
            "以下种子不在 config.SEEDS 中: "
            + ", ".join(map(str, unknown))
            + "\n可选种子: "
            + ", ".join(map(str, config.SEEDS))
        )

    # 去重且保持输入顺序
    selected_seeds = []
    seen = set()
    for seed in cli_seeds:
        if seed not in seen:
            selected_seeds.append(seed)
            seen.add(seed)
    return selected_seeds


def main():
    args = _parse_args()
    selected_envs = _resolve_selected_envs(args.envs)
    selected_modes = _resolve_selected_modes(args.modes)
    selected_seeds = _resolve_selected_seeds(args.seeds)
    selected_adaptive_c_min = args.adaptive_c_min
    selected_adaptive_epsilon_tol = args.adaptive_epsilon_tol

    runtime_gpus = _resolve_runtime_gpus(config.AVAILABLE_GPUS)
    runtime_max_parallel = max(1, len(runtime_gpus) * config.NUM_WORKERS_PER_GPU) if runtime_gpus else 1

    print("=" * 80)
    print("DDQN 自适应目标网络更新 - 多环境并行训练系统")
    print(f"配置GPU列表: {config.AVAILABLE_GPUS}")
    print(f"运行时可见GPU: {runtime_gpus if runtime_gpus else '无'}")
    print(f"最大并行任务数(运行时): {runtime_max_parallel}")
    print(f"待训练环境: {len(selected_envs)} 个")
    print(f"环境列表: {selected_envs}")
    print(f"训练模式: {selected_modes}")
    print(f"Seed列表: {selected_seeds}")
    print(f"adaptive_C_min: {selected_adaptive_c_min}")
    print(f"adaptive_epsilon_tol: {selected_adaptive_epsilon_tol}")
    print("=" * 80)

    # ===================== 生成所有任务 =====================
    all_tasks = []
    for env_id in selected_envs:
        paths = config.get_env_paths(env_id)
        for mode in selected_modes:
            is_fixed = MODE_TO_IS_FIXED[mode]
            for seed in selected_seeds:
                all_tasks.append(
                    (
                        env_id,
                        seed,
                        is_fixed,
                        paths,
                        selected_adaptive_c_min,
                        selected_adaptive_epsilon_tol,
                    )
                )

    print(f"\n总任务数: {len(all_tasks)} 个\n")

    # ===================== 多进程启动 =====================
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
        p = mp.Process(
            target=worker_wrapper,
            args=(task_queue, result_queue, gpu_id)
        )
        p.start()
        workers.append(p)

    # ===================== 收集结果 =====================
    all_results = defaultdict(list)
    finished_count = 0
    active_workers = len(workers)

    print("开始训练...\n")

    try:
        while active_workers > 0:
            try:
                result = result_queue.get(timeout=5)
                if result is None:
                    active_workers -= 1
                    continue

                finished_count += 1
                if isinstance(result, dict) and result.get("status") == "error":
                    print(
                        f"\r[进度] {finished_count}/{len(all_tasks)} | "
                        f"{result['env_id']} S{result['seed']} "
                        f"{'Fixed' if result['is_fixed'] else 'Adaptive'} 失败: {result.get('error', 'unknown')}"
                    )
                    continue

                all_results[result['env_id']].append(result)
                print(
                    f"\r[进度] {finished_count}/{len(all_tasks)} | "
                    f"{result['env_id']} S{result['seed']} {'Fixed' if result['is_fixed'] else 'Adaptive'} 完成",
                    end="",
                    flush=True,
                )
            except queue.Empty:
                continue

    except KeyboardInterrupt:
        print("\n\n正在停止...")
    finally:
        print("\n清理进程...")
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

    # ===================== 绘图 =====================
    print("\n\n生成图表...")
    for env_id in selected_envs:
        try:
            plot_env_results(env_id)
        except Exception as e:
            print(f"生成 {env_id} 图表出错: {e}")

    print("\n所有工作完成！")


def worker_wrapper(task_queue, result_queue, gpu_id):
    """Worker包装器：设置环境变量并导入。"""
    # 设置 CUDA_VISIBLE_DEVICES，让每个进程绑定到指定 GPU；None 表示 CPU。
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import train_worker
    train_worker.run_single_training(task_queue, result_queue, gpu_id)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    try:
        main()
    except ValueError as e:
        print(f"参数错误: {e}", file=sys.stderr)
        sys.exit(2)

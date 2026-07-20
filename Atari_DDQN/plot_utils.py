import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pickle
import glob
from datetime import datetime
import config

PLOT_MAX_STEPS = 10_000_000


def _get_step_limit():
    total_steps = getattr(config, "TOTAL_STEPS", PLOT_MAX_STEPS)
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


def smooth_curve(y, window_size=1000):
    if len(y) < window_size:
        return y
    return np.convolve(y, np.ones(window_size) / window_size, mode='valid')


def _setup_plot_style():
    """配置绘图风格，并优先选择可用的中文字体避免方框乱码。"""
    candidate_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans CN",
        "WenQuanYi Zen Hei",
    ]
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    selected_font = next((name for name in candidate_fonts if name in available_fonts), None)

    if selected_font:
        plt.rcParams['font.sans-serif'] = [selected_font, 'DejaVu Sans']
    else:
        # 兜底为英文友好字体，避免出现乱码方框。
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 300


def _interp_runs(run_dicts, step_key, value_key, common_grid, smooth=False, max_step=None):
    curves = []
    for data in run_dicts:
        steps = np.asarray(data.get(step_key, []), dtype=np.float64)
        values = np.asarray(data.get(value_key, []), dtype=np.float64)
        if steps.size == 0 or values.size == 0:
            continue

        if smooth:
            values = smooth_curve(values)
            steps = steps[:len(values)]
            if len(values) == 0:
                continue

        if max_step is not None:
            n = min(steps.size, values.size)
            steps = steps[:n]
            values = values[:n]

            mask = steps <= max_step
            steps = steps[mask]
            values = values[mask]
            if steps.size == 0 or values.size == 0:
                continue

        order = np.argsort(steps)
        steps = steps[order]
        values = values[order]

        # np.interp 要求 x 单调递增，重复 step 仅保留第一次。
        unique_steps, unique_idx = np.unique(steps, return_index=True)
        unique_values = values[unique_idx]
        if unique_steps.size == 0:
            continue

        interp = np.interp(
            common_grid,
            unique_steps,
            unique_values,
            left=unique_values[0],
            right=unique_values[-1],
        )
        curves.append(interp)

    if len(curves) == 0:
        return None, None

    stacked = np.array(curves)
    return np.mean(stacked, axis=0), np.std(stacked, axis=0)


def _load_pickle_compat(pkl_path):
    """兼容不同 numpy 版本之间的 pickle 模块路径差异。"""
    with open(pkl_path, 'rb') as f_pkl:
        try:
            return pickle.load(f_pkl)
        except ModuleNotFoundError as exc:
            if getattr(exc, 'name', '') and exc.name.startswith('numpy._core'):
                sys.modules.setdefault('numpy._core', np.core)
                sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
                f_pkl.seek(0)
                return pickle.load(f_pkl)
            raise


def _create_plot_dir(paths):
    """仅在确认会产出图片时创建图片目录。"""
    plot_root_dir = paths.get("plot_root_dir", paths["env_dir"])
    if config.PLOT_USE_TIMESTAMP_DIR:
        img_dir = os.path.join(plot_root_dir, f"plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    else:
        img_dir = os.path.join(plot_root_dir, config.PLOT_OUTPUT_SUBDIR)
    os.makedirs(img_dir, exist_ok=True)
    return img_dir


def plot_env_results(env_id):
    paths = config.get_env_paths(env_id)
    step_limit = _get_step_limit()

    # 加载所有Pickle数据
    all_data = {}
    pkl_files = glob.glob(os.path.join(paths["env_dir"], "*.pkl"))

    if not pkl_files:
        print(f"[Plot] 未找到 {env_id} 的数据文件")
        return

    for pkl_path in pkl_files:
        data = _load_pickle_compat(pkl_path)
        key = "fixed" if data["is_fixed"] else "adaptive"
        if key not in all_data:
            all_data[key] = []
        all_data[key].append(data)

    if not all_data:
        return

    print(f"[Plot] 开始生成 {env_id} 的图表...")
    _setup_plot_style()

    # 统一Step网格
    if 'fixed' in all_data and len(all_data['fixed']) > 0:
        eval_steps = np.asarray(all_data['fixed'][0]['step_records'], dtype=np.float64)
    elif 'adaptive' in all_data:
        eval_steps = np.asarray(all_data['adaptive'][0]['step_records'], dtype=np.float64)
    else:
        return

    eval_steps = eval_steps[eval_steps <= step_limit]
    if eval_steps.size == 0:
        print(f"[Plot] {env_id} has no eval steps <= {step_limit}")
        return

    # 到这里说明至少会保存第1张图，才创建时间戳文件夹
    img_dir = _create_plot_dir(paths)

    # -------------------- 图1：回报对比 --------------------
    plt.figure(figsize=(10, 6))

    if 'fixed' in all_data:
        f_mean, f_std = _interp_runs(
            all_data['fixed'],
            step_key='step_records',
            value_key='returns',
            common_grid=eval_steps,
            max_step=step_limit,
        )
        if f_mean is not None:
            plt.plot(eval_steps, f_mean, label='DDQN', color='#1f77b4', linewidth=2)
            plt.fill_between(eval_steps, f_mean - f_std, f_mean + f_std, color='#1f77b4', alpha=0.2)

    if 'adaptive' in all_data:
        a_mean, a_std = _interp_runs(
            all_data['adaptive'],
            step_key='step_records',
            value_key='returns',
            common_grid=eval_steps,
            max_step=step_limit,
        )
        if a_mean is not None:
            plt.plot(eval_steps, a_mean, label='Hard adaptive update', color='#d62728', linewidth=2)
            plt.fill_between(eval_steps, a_mean - a_std, a_mean + a_std, color='#d62728', alpha=0.2)

    plt.xlabel('Steps', fontsize=12)
    plt.ylabel('Average Return', fontsize=12)
    plt.title(f'{env_id}', fontsize=14)
    plt.legend(fontsize=12)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(img_dir, f'return_comparison_{env_id}.png'), bbox_inches='tight')
    plt.close()

    # -------------------- 图2：自适应C值变化 --------------------
    if 'adaptive' in all_data:
        common_grid = np.linspace(0, step_limit, num=200)
        all_C_interp = []
        for d in all_data['adaptive']:
            steps, values = _truncate_by_step(d.get('C_steps', []), d.get('C_values', []), step_limit)
            if steps.size == 0 or values.size == 0:
                continue

            order = np.argsort(steps)
            steps = steps[order]
            values = values[order]

            unique_steps, unique_idx = np.unique(steps, return_index=True)
            unique_values = values[unique_idx]
            if unique_steps.size == 0:
                continue

            c_interp = np.interp(
                common_grid,
                unique_steps,
                unique_values,
                left=unique_values[0],
                right=unique_values[-1],
            )
            all_C_interp.append(c_interp)

        if len(all_C_interp) > 0:
            all_C_interp = np.array(all_C_interp)
            c_mean = np.mean(all_C_interp, axis=0)
            c_std = np.std(all_C_interp, axis=0)

            plt.figure(figsize=(10, 6))
            plt.step(common_grid, c_mean, where='post', linewidth=2, color='#1f77b4')
            plt.fill_between(common_grid, c_mean - c_std, c_mean + c_std, color='#1f77b4', alpha=0.2, step='post')
            plt.xlabel('Steps', fontsize=12)
            plt.ylabel('Update Interval C', fontsize=12)
            plt.title(f'Adaptive C Curve ({env_id})', fontsize=14)
            plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(img_dir, f'adaptive_C_curve_{env_id}.png'), bbox_inches='tight')
            plt.close()

    # -------------------- 图3：Loss变化 --------------------
    common_grid = np.linspace(0, step_limit, num=200)
    fixed_loss_mean, fixed_loss_std = (None, None)
    adaptive_loss_mean, adaptive_loss_std = (None, None)

    if 'fixed' in all_data:
        fixed_loss_mean, fixed_loss_std = _interp_runs(
            all_data['fixed'],
            step_key='loss_steps',
            value_key='loss_values',
            common_grid=common_grid,
            smooth=True,
            max_step=step_limit,
        )

    if 'adaptive' in all_data:
        adaptive_loss_mean, adaptive_loss_std = _interp_runs(
            all_data['adaptive'],
            step_key='loss_steps',
            value_key='loss_values',
            common_grid=common_grid,
            smooth=True,
            max_step=step_limit,
        )

    if fixed_loss_mean is not None or adaptive_loss_mean is not None:
        plt.figure(figsize=(10, 6))

        if fixed_loss_mean is not None:
            plt.plot(common_grid, fixed_loss_mean, linewidth=1.8, color='#1f77b4', label=f'DDQN')
            plt.fill_between(
                common_grid,
                fixed_loss_mean - fixed_loss_std,
                fixed_loss_mean + fixed_loss_std,
                color='#1f77b4',
                alpha=0.2,
            )

        if adaptive_loss_mean is not None:
            plt.plot(common_grid, adaptive_loss_mean, linewidth=1.8, color='#d62728', label='Hard adaptive update')
            plt.fill_between(
                common_grid,
                adaptive_loss_mean - adaptive_loss_std,
                adaptive_loss_mean + adaptive_loss_std,
                color='#d62728',
                alpha=0.2,
            )

        plt.xlabel('Steps', fontsize=12)
        plt.ylabel('Critic Loss', fontsize=12)
        plt.title(f'Loss Curve Comparison ({env_id})', fontsize=14)
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(img_dir, f'loss_curve_{env_id}.png'), bbox_inches='tight')
        plt.close()

    print(f"[Plot] {env_id} 图表生成完毕，保存在: {img_dir}")


def _discover_env_ids_with_data():
    """自动发现已有 pkl 数据的环境目录。"""
    discovered = []
    if not os.path.isdir(config.BASE_RESULTS_DIR):
        return discovered

    for name in sorted(os.listdir(config.BASE_RESULTS_DIR)):
        env_dir = os.path.join(config.BASE_RESULTS_DIR, name)
        if not os.path.isdir(env_dir):
            continue
        if glob.glob(os.path.join(env_dir, "*.pkl")):
            discovered.append(name)
    return discovered


def _parse_args():
    parser = argparse.ArgumentParser(
        description="DDQN 实验结果绘图脚本（支持单环境或批量重绘）"
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default=None,
        help="单个环境ID，例如 PongNoFrameskip-v4",
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        default=None,
        help="多个环境ID，空格分隔，例如 --envs PongNoFrameskip-v4 BreakoutNoFrameskip-v4",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="按 config.ENV_IDS 重绘全部环境",
    )
    parser.add_argument(
        "--timestamp-dir",
        action="store_true",
        help="输出到 plots_时间戳 目录（默认使用 config.PLOT_OUTPUT_SUBDIR）",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default=None,
        help="覆盖默认输出子目录名，例如 plots_latest",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    if args.timestamp_dir:
        config.PLOT_USE_TIMESTAMP_DIR = True
    if args.output_subdir:
        config.PLOT_OUTPUT_SUBDIR = args.output_subdir

    if args.all:
        env_ids = list(config.ENV_IDS)
    elif args.envs:
        env_ids = list(dict.fromkeys(args.envs))
    elif args.env_id:
        env_ids = [args.env_id]
    else:
        env_ids = _discover_env_ids_with_data()
        if not env_ids:
            print("[Plot] 未发现可绘图的数据目录，请使用 --env-id/--envs/--all 指定环境")
            return

    print(f"[Plot] 待绘图环境: {env_ids}")
    for env_id in env_ids:
        try:
            plot_env_results(env_id)
        except Exception as exc:
            print(f"[Plot] {env_id} 绘图失败: {exc}")


if __name__ == "__main__":
    main()

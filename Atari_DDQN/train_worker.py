import glob
import os
import pickle
import random
import re
import sys
import time


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

import numpy as np
import torch
from tqdm import tqdm

import config
from atari_wrappers import make_atari_env
from ddqn_agent import DDQNAgent
from replay_buffer import ReplayBuffer


def _mode_prefix(is_fixed):
    return f"fixed_C{config.FIXED_C}" if is_fixed else "adaptive"


def _step_ckpt_name(env_id, seed, is_fixed, step):
    return f"{_mode_prefix(is_fixed)}_{env_id}_seed{seed}_step{int(step)}.pth"


def _final_ckpt_name(env_id, seed, is_fixed):
    return f"{_mode_prefix(is_fixed)}_{env_id}_seed{seed}_final.pth"


def _extract_step_from_ckpt_name(path):
    m = re.search(r"_step(\d+)\.pth$", os.path.basename(path))
    return int(m.group(1)) if m else 0


def _candidate_prefixes(is_fixed):
    if not is_fixed:
        return ["adaptive"]
    # 兼容历史固定模式命名：fixed_C{C} / fixed_C* / fixed
    return [f"fixed_C{config.FIXED_C}", "fixed_C*", "fixed"]


def _find_latest_checkpoint(paths, env_id, seed, is_fixed, total_steps=None):
    if total_steps is None:
        total_steps = config.TOTAL_STEPS
    ckpt_dir = paths["pth_dir"]

    final_candidates = [
        os.path.join(ckpt_dir, f"{prefix}_{env_id}_seed{seed}_final.pth")
        for prefix in _candidate_prefixes(is_fixed)
    ]

    for p in final_candidates:
        if os.path.exists(p):
            return p, int(total_steps)

    best_step = -1
    best_path = None
    for prefix in _candidate_prefixes(is_fixed):
        step_pattern = os.path.join(ckpt_dir, f"{prefix}_{env_id}_seed{seed}_step*.pth")
        for p in glob.glob(step_pattern):
            step = _extract_step_from_ckpt_name(p)
            if step > best_step:
                best_step = step
                best_path = p

    if best_path is None:
        return None, 0

    return best_path, max(0, best_step)


def _scheduled_epsilon(step):
    return max(
        config.EPSILON_END,
        config.EPSILON_START
        - (config.EPSILON_START - config.EPSILON_END) * step / config.EPSILON_DECAY_STEPS,
    )


def _save_training_checkpoint(
    agent,
    save_path,
    current_step,
    epsilon,
    env_id,
    seed,
    is_fixed,
    elapsed_time_sec=0.0,
    total_steps=None,
):
    if total_steps is None:
        total_steps = config.TOTAL_STEPS
    payload = agent.get_checkpoint()
    payload.update(
        {
            "env_id": env_id,
            "seed": int(seed),
            "is_fixed": bool(is_fixed),
            "current_step": int(current_step),
            "epsilon": float(epsilon),
            "elapsed_time_sec": float(elapsed_time_sec),
            "total_steps": int(total_steps),
        }
    )
    torch.save(payload, save_path)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        deterministic = config.PERFORMANCE_OPTS.get("deterministic", False)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = config.PERFORMANCE_OPTS["cudnn_benchmark"] and (not deterministic)


def evaluate_agent(agent, env, eval_episodes=None, epsilon=None, max_frames=None):
    """评估函数（论文口径：100局，18,000帧上限，epsilon=0.05）"""
    if eval_episodes is None:
        eval_episodes = config.EVAL_EPISODES
    if epsilon is None:
        epsilon = config.EVAL_EPSILON
    if max_frames is None:
        max_frames = config.EVAL_MAX_FRAMES

    total_rewards = []
    for _ in range(eval_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        frame_count = 0
        while (not done) and (frame_count < max_frames):
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
            frame_count += 1
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)


def run_single_training(task_queue, result_queue, gpu_id):
    """Worker主训练函数"""
    # 设备设置：因为通过 CUDA_VISIBLE_DEVICES 绑定了，这里直接用 cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 性能优化设置
    torch.set_num_threads(max(1, int(config.PERFORMANCE_OPTS["torch_num_threads"])))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 训练更新逻辑统一放在 agent.update 内部，避免外层重复 backward/step
    device_name = f"cuda:{gpu_id}" if device.type == "cuda" else "cpu"
    print(f"[Worker GPU:{gpu_id}] 启动就绪 | device={device_name}")

    while True:
        try:
            task = task_queue.get(timeout=30)
        except Exception:
            print(f"[Worker GPU:{gpu_id}] 队列等待超时，尝试安全退出...")
            result_queue.put(None)
            break

        if task is None:
            print(f"[Worker GPU:{gpu_id}] 收到结束信号")
            result_queue.put(None)
            break

        # 解析任务（兼容旧格式: 4元组；新格式: 6元组）
        if len(task) == 7:
            (env_id, seed, is_fixed, paths, adaptive_c_min, adaptive_epsilon_tol, total_steps) = task
        elif len(task) == 6:
            (env_id, seed, is_fixed, paths, adaptive_c_min, adaptive_epsilon_tol) = task
            total_steps = config.TOTAL_STEPS
        elif len(task) == 4:
            (env_id, seed, is_fixed, paths) = task
            adaptive_c_min = config.ADAPTIVE_PARAMS.get("adaptive_C_min")
            adaptive_epsilon_tol = config.ADAPTIVE_PARAMS["adaptive_epsilon_tol"]
            total_steps = config.TOTAL_STEPS
        else:
            result_queue.put(
                {
                    "status": "error",
                    "env_id": "unknown",
                    "seed": -1,
                    "is_fixed": True,
                    "error": f"bad_task_format: len={len(task)}",
                }
            )
            continue
        total_steps = int(total_steps)
        if total_steps <= 0:
            result_queue.put(
                {
                    "status": "error",
                    "env_id": env_id,
                    "seed": seed,
                    "is_fixed": is_fixed,
                    "error": f"bad_total_steps: {total_steps}",
                }
            )
            continue
        task_wall_start = time.perf_counter()
        previous_elapsed_time_sec = 0.0
        print(f"\n[Worker GPU:{gpu_id}] 开始任务: Env={env_id}, Seed={seed}, Mode={'Fixed' if is_fixed else 'Adaptive'}")

        set_seed(seed)

        # 创建环境（单环境）
        try:
            train_env = make_atari_env(
                env_id,
                seed,
                episode_life=True,
                clip_rewards=True,
            )
            eval_env = make_atari_env(
                env_id,
                seed + 100,
                episode_life=False,
                clip_rewards=False,
            )
        except Exception as e:
            print(f"[Worker GPU:{gpu_id}] 创建环境失败: {e}", file=sys.stderr)
            result_queue.put(
                {
                    "status": "error",
                    "env_id": env_id,
                    "seed": seed,
                    "is_fixed": is_fixed,
                    "elapsed_time_sec": time.perf_counter() - task_wall_start,
                    "error": f"env_init_failed: {e}",
                }
            )
            continue

        input_shape = train_env.observation_space.shape
        num_actions = train_env.action_space.n

        # 初始化回放池
        replay_buffer = ReplayBuffer(capacity=config.REPLAY_CAPACITY)

        C_min = 500
        if not is_fixed:
            if adaptive_c_min is None:
                gamma = 0.99
                rho_est = 0.99
                C_min = np.ceil(np.log((1 - gamma) / (1 + gamma)) / np.log(rho_est)).astype(int)
            else:
                C_min = adaptive_c_min

        # 初始化Agent
        agent = DDQNAgent(
            input_shape=input_shape,
            num_actions=num_actions,
            device=device,
            gamma=config.GAMMA,
            fixed_update_interval=config.FIXED_C,
            use_adaptive_update=not is_fixed,
            C_min=C_min,
            epsilon_tol=adaptive_epsilon_tol,
        )

        # 训练状态初始化
        state, _ = train_env.reset()
        current_step = 0
        epsilon = config.EPSILON_START

        # 记录数据
        step_records = []
        return_records = []
        latest_return = -np.inf
        latest_loss = 0.0

        # 自动断点续训
        resume_path, resume_step = _find_latest_checkpoint(paths, env_id, seed, is_fixed, total_steps=total_steps)
        if resume_path is not None:
            try:
                ckpt_obj = torch.load(resume_path, map_location=device, weights_only=False)
                restored_full = agent.load_checkpoint(ckpt_obj)

                if isinstance(ckpt_obj, dict):
                    resume_step = int(ckpt_obj.get("current_step", resume_step))
                    epsilon = float(ckpt_obj.get("epsilon", _scheduled_epsilon(resume_step)))
                    previous_elapsed_time_sec = float(ckpt_obj.get("elapsed_time_sec", 0.0))
                else:
                    epsilon = _scheduled_epsilon(resume_step)

                current_step = max(0, min(resume_step, total_steps))
                print(
                    f"[Worker GPU:{gpu_id}] 已加载checkpoint: {os.path.basename(resume_path)} | "
                    f"resume_step={current_step} | full_state={'yes' if restored_full else 'no(legacy)'}"
                )
            except Exception as e:
                print(f"[Worker GPU:{gpu_id}] 加载checkpoint失败，回退冷启动: {e}")
                current_step = 0
                epsilon = config.EPSILON_START
                previous_elapsed_time_sec = 0.0

        if current_step >= total_steps:
            elapsed_time_sec = previous_elapsed_time_sec + (time.perf_counter() - task_wall_start)
            print(
                f"[Worker GPU:{gpu_id}] 已达到总步数({total_steps})，跳过训练: "
                f"{env_id} Seed{seed} {'Fixed' if is_fixed else 'Adaptive'}"
            )
            pkl_name = f"{'fixed' if is_fixed else 'adaptive'}_{env_id}_seed{seed}.pkl"
            pkl_path = os.path.join(paths["env_dir"], pkl_name)
            should_update_pkl = False
            if os.path.exists(pkl_path):
                with open(pkl_path, "rb") as f:
                    result_data = pickle.load(f)
                if "elapsed_time_sec" not in result_data:
                    result_data["elapsed_time_sec"] = float(elapsed_time_sec)
                    result_data["elapsed_time_hours"] = float(elapsed_time_sec / 3600.0)
                    should_update_pkl = True
                if "elapsed_time_hours" not in result_data:
                    sec = float(result_data.get("elapsed_time_sec", elapsed_time_sec))
                    result_data["elapsed_time_hours"] = float(sec / 3600.0)
                    should_update_pkl = True
                if "adaptive_C_min" not in result_data:
                    result_data["adaptive_C_min"] = None if adaptive_c_min is None else int(adaptive_c_min)
                    should_update_pkl = True
                if "adaptive_epsilon_tol" not in result_data:
                    result_data["adaptive_epsilon_tol"] = float(adaptive_epsilon_tol)
                    should_update_pkl = True
                if "total_steps" not in result_data:
                    result_data["total_steps"] = int(total_steps)
                    should_update_pkl = True
            else:
                result_data = {
                    "env_id": env_id,
                    "seed": seed,
                    "is_fixed": is_fixed,
                    "total_steps": int(total_steps),
                    "elapsed_time_sec": float(elapsed_time_sec),
                    "elapsed_time_hours": float(elapsed_time_sec / 3600.0),
                    "adaptive_C_min": None if adaptive_c_min is None else int(adaptive_c_min),
                    "adaptive_epsilon_tol": float(adaptive_epsilon_tol),
                    "step_records": np.array(step_records),
                    "returns": np.array(return_records),
                    "C_steps": np.array(agent.adaptive_C_step_history) if not is_fixed else np.array([]),
                    "C_values": np.array(agent.adaptive_C_history) if not is_fixed else np.array([]),
                    "loss_steps": np.array(agent.loss_step_history),
                    "loss_values": np.array(agent.loss_history),
                }
                should_update_pkl = True
            if should_update_pkl:
                with open(pkl_path, "wb") as f:
                    pickle.dump(result_data, f)
            result_queue.put(result_data)
            try:
                train_env.close()
                eval_env.close()
            except Exception:
                pass
            continue

        # 进度条
        pbar = tqdm(
            total=total_steps,
            desc=f"GPU{gpu_id} {env_id.split('No')[0]} S{seed} {'Fix' if is_fixed else 'Adp'}",
            mininterval=2.0,
            leave=False,
        )
        if current_step > 0:
            pbar.update(current_step)

        pbar_update_every = int(config.PERFORMANCE_OPTS.get("pbar_update_every", 100))
        pbar_pending = 0

        try:
            while current_step < total_steps:
                # 更新进度条显示
                if current_step % 500 == 0:
                    pbar.set_postfix(
                        {
                            "Eps": f"{epsilon:.2f}",
                            "R": f"{latest_return:.1f}",
                            "Loss": f"{latest_loss:.3f}",
                        },
                        refresh=False,
                    )

                # 更新探索率
                epsilon = _scheduled_epsilon(current_step)

                # 环境交互
                action = agent.select_action(state, epsilon)
                next_state, reward, terminated, truncated, _ = train_env.step(action)
                done = terminated or truncated
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state

                if done:
                    state, _ = train_env.reset()

                # 网络更新
                if (
                    current_step >= config.WARMUP_STEPS
                    and len(replay_buffer) >= config.BATCH_SIZE
                    and current_step % config.TRAIN_FREQ == 0
                ):
                    batch = replay_buffer.sample(config.BATCH_SIZE, device)

                    # 由 agent.update 统一完成 forward/backward/optimizer.step
                    loss = agent.update(batch, current_step)
                    latest_loss = loss

                # 定期评估
                if current_step % config.EVAL_INTERVAL == 0 and current_step >= config.WARMUP_STEPS:
                    latest_return = evaluate_agent(
                        agent,
                        eval_env,
                        eval_episodes=config.EVAL_EPISODES,
                        epsilon=config.EVAL_EPSILON,
                        max_frames=config.EVAL_MAX_FRAMES,
                    )
                    step_records.append(current_step)
                    return_records.append(latest_return)
                    pbar.write(f"[GPU{gpu_id}] Step {current_step} | 平均回报: {latest_return:.2f}")

                # 定期保存模型（现在fixed/adaptive都保存，并保存完整训练状态）
                if current_step % config.SAVE_INTERVAL == 0 and current_step > 0:
                    model_name = _step_ckpt_name(env_id, seed, is_fixed, current_step)
                    _save_training_checkpoint(
                        agent,
                        os.path.join(paths["pth_dir"], model_name),
                        current_step=current_step,
                        epsilon=epsilon,
                        env_id=env_id,
                        seed=seed,
                        is_fixed=is_fixed,
                        elapsed_time_sec=previous_elapsed_time_sec + (time.perf_counter() - task_wall_start),
                        total_steps=total_steps,
                    )

                current_step += 1
                pbar_pending += 1
                if pbar_pending >= pbar_update_every:
                    pbar.update(pbar_pending)
                    pbar_pending = 0

            if pbar_pending > 0:
                pbar.update(pbar_pending)

            elapsed_time_sec = previous_elapsed_time_sec + (time.perf_counter() - task_wall_start)

            # 保存最终模型
            model_name = _final_ckpt_name(env_id, seed, is_fixed)
            _save_training_checkpoint(
                agent,
                os.path.join(paths["pth_dir"], model_name),
                current_step=total_steps,
                epsilon=epsilon,
                env_id=env_id,
                seed=seed,
                is_fixed=is_fixed,
                elapsed_time_sec=elapsed_time_sec,
                total_steps=total_steps,
            )

            # 保存结果数据
            result_data = {
                "env_id": env_id,
                "seed": seed,
                "is_fixed": is_fixed,
                "total_steps": int(total_steps),
                "elapsed_time_sec": float(elapsed_time_sec),
                "elapsed_time_hours": float(elapsed_time_sec / 3600.0),
                "adaptive_C_min": None if adaptive_c_min is None else int(adaptive_c_min),
                "adaptive_epsilon_tol": float(adaptive_epsilon_tol),
                "step_records": np.array(step_records),
                "returns": np.array(return_records),
                "C_steps": np.array(agent.adaptive_C_step_history) if not is_fixed else np.array([]),
                "C_values": np.array(agent.adaptive_C_history) if not is_fixed else np.array([]),
                "loss_steps": np.array(agent.loss_step_history),
                "loss_values": np.array(agent.loss_history),
            }

            pkl_name = f"{'fixed' if is_fixed else 'adaptive'}_{env_id}_seed{seed}.pkl"
            with open(os.path.join(paths["env_dir"], pkl_name), "wb") as f:
                pickle.dump(result_data, f)

            result_queue.put(result_data)
            print(f"[Worker GPU:{gpu_id}] 任务完成: {env_id} Seed{seed}")

        except Exception as e:
            print(f"[Worker GPU:{gpu_id}] 任务出错: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            result_queue.put(
                {
                    "status": "error",
                    "env_id": env_id,
                    "seed": seed,
                    "is_fixed": is_fixed,
                    "elapsed_time_sec": previous_elapsed_time_sec + (time.perf_counter() - task_wall_start),
                    "error": str(e),
                }
            )
        finally:
            pbar.close()
            try:
                train_env.close()
                eval_env.close()
            except Exception:
                pass

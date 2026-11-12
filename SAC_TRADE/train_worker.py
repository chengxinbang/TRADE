import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from envs import make_env
from replay_buffer import ReplayBuffer
from sac_agent import SACAgent
from utils import set_seed


def evaluate(agent, env, episodes, horizon, suite):
    returns, successes = [], []
    for _ in range(episodes):
        observation, _ = env.reset()
        total, success = 0.0, 0.0
        for _ in range(horizon):
            observation, reward, terminated, truncated, info = env.step(agent.act(observation))
            total += float(reward)
            if suite == "metaworld":
                success = max(success, float(info["success"]))
            if terminated or truncated:
                break
        returns.append(total)
        if suite == "metaworld":
            successes.append(success)
    return float(np.mean(returns)), (float(np.mean(successes)) if successes else np.nan)


def train(cfg, device="cpu", output="results"):
    set_seed(cfg.seed)
    torch.set_num_threads(1)
    path = Path(output) / cfg.suite / cfg.env / cfg.mode / f"seed_{cfg.seed}"
    path.mkdir(parents=True, exist_ok=True)
    if (path / "curves.npz").exists():
        raise FileExistsError(f"Completed run exists: {path}; choose another output directory")
    settings = asdict(cfg)
    settings.update(protocol="loss-ema-env-step-v1", device=device,
                    torch_version=torch.__version__, numpy_version=np.__version__)
    (path / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    env = make_env(cfg.suite, cfg.env, cfg.seed)
    evaluation_env = make_env(cfg.suite, cfg.env, cfg.seed + 100, evaluation=True)
    try:
        agent = SACAgent(env.observation_space.shape[0], env.action_space.shape[0], device,
                         target_update_mode=cfg.mode, c_min=cfg.c_min,
                         discount=cfg.discount,
                         init_temperature=cfg.init_temperature,
                         actor_lr=cfg.learning_rate, critic_lr=cfg.learning_rate,
                         alpha_lr=cfg.learning_rate, critic_tau=cfg.soft_update_coefficient,
                         batch_size=cfg.batch_size)
        replay = ReplayBuffer(env.observation_space.shape, env.action_space.shape,
                              cfg.replay_capacity, agent.device)
        observation, _ = env.reset()
        episode_step = 0
        steps, returns, success_rates = [], [], []
        loss_steps, losses = [], []
        start = time.perf_counter()

        def save_curves():
            events = agent.scheduler.history
            np.savez_compressed(path / "curves.npz", steps=steps, returns=returns,
                                success_rates=success_rates, loss_steps=loss_steps,
                                losses=losses, refresh_steps=[e["step"] for e in events],
                                refresh_intervals=[e["interval"] for e in events],
                                refresh_scores=[e["score"] for e in events])

        for step in trange(1, cfg.total_steps + 1, desc=f"{cfg.env} {cfg.mode} S{cfg.seed}"):
            action = (env.action_space.sample() if step <= cfg.warmup_steps
                      else agent.act(observation, sample=True))
            next_observation, reward, terminated, truncated, _ = env.step(action)
            episode_step += 1
            truncated = bool(truncated or episode_step >= cfg.horizon)
            replay.add(observation, action, reward, next_observation,
                       bool(terminated or truncated), bool(terminated))
            observation = next_observation
            if terminated or truncated:
                observation, _ = env.reset()
                episode_step = 0
            if step > cfg.warmup_steps and len(replay) >= cfg.batch_size:
                loss_steps.append(step)
                losses.append(agent.update(replay, step))
            if step % cfg.eval_interval == 0 or step == cfg.total_steps:
                score, success = evaluate(agent, evaluation_env, cfg.eval_episodes,
                                          cfg.horizon, cfg.suite)
                steps.append(step)
                returns.append(score)
                success_rates.append(success)
                save_curves()
        torch.save(dict(agent=agent.checkpoint(), step=cfg.total_steps, config=settings),
                   path / "final.pt")
        (path / "summary.json").write_text(json.dumps(dict(
            completed_steps=cfg.total_steps, elapsed_seconds=time.perf_counter()-start,
            target_refresh_count=agent.target_refresh_count,
            final_return=returns[-1],
            final_success_rate=success_rates[-1] if cfg.suite == "metaworld" else None),
            indent=2), encoding="utf-8")
    finally:
        env.close()
        evaluation_env.close()
    return path

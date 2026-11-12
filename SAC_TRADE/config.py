"""Paper defaults. All budgets and target intervals count agent/environment steps."""
from dataclasses import dataclass

SEEDS = [0, 1, 2, 3, 4]
MODES = ["soft", "hard_adaptive"]
DMC_TASKS = ["cheetah_run", "hopper_hop", "walker_walk", "humanoid_run",
             "ball_in_cup_catch", "cartpole_balance", "cartpole_balance_sparse",
             "cartpole_swingup", "finger_spin", "finger_turn_easy", "finger_turn_hard",
             "fish_swim", "fish_upright", "hopper_stand", "humanoid_stand",
             "humanoid_walk", "pendulum_swingup", "quadruped_run", "quadruped_walk",
             "reacher_easy", "reacher_hard", "swimmer_swimmer6", "swimmer_swimmer15",
             "walker_run", "walker_stand", "acrobot_swingup", "cartpole_swingup_sparse"]
METAWORLD_TASKS = ["button-press-topdown-v2", "dial-turn-v2", "window-open-v2",
                   "door-open-v2", "drawer-open-v2", "reach-v2", "push-v2",
                   "pick-place-v2", "peg-insert-side-v2", "handle-pull-v2"]


@dataclass
class RunConfig:
    suite: str
    env: str
    mode: str
    seed: int
    total_steps: int = 1_000_000
    c_min: int = 100
    warmup_steps: int = 1000
    horizon: int = 1000
    eval_interval: int = 20_000
    eval_episodes: int = 10
    init_temperature: float = 0.1
    replay_capacity: int = 1_000_000
    batch_size: int = 256
    discount: float = 0.99
    learning_rate: float = 3e-4
    soft_update_coefficient: float = 0.005
    ema_coefficient: float = 0.99
    numerical_loss_floor: float = 1e-8


def configuration(suite, env, mode, seed, total_steps=None, c_min=100,
                  eval_episodes=10):
    if suite not in {"dmc", "metaworld", "mujoco"} or mode not in MODES:
        raise ValueError("Invalid suite or update mode")
    if c_min < 1 or eval_episodes < 1 or (total_steps is not None and total_steps < 1):
        raise ValueError("Budgets, evaluation episodes and C_min must be positive")
    if suite == "dmc" and env not in DMC_TASKS:
        raise ValueError(f"Unknown DMC task: {env}")
    if suite == "metaworld" and not env.endswith("-v2"):
        raise ValueError("The paper protocol requires an explicit Meta-World v2 task")
    cfg = RunConfig(suite, env, mode, seed, c_min=c_min, eval_episodes=eval_episodes)
    if suite == "metaworld":
        cfg.warmup_steps = 10_000
        cfg.horizon = 500
        cfg.eval_interval = 10_000
        cfg.init_temperature = 1.0
    if total_steps is not None:
        cfg.total_steps = total_steps
    return cfg

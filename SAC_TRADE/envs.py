"""Explicit task adapters; v2 tasks are never silently replaced with v3."""
import random

import gymnasium as gym
import numpy as np


class DMCEnv:
    def __init__(self, task, seed):
        from dm_control import suite
        if task.startswith("ball_in_cup_"):
            domain, task = "ball_in_cup", task[len("ball_in_cup_"):]
        else:
            domain, task = task.split("_", 1)
        self.env = suite.load(domain, task, task_kwargs={"random": seed})
        self.keys = list(self.env.observation_spec())
        spec = self.env.action_spec()
        self.low, self.high = spec.minimum, spec.maximum
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=spec.shape, dtype=np.float32)
        n = sum(int(np.prod(v.shape)) for v in self.env.observation_spec().values())
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (n,), np.float32)
        self.action_space.seed(seed)

    def _observation(self, timestep):
        return np.concatenate([np.asarray(timestep.observation[k], dtype=np.float32).reshape(-1)
                               for k in self.keys])

    def reset(self):
        return self._observation(self.env.reset()), {}

    def step(self, action):
        action = self.low + (np.clip(action, -1, 1) + 1) * 0.5 * (self.high - self.low)
        t = self.env.step(action)
        # DMC distinguishes genuine termination (discount=0) from its time limit.
        terminated = bool(t.last() and t.discount == 0)
        truncated = bool(t.last() and not terminated)
        return self._observation(t), float(t.reward or 0), terminated, truncated, {}

    def close(self):
        self.env.close()


class MetaWorldEnv:
    def __init__(self, task, seed, evaluation=False):
        import metaworld
        try:
            benchmark = metaworld.ML1(task, seed=seed)
        except (KeyError, ValueError) as error:
            raise ValueError("Install the pinned Meta-World v2 revision; no v3 fallback is used") from error
        self.env = benchmark.train_classes[task]()
        self.tasks = benchmark.test_tasks if evaluation else benchmark.train_tasks
        self.random = random.Random(seed)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        if tuple(self.observation_space.shape) != (39,) or tuple(self.action_space.shape) != (4,):
            raise ValueError("Expected Meta-World v2 observations (39,) and actions (4,)")
        self.action_space.seed(seed)
        self.env.seed(seed) if hasattr(self.env, "seed") else None
        self.steps = 0

    def reset(self):
        self.env.set_task(self.random.choice(self.tasks))
        self.steps = 0
        result = self.env.reset()
        return result if isinstance(result, tuple) else (result, {})

    def step(self, action):
        result = self.env.step(np.clip(action, -1, 1))
        self.steps += 1
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            obs, reward, done, info = result
            terminated, truncated = bool(done), False
        # Meta-World has a finite horizon; success is a metric, not early terminal.
        if self.steps >= 500:
            terminated, truncated = False, True
        if "success" not in info:
            raise ValueError("Meta-World must expose info['success']; no reward-based proxy")
        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        self.env.close()


def make_env(suite, task, seed, evaluation=False):
    if suite == "dmc":
        return DMCEnv(task, seed)
    if suite == "metaworld":
        return MetaWorldEnv(task, seed, evaluation)
    env = gym.make(task)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    # The provided MuJoCo tasks expose [-1,1] actions like the retained actor.
    if not (np.allclose(env.action_space.low, -1) and np.allclose(env.action_space.high, 1)):
        raise ValueError("Use a task with normalized continuous actions in [-1,1]")
    return env

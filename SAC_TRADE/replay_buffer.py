import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_shape, action_shape, capacity, device):
        self.capacity = int(capacity)
        self.device = device

        self.obses = np.empty((self.capacity, *obs_shape), dtype=np.float32)
        self.next_obses = np.empty((self.capacity, *obs_shape), dtype=np.float32)
        self.actions = np.empty((self.capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((self.capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((self.capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], 1.0 - float(done))
        np.copyto(self.not_dones_no_max[self.idx], 1.0 - float(done_no_max))

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        upper = self.capacity if self.full else self.idx
        idxs = np.random.randint(0, upper, size=int(batch_size))

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max

    def state_dict(self):
        return {
            "capacity": int(self.capacity),
            "idx": int(self.idx),
            "full": bool(self.full),
            "obses": np.array(self.obses, copy=True),
            "next_obses": np.array(self.next_obses, copy=True),
            "actions": np.array(self.actions, copy=True),
            "rewards": np.array(self.rewards, copy=True),
            "not_dones": np.array(self.not_dones, copy=True),
            "not_dones_no_max": np.array(self.not_dones_no_max, copy=True),
        }

    def load_state_dict(self, state):
        required = [
            "capacity",
            "idx",
            "full",
            "obses",
            "next_obses",
            "actions",
            "rewards",
            "not_dones",
            "not_dones_no_max",
        ]
        missing = [k for k in required if k not in state]
        if missing:
            raise KeyError(f"ReplayBuffer state missing keys: {missing}")

        state_capacity = int(state["capacity"])
        if state_capacity != self.capacity:
            raise ValueError(
                f"ReplayBuffer capacity mismatch: state={state_capacity}, current={self.capacity}"
            )

        shape_pairs = [
            ("obses", self.obses.shape),
            ("next_obses", self.next_obses.shape),
            ("actions", self.actions.shape),
            ("rewards", self.rewards.shape),
            ("not_dones", self.not_dones.shape),
            ("not_dones_no_max", self.not_dones_no_max.shape),
        ]
        for key, expected_shape in shape_pairs:
            got_shape = np.asarray(state[key]).shape
            if got_shape != expected_shape:
                raise ValueError(
                    f"ReplayBuffer shape mismatch for {key}: state={got_shape}, expected={expected_shape}"
                )

        np.copyto(self.obses, np.asarray(state["obses"], dtype=np.float32))
        np.copyto(self.next_obses, np.asarray(state["next_obses"], dtype=np.float32))
        np.copyto(self.actions, np.asarray(state["actions"], dtype=np.float32))
        np.copyto(self.rewards, np.asarray(state["rewards"], dtype=np.float32))
        np.copyto(self.not_dones, np.asarray(state["not_dones"], dtype=np.float32))
        np.copyto(
            self.not_dones_no_max,
            np.asarray(state["not_dones_no_max"], dtype=np.float32),
        )

        self.idx = int(state["idx"])
        self.full = bool(state["full"])

        if not (0 <= self.idx < self.capacity):
            raise ValueError(f"ReplayBuffer idx out of range: {self.idx}")

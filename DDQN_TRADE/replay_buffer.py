import numpy as np
import torch


class ReplayBuffer:
    """A memory-efficient ring buffer with O(1) random indexing."""

    def __init__(self, capacity=100000):
        self.capacity = int(capacity)
        self.size = 0
        self.pos = 0

        self._initialized = False
        self.states = None
        self.next_states = None
        self.actions = None
        self.rewards = None
        self.dones = None

    def _lazy_init(self, state):
        state = np.asarray(state)
        self.states = np.empty((self.capacity, *state.shape), dtype=np.uint8)
        self.next_states = np.empty((self.capacity, *state.shape), dtype=np.uint8)
        self.actions = np.empty((self.capacity,), dtype=np.int64)
        self.rewards = np.empty((self.capacity,), dtype=np.float32)
        self.dones = np.empty((self.capacity,), dtype=np.float32)
        self._initialized = True

    def add(self, state, action, reward, next_state, done):
        if not self._initialized:
            self._lazy_init(state)

        self.states[self.pos] = np.asarray(state, dtype=np.uint8)
        self.next_states[self.pos] = np.asarray(next_state, dtype=np.uint8)
        self.actions[self.pos] = int(action)
        self.rewards[self.pos] = float(reward)
        self.dones[self.pos] = float(done)

        self.pos = (self.pos + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)

        states = torch.as_tensor(self.states[idx], device=device)
        actions = torch.as_tensor(self.actions[idx], device=device)
        rewards = torch.as_tensor(self.rewards[idx], device=device)
        next_states = torch.as_tensor(self.next_states[idx], device=device)
        dones = torch.as_tensor(self.dones[idx], device=device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size

import random

import torch
import torch.nn as nn
import torch.optim as optim

from networks import DQNNet
from trade_scheduler import TRADEScheduler


class DDQNAgent:
    def __init__(self, input_shape, num_actions, device, lr=2.5e-4,
                 gamma=0.99, fixed_update_interval=30000,
                 use_adaptive_update=False, C_min=500):
        self.gamma = gamma
        self.num_actions = num_actions
        self.device = device
        self.use_adaptive_update = use_adaptive_update
        self.online_net = DQNNet(input_shape, num_actions).to(device)
        self.target_net = DQNNet(input_shape, num_actions).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.online_net.parameters(), lr=lr,
                                       alpha=0.95, eps=1e-2)
        self.loss_fn = nn.MSELoss()
        self.fixed_update_interval = int(fixed_update_interval)
        if self.fixed_update_interval < 1:
            raise ValueError("fixed_update_interval must be positive")
        self.scheduler = TRADEScheduler(C_min)
        self.adaptive_C_history = []
        self.adaptive_C_step_history = []
        self.loss_history = []
        self.loss_step_history = []
        self.target_refresh_count = 0

    def get_checkpoint(self):
        return dict(protocol=TRADEScheduler.protocol,
                    settings=dict(gamma=self.gamma, fixed_interval=self.fixed_update_interval),
                    online_net=self.online_net.state_dict(),
                    target_net=self.target_net.state_dict(),
                    optimizer=self.optimizer.state_dict(),
                    use_adaptive_update=self.use_adaptive_update,
                    scheduler=self.scheduler.state_dict(),
                    target_refresh_count=self.target_refresh_count,
                    adaptive_C_history=self.adaptive_C_history,
                    adaptive_C_step_history=self.adaptive_C_step_history,
                    loss_history=self.loss_history,
                    loss_step_history=self.loss_step_history)

    def load_checkpoint(self, payload):
        if payload.get("protocol") != TRADEScheduler.protocol:
            raise ValueError("Legacy checkpoints do not use the paper's loss-EMA protocol")
        if payload["use_adaptive_update"] != self.use_adaptive_update:
            raise ValueError("Checkpoint target-update mode differs")
        if payload["settings"] != dict(gamma=self.gamma, fixed_interval=self.fixed_update_interval):
            raise ValueError("Checkpoint learning configuration differs")
        self.scheduler.load_state_dict(payload["scheduler"])
        self.online_net.load_state_dict(payload["online_net"])
        self.target_net.load_state_dict(payload["target_net"])
        self.optimizer.load_state_dict(payload["optimizer"])
        for key in ["target_refresh_count", "adaptive_C_history",
                    "adaptive_C_step_history", "loss_history", "loss_step_history"]:
            setattr(self, key, payload[key])
        return True

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.num_actions)
        with torch.no_grad():
            state = torch.as_tensor(state, device=self.device).unsqueeze(0)
            return self.online_net(state).argmax().item()

    def update(self, batch, current_step):
        states, actions, rewards, next_states, dones = batch
        self.online_net.train()
        with torch.no_grad():
            best_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, best_actions).squeeze(1)
            target = rewards + self.gamma * next_q * (1 - dones)
        prediction = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = self.loss_fn(prediction, target)
        loss_value = float(loss.detach().item())  # pre-optimizer fitting loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.loss_history.append(loss_value)
        self.loss_step_history.append(int(current_step))
        if self.use_adaptive_update:
            refresh = self.scheduler.update(loss_value, current_step)
        else:
            refresh = current_step % self.fixed_update_interval == 0
        if refresh:
            self.target_net.load_state_dict(self.online_net.state_dict())
            self.target_refresh_count += 1
            if self.use_adaptive_update:
                self.adaptive_C_history.append(self.scheduler.history[-1]["interval"])
                self.adaptive_C_step_history.append(int(current_step))
        return loss_value

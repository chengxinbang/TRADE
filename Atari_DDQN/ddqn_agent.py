import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from networks import DQNNet


class DDQNAgent:
    def __init__(
        self,
        input_shape,
        num_actions,
        device,
        lr=2.5e-4,
        gamma=0.99,
        fixed_update_interval=30000,
        use_adaptive_update=False,
        C_min=500,
        epsilon_tol=1e-3,
    ):
        self.gamma = gamma
        self.num_actions = num_actions
        self.device = device
        self.use_adaptive_update = use_adaptive_update

        # 网络初始化
        self.online_net = DQNNet(input_shape, num_actions).to(device)
        self.target_net = DQNNet(input_shape, num_actions).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.online_net.parameters(), lr=lr, alpha=0.95, eps=1e-2)
        self.loss_fn = nn.SmoothL1Loss()

        # 固定更新模式参数
        self.fixed_update_interval = fixed_update_interval

        # 自适应更新模式状态变量
        self.C_min = C_min
        self.epsilon_tol = epsilon_tol
        self.current_k = 0
        self.L0 = None
        self.rho_min = float("inf")
        self.adaptive_C_history = []
        self.adaptive_C_step_history = []
        self.loss_history = []
        self.loss_step_history = []

    def get_checkpoint(self):
        """返回可用于断点续训的完整状态。"""
        return {
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "use_adaptive_update": bool(self.use_adaptive_update),
            "adaptive_state": {
                "current_k": int(self.current_k),
                "L0": None if self.L0 is None else float(self.L0),
                "rho_min": float(self.rho_min),
                "adaptive_C_history": list(self.adaptive_C_history),
                "adaptive_C_step_history": list(self.adaptive_C_step_history),
            },
            "loss_state": {
                "loss_history": list(self.loss_history),
                "loss_step_history": list(self.loss_step_history),
            },
        }

    def load_checkpoint(self, checkpoint_obj):
        """
        兼容两种格式：
        1) 新格式：包含 online_net/target_net/optimizer 的完整状态字典。
        2) 旧格式：直接保存的 online_net.state_dict()。

        返回值：
            True  -> 恢复了完整训练状态（含优化器/目标网/自适应状态）。
            False -> 仅恢复了 online_net，target_net 已与其同步。
        """
        if not isinstance(checkpoint_obj, dict):
            raise TypeError("checkpoint_obj must be a dict")

        # 旧格式：纯网络权重
        if "online_net" not in checkpoint_obj:
            self.online_net.load_state_dict(checkpoint_obj)
            self.target_net.load_state_dict(self.online_net.state_dict())
            return False

        # 新格式：完整恢复
        self.online_net.load_state_dict(checkpoint_obj["online_net"])
        if "target_net" in checkpoint_obj and checkpoint_obj["target_net"] is not None:
            self.target_net.load_state_dict(checkpoint_obj["target_net"])
        else:
            self.target_net.load_state_dict(self.online_net.state_dict())

        if "optimizer" in checkpoint_obj and checkpoint_obj["optimizer"] is not None:
            self.optimizer.load_state_dict(checkpoint_obj["optimizer"])

        adaptive_state = checkpoint_obj.get("adaptive_state", {})
        self.current_k = int(adaptive_state.get("current_k", 0))
        self.L0 = adaptive_state.get("L0", None)
        self.rho_min = float(adaptive_state.get("rho_min", float("inf")))
        self.adaptive_C_history = list(adaptive_state.get("adaptive_C_history", []))
        self.adaptive_C_step_history = list(adaptive_state.get("adaptive_C_step_history", []))

        loss_state = checkpoint_obj.get("loss_state", {})
        self.loss_history = list(loss_state.get("loss_history", []))
        self.loss_step_history = list(loss_state.get("loss_step_history", []))
        return True

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)

        with torch.no_grad():
            state = torch.as_tensor(state, device=self.device).unsqueeze(0)
            q_values = self.online_net(state)
            return q_values.argmax().item()

    def update(self, batch, current_step):
        states, actions, rewards, next_states, dones = batch
        self.online_net.train()

        # DDQN核心逻辑
        with torch.no_grad():
            next_q_online = self.online_net(next_states)
            best_next_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = self.target_net(next_states)
            target_q = next_q_target.gather(1, best_next_actions).squeeze()
            td_target = rewards + self.gamma * target_q * (1 - dones)

        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        loss = self.loss_fn(current_q, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.loss_history.append(loss.item())
        self.loss_step_history.append(current_step)

        # 目标网络更新逻辑
        if not self.use_adaptive_update:
            if current_step % self.fixed_update_interval == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())
        else:
            if self.current_k == 0:
                self.L0 = max(loss.item(), 1e-8)
                self.current_k += 1
            else:
                self.current_k += 1

            if self.current_k >= self.C_min:
                Lk = max(loss.item(), 1e-8)
                ln_rho_hat = (np.log(Lk) - np.log(self.L0)) / (2 * self.current_k)
                rho_hat = np.exp(ln_rho_hat)

                if rho_hat < self.rho_min:
                    self.rho_min = rho_hat
                elif rho_hat > self.rho_min + self.epsilon_tol:
                    self.target_net.load_state_dict(self.online_net.state_dict())
                    self.adaptive_C_history.append(self.current_k)
                    self.adaptive_C_step_history.append(current_step)
                    self.current_k = 0
                    self.L0 = None
                    self.rho_min = float("inf")

        return loss.item()

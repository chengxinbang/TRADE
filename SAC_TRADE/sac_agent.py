"""SAC backbone retained from the local SAC-DMC implementation."""
import numpy as np
import torch
import torch.nn.functional as F

from actor import DiagGaussianActor
from critic import DoubleQCritic
from trade_scheduler import TRADEScheduler
import utils


class SACAgent:
    def __init__(self, obs_dim, action_dim, device, target_update_mode="hard_adaptive",
                 c_min=100, discount=0.99,
                 init_temperature=0.1, actor_lr=3e-4, critic_lr=3e-4,
                 alpha_lr=3e-4, critic_tau=0.005, batch_size=256,
                 hidden_dim=256, hidden_depth=2):
        if target_update_mode not in {"soft", "hard_adaptive"}:
            raise ValueError("Expected soft or hard_adaptive")
        self.device = torch.device(device)
        self.discount = discount
        self.batch_size = batch_size
        self.critic_tau = critic_tau
        self.target_update_mode = target_update_mode
        self.scheduler = TRADEScheduler(c_min)
        self.actor = DiagGaussianActor(obs_dim, action_dim, hidden_dim,
                                       hidden_depth, (-20, 2)).to(self.device)
        self.critic = DoubleQCritic(obs_dim, action_dim, hidden_dim, hidden_depth).to(self.device)
        self.critic_target = DoubleQCritic(obs_dim, action_dim, hidden_dim, hidden_depth).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.log_alpha = torch.tensor(np.log(init_temperature), device=self.device,
                                      requires_grad=True)
        self.target_entropy = -action_dim
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_refresh_count = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, observation, sample=False):
        obs = torch.as_tensor(observation, device=self.device, dtype=torch.float32).unsqueeze(0)
        distribution = self.actor(obs)
        action = distribution.sample() if sample else distribution.mean
        return action.clamp(-1, 1)[0].cpu().numpy()

    def _update_target_network(self, pre_update_loss, step):
        if self.target_update_mode == "soft":
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)
            self.target_refresh_count += 1
            return
        refresh = self.scheduler.update(pre_update_loss, step)
        if refresh:
            utils.hard_update_params(self.critic, self.critic_target)
            self.target_refresh_count += 1

    def update(self, replay_buffer, step):
        obs, action, reward, next_obs, _, not_done = replay_buffer.sample(self.batch_size)
        with torch.no_grad():
            distribution = self.actor(next_obs)
            next_action = distribution.rsample()
            log_probability = distribution.log_prob(next_action).sum(-1, keepdim=True)
            q1, q2 = self.critic_target(next_obs, next_action)
            target = reward + not_done * self.discount * (
                torch.min(q1, q2) - self.alpha.detach() * log_probability)
        q1, q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        monitored_loss = float(critic_loss.detach().item())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        distribution = self.actor(obs)
        policy_action = distribution.rsample()
        log_probability = distribution.log_prob(policy_action).sum(-1, keepdim=True)
        q1, q2 = self.critic(obs, policy_action)
        actor_loss = (self.alpha.detach() * log_probability - torch.min(q1, q2)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        alpha_loss = (self.alpha * (-log_probability - self.target_entropy).detach()).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        self._update_target_network(monitored_loss, step)
        return monitored_loss

    def checkpoint(self):
        return dict(protocol=TRADEScheduler.protocol, actor=self.actor.state_dict(),
                    critic=self.critic.state_dict(), target=self.critic_target.state_dict(),
                    log_alpha=self.log_alpha.detach().cpu(),
                    actor_optimizer=self.actor_optimizer.state_dict(),
                    critic_optimizer=self.critic_optimizer.state_dict(),
                    alpha_optimizer=self.log_alpha_optimizer.state_dict(),
                    scheduler=self.scheduler.state_dict(), mode=self.target_update_mode,
                    target_refresh_count=self.target_refresh_count)

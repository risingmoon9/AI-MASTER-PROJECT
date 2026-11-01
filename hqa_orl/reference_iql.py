"""PyTorch port of the pinned official IQL equations and Kitchen configuration.

Derived from ikostrikov/implicit_q_learning, commit
09d700248117881a75cb21f0adb95c6c8a694cb2. Copyright (c) 2021 Ilya
Kostrikov, Ashvin Nair, Sergey Levine. MIT license: reference/upstream/LICENSE.
See reference/IQL_REFERENCE.md for verified properties and protocol differences.
This implementation does not establish D4RL benchmark reproduction.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ReferenceIQLConfig:
    discount: float = 0.99
    tau: float = 0.005
    expectile: float = 0.7
    temperature: float = 0.5
    dropout_rate: float = 0.1
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    value_lr: float = 3e-4
    hidden_dim: int = 256
    hidden_depth: int = 2
    max_adv_weight: float = 100.0
    max_steps: int = 200000
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    action_clip_eps: float = 1e-5

    def __post_init__(self):
        if self.max_steps <= 0 or self.hidden_dim <= 0 or self.hidden_depth < 0:
            raise ValueError("Invalid reference IQL dimensions or max_steps")
        if not (0 < self.expectile < 1 and 0 <= self.dropout_rate < 1):
            raise ValueError("Invalid expectile/dropout_rate")
        if not (0 <= self.discount <= 1 and 0 <= self.tau <= 1):
            raise ValueError("Invalid discount/tau")
        if self.temperature <= 0 or self.max_adv_weight != 100.0:
            raise ValueError("IQL-Ref requires positive temperature and official cap 100")
        if min(self.actor_lr, self.critic_lr, self.value_lr) <= 0:
            raise ValueError("Learning rates must be positive")
        if not (self.log_std_min < self.log_std_max and 0 <= self.action_clip_eps < 1):
            raise ValueError("Invalid standard-deviation/action bounds")


def _linear(input_dim: int, output_dim: int) -> nn.Linear:
    layer = nn.Linear(input_dim, output_dim)
    # Official common.default_init: orthogonal(sqrt(2)), including output layers.
    nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
    nn.init.zeros_(layer.bias)
    return layer


class _MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_depth, output_dim,
                 dropout_rate=0.0):
        super().__init__()
        sizes = [input_dim] + [hidden_dim] * hidden_depth + [output_dim]
        self.layers = nn.ModuleList([_linear(a, b) for a, b in zip(sizes, sizes[1:])])
        self.dropout_rate = dropout_rate

    def forward(self, inputs, training=False):
        for layer in self.layers[:-1]:
            inputs = F.relu(layer(inputs))
            if self.dropout_rate:
                inputs = F.dropout(inputs, self.dropout_rate, training=training)
        return self.layers[-1](inputs)


class ReferencePolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, cfg):
        super().__init__()
        self.net = _MLP(obs_dim, cfg.hidden_dim, cfg.hidden_depth, action_dim,
                        cfg.dropout_rate)
        self.log_stds = nn.Parameter(torch.zeros(action_dim))
        self.log_std_min = cfg.log_std_min
        self.log_std_max = cfg.log_std_max

    def forward(self, obs, training=False):
        # The mean is bounded; the distribution itself is NOT tanh transformed.
        mean = torch.tanh(self.net(obs, training=training))
        log_std = self.log_stds.clamp(self.log_std_min, self.log_std_max)
        return mean, log_std.expand_as(mean)

    def log_prob(self, obs, actions, training=False):
        mean, log_std = self.forward(obs, training=training)
        # MultivariateNormalDiag log density; no inverse tanh or Jacobian term.
        return (-0.5 * ((actions - mean) * torch.exp(-log_std)).square()
                - log_std - 0.5 * math.log(2.0 * math.pi)).sum(-1, keepdim=True)

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        mean, log_std = self.forward(obs, training=False)
        action = mean if deterministic else mean + log_std.exp() * torch.randn_like(mean)
        return action.clamp(-1.0, 1.0)


class _TwinQ(nn.Module):
    def __init__(self, obs_dim, action_dim, cfg):
        super().__init__()
        self.q1 = _MLP(obs_dim + action_dim, cfg.hidden_dim, cfg.hidden_depth, 1)
        self.q2 = _MLP(obs_dim + action_dim, cfg.hidden_dim, cfg.hidden_depth, 1)

    def both(self, obs, actions):
        inputs = torch.cat((obs, actions), -1)
        return self.q1(inputs), self.q2(inputs)

    def forward(self, obs, actions):
        q1, q2 = self.both(obs, actions)
        return torch.minimum(q1, q2)


class ReferenceIQLAgent:
    def __init__(self, obs_dim: int, action_dim: int, cfg: ReferenceIQLConfig,
                 device: torch.device):
        self.cfg, self.device = cfg, device
        self.actor = ReferencePolicy(obs_dim, action_dim, cfg).to(device)
        self.q = _TwinQ(obs_dim, action_dim, cfg).to(device)
        self.v = _MLP(obs_dim, cfg.hidden_dim, cfg.hidden_depth, 1).to(device)
        self.target_q = _TwinQ(obs_dim, action_dim, cfg).to(device)
        self.target_q.load_state_dict(self.q.state_dict())
        self.target_q.requires_grad_(False)
        # Optax scale_by_adam defaults: b1=.9, b2=.999, eps=1e-8, eps_root=0.
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr,
                                          betas=(0.9, 0.999), eps=1e-8)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=cfg.critic_lr,
                                      betas=(0.9, 0.999), eps=1e-8)
        self.v_opt = torch.optim.Adam(self.v.parameters(), lr=cfg.value_lr,
                                      betas=(0.9, 0.999), eps=1e-8)
        self.total_updates = 0

    def actor_learning_rate(self):
        # Optax cosine_decay_schedule is evaluated at count 0 for first update.
        progress = min(self.total_updates, self.cfg.max_steps) / self.cfg.max_steps
        return self.cfg.actor_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    def update(self, batch: Dict[str, torch.Tensor], collect_metrics=True) -> Dict[str, float]:
        obs, next_obs = batch['obs'], batch['next_obs']
        actions = batch['actions'].clamp(-1.0 + self.cfg.action_clip_eps,
                                         1.0 - self.cfg.action_clip_eps)
        rewards = batch['rewards'].reshape(-1, 1)
        dones = batch['dones'].reshape(-1, 1)
        # No quality weighting or added BC term in any of the three objectives.
        with torch.no_grad():
            target_q = self.target_q(obs, actions)
        value = self.v(obs)
        diff = target_q - value
        expectile_weight = torch.where(diff > 0, diff.new_tensor(self.cfg.expectile),
                                       diff.new_tensor(1.0 - self.cfg.expectile))
        value_loss = (expectile_weight * diff.square()).mean()
        self.v_opt.zero_grad(set_to_none=True)
        value_loss.backward()
        self.v_opt.step()

        # Official order: updated V, old target Q -> actor -> Q -> target Q.
        with torch.no_grad():
            adv = target_q - self.v(obs)
            # Algebraically equivalent to min(exp(temperature * adv), 100),
            # but pre-exp clipping avoids an unnecessary intermediate overflow.
            aw = torch.exp((self.cfg.temperature * adv).clamp(max=math.log(100.0))).clamp(max=100.0)
        log_prob = self.actor.log_prob(obs, actions, training=True)
        actor_loss = -(aw * log_prob).mean()
        actor_lr = self.actor_learning_rate()
        for group in self.actor_opt.param_groups:
            group['lr'] = actor_lr
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        with torch.no_grad():
            q_target = rewards + self.cfg.discount * (1.0 - dones) * self.v(next_obs)
        q1, q2 = self.q.both(obs, actions)
        q_loss = ((q1 - q_target).square() + (q2 - q_target).square()).mean()
        self.q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_opt.step()
        with torch.no_grad():
            for parameter, target in zip(self.q.parameters(), self.target_q.parameters()):
                target.mul_(1.0 - self.cfg.tau).add_(parameter, alpha=self.cfg.tau)
        self.total_updates += 1
        if not collect_metrics:
            return {}
        # One transfer per logging step; no host synchronization on ordinary steps.
        values = torch.stack((value_loss.detach(), q_loss.detach(), actor_loss.detach(),
                              adv.mean(), aw.mean())).cpu().tolist()
        return dict(zip(('loss/value', 'loss/q', 'loss/actor', 'stats/adv_mean',
                         'stats/awr_mean'), values), **{'stats/actor_lr': actor_lr})

    def state_dict(self):
        return {'actor': self.actor.state_dict(), 'q': self.q.state_dict(),
                'v': self.v.state_dict(), 'target_q': self.target_q.state_dict(),
                'cfg': asdict(self.cfg), 'total_updates': self.total_updates,
                'optimizers': {'actor': self.actor_opt.state_dict(),
                               'q': self.q_opt.state_dict(), 'v': self.v_opt.state_dict()},
                'reference_commit': '09d700248117881a75cb21f0adb95c6c8a694cb2'}

    def load_state_dict(self, state):
        if state.get('cfg') != asdict(self.cfg):
            raise ValueError('Reference IQL configuration changed across checkpoint restore')
        for name in ('actor', 'q', 'target_q', 'v'):
            getattr(self, name).load_state_dict(state[name])
        for name in ('actor', 'q', 'v'):
            getattr(self, name + '_opt').load_state_dict(state['optimizers'][name])
        self.total_updates = int(state['total_updates'])
        if self.total_updates < 0:
            raise ValueError('Negative reference IQL update counter')

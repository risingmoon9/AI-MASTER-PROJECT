from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform


def mlp(input_dim: int, hidden_dim: int, output_dim: int, hidden_depth: int, activation=nn.ReLU) -> nn.Sequential:
    layers: List[nn.Module] = []
    last_dim = input_dim
    for _ in range(hidden_depth):
        layers += [nn.Linear(last_dim, hidden_dim), activation()]
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class TwinQ(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, hidden_depth: int = 2):
        super().__init__()
        in_dim = obs_dim + action_dim
        self.q1 = mlp(in_dim, hidden_dim, 1, hidden_depth)
        self.q2 = mlp(in_dim, hidden_dim, 1, hidden_depth)

    def both(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.both(obs, action)
        return torch.minimum(q1, q2)


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 256, hidden_depth: int = 2):
        super().__init__()
        self.v = mlp(obs_dim, hidden_dim, 1, hidden_depth)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v(obs)


class TanhGaussianPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        hidden_depth: int = 2,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.net = mlp(obs_dim, hidden_dim, 2 * action_dim, hidden_depth)
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(obs)
        mean, log_std = torch.chunk(out, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def dist(self, obs: torch.Tensor) -> TransformedDistribution:
        mean, log_std = self.forward(obs)
        base = Normal(mean, torch.exp(log_std))
        return TransformedDistribution(base, [TanhTransform(cache_size=1)])

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        action = torch.clamp(action, -1.0 + eps, 1.0 - eps)
        d = self.dist(obs)
        lp = d.log_prob(action)
        return lp.sum(dim=-1, keepdim=True)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        mean, log_std = self.forward(obs)
        if deterministic:
            return torch.tanh(mean)
        d = self.dist(obs)
        return d.sample()

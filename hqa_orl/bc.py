from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from .models import TanhGaussianPolicy


@dataclass
class BCConfig:
    actor_lr: float = 1e-4
    hidden_dim: int = 256
    hidden_depth: int = 2
    grad_clip_norm: float = 10.0


class BCAgent:
    """Behavior Cloning baseline.

    The policy architecture is the same TanhGaussianPolicy used by IQL,
    but training only minimizes negative log-likelihood of dataset actions.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        cfg: BCConfig,
        device: torch.device,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.cfg = cfg
        self.device = device

        self.actor = TanhGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=cfg.hidden_dim,
            hidden_depth=cfg.hidden_depth,
        ).to(device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.total_updates = 0

    def update(self, batch: Dict[str, torch.Tensor], collect_metrics: bool = True) -> Dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"]

        log_prob = self.actor.log_prob(obs, actions)
        loss = -log_prob.mean()

        self.actor_opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip_norm)
        self.actor_opt.step()

        self.total_updates += 1

        if not collect_metrics:
            return {}
        return {
            "loss/bc": float(loss.detach().cpu()),
            "stats/log_prob_mean": float(log_prob.mean().detach().cpu()),
            "stats/actor_grad_norm_before_clip": float(grad.detach().cpu()),
        }

    def state_dict(self) -> Dict[str, object]:
        return {
            "actor": self.actor.state_dict(),
            "cfg": self.cfg.__dict__,
            "total_updates": self.total_updates,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.total_updates = int(state.get("total_updates", 0))

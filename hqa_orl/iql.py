from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .models import TanhGaussianPolicy, TwinQ, ValueNet
from .utils import hard_update, soft_update


def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return weight * diff.pow(2)


@dataclass
class IQLConfig:
    discount: float = 0.99
    tau: float = 0.005
    expectile: float = 0.7
    temperature: float = 3.0
    max_adv_weight: float = 100.0
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    value_lr: float = 3e-4
    hidden_dim: int = 256
    hidden_depth: int = 2
    target_update_period: int = 1
    grad_clip_norm: float = 10.0
    actor_loss_normalization: str = 'self_normalized'
    use_quality_weights: bool = True
    use_adaptive_bc_regularization: bool = True
    lambda_base: float = 0.05
    lambda_rho_uncertainty: float = 1.0
    lambda_eta_support: float = 0.5
    lambda_min: float = 0.0
    lambda_max: float = 2.0
    regularization_mode: str = 'adaptive'
    lambda_constant: float = 0.0


class IQLAgent:
    def __init__(self, obs_dim: int, action_dim: int, cfg: IQLConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.actor = TanhGaussianPolicy(obs_dim, action_dim, cfg.hidden_dim, cfg.hidden_depth).to(device)
        self.q = TwinQ(obs_dim, action_dim, cfg.hidden_dim, cfg.hidden_depth).to(device)
        self.target_q = TwinQ(obs_dim, action_dim, cfg.hidden_dim, cfg.hidden_depth).to(device)
        self.v = ValueNet(obs_dim, cfg.hidden_dim, cfg.hidden_depth).to(device)
        hard_update(self.q, self.target_q)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=cfg.critic_lr)
        self.v_opt = torch.optim.Adam(self.v.parameters(), lr=cfg.value_lr)
        self.total_updates = 0

    def update(self, batch: Dict[str, torch.Tensor], collect_metrics: bool = True) -> Dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]
        if self.cfg.use_quality_weights:
            weights = batch["weights"]
        else:
            weights = torch.ones_like(batch["weights"])

        # Value update: expectile regression from Q(s,a) to V(s)
        with torch.no_grad():
            target_q_val = self.target_q(obs, actions)
        v = self.v(obs)
        value_loss_unreduced = expectile_loss(target_q_val - v, self.cfg.expectile)
        value_loss = (weights * value_loss_unreduced).sum() / (weights.sum() + 1e-6)
        self.v_opt.zero_grad(set_to_none=True)
        value_loss.backward()
        v_grad = torch.nn.utils.clip_grad_norm_(self.v.parameters(), self.cfg.grad_clip_norm)
        self.v_opt.step()

        # Critic update
        with torch.no_grad():
            target_v_next = self.v(next_obs)
            q_target = rewards + self.cfg.discount * (1.0 - dones) * target_v_next
        q1, q2 = self.q.both(obs, actions)
        q_loss_unreduced = (q1 - q_target).pow(2) + (q2 - q_target).pow(2)
        q_loss = (weights * q_loss_unreduced).sum() / (weights.sum() + 1e-6)
        self.q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        q_grad = torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip_norm)
        self.q_opt.step()

        # Actor update: advantage-weighted BC plus adaptive BC regularization.
        with torch.no_grad():
            q_for_adv = self.target_q(obs, actions)
            v_for_adv = self.v(obs)
            adv = q_for_adv - v_for_adv
            # Clamp before exp to avoid overflow and reduce actor collapse.
            log_aw = torch.clamp(adv * self.cfg.temperature, max=torch.log(torch.tensor(self.cfg.max_adv_weight, device=self.device)))
            aw = torch.exp(log_aw).clamp(max=self.cfg.max_adv_weight)

        log_prob = self.actor.log_prob(obs, actions)
        actor_weight = weights * aw
        if self.cfg.actor_loss_normalization == 'self_normalized':
            actor_loss = -(actor_weight * log_prob).sum() / (actor_weight.sum() + 1e-6)
        elif self.cfg.actor_loss_normalization == 'batch_mean':
            actor_loss = -(actor_weight * log_prob).mean()
        else:
            raise ValueError('Unknown actor_loss_normalization')

        if self.cfg.use_adaptive_bc_regularization:
            support = batch["support"]
            uncertainty = batch["uncertainty"]
        
            if self.cfg.regularization_mode == 'adaptive':
                lambda_sa = self.cfg.lambda_base * (
                    1.0
                    + self.cfg.lambda_rho_uncertainty * uncertainty
                    - self.cfg.lambda_eta_support * support
                )
                lambda_sa = lambda_sa.clamp(self.cfg.lambda_min, self.cfg.lambda_max)
            elif self.cfg.regularization_mode == 'constant':
                # Frozen whole-buffer average of the adaptive coefficient, not
                # a fitted scalar or a minibatch-dependent adaptive quantity.
                lambda_sa = torch.full_like(weights, self.cfg.lambda_constant)
            else:
                raise ValueError('regularization_mode must be adaptive or constant')
        
            # Important: use mean, not normalization by lambda_sa.sum().
            # Otherwise lambda_base is mostly cancelled and cannot control
            # the strength of adaptive regularization.
            bc_reg = (lambda_sa * (-log_prob)).mean()
            actor_loss = actor_loss + bc_reg
        else:
            lambda_sa = torch.zeros_like(weights)

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip_norm)
        self.actor_opt.step()

        if self.total_updates % self.cfg.target_update_period == 0:
            soft_update(self.q, self.target_q, self.cfg.tau)
        self.total_updates += 1

        if not collect_metrics:
            return {}
        return {
            "loss/value": float(value_loss.detach().cpu()),
            "loss/q": float(q_loss.detach().cpu()),
            "loss/actor": float(actor_loss.detach().cpu()),
            "stats/adv_mean": float(adv.mean().detach().cpu()),
            "stats/awr_mean": float(aw.mean().detach().cpu()),
            "stats/quality_weight_mean": float(weights.mean().detach().cpu()),
            "stats/lambda_mean": float(lambda_sa.mean().detach().cpu()),
            "stats/lambda_std": float(lambda_sa.std(unbiased=False).detach().cpu()),
            "stats/adv_std": float(adv.std(unbiased=False).detach().cpu()),
            "stats/awr_max": float(aw.max().detach().cpu()),
            "stats/awr_cap_fraction": float((aw >= self.cfg.max_adv_weight * (1 - 1e-6)).float().mean().cpu()),
            "stats/actor_weight_ess_fraction": float((actor_weight.sum().square() / (actor_weight.square().sum() * actor_weight.numel() + 1e-12)).detach().cpu()),
            "stats/q_mean": float(q_for_adv.mean().detach().cpu()),
            "stats/v_mean": float(v_for_adv.mean().detach().cpu()),
            "stats/log_prob_mean": float(log_prob.mean().detach().cpu()),
            "stats/actor_grad_norm_before_clip": float(actor_grad.detach().cpu()),
            "stats/q_grad_norm_before_clip": float(q_grad.detach().cpu()),
            "stats/v_grad_norm_before_clip": float(v_grad.detach().cpu()),
        }

    def state_dict(self) -> Dict[str, object]:
        return {
            "actor": self.actor.state_dict(),
            "q": self.q.state_dict(),
            "target_q": self.target_q.state_dict(),
            "v": self.v.state_dict(),
            "cfg": self.cfg.__dict__,
            "total_updates": self.total_updates,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.q.load_state_dict(state["q"])
        self.target_q.load_state_dict(state["target_q"])
        self.v.load_state_dict(state["v"])
        self.total_updates = int(state.get("total_updates", 0))

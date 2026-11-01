from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from .utils import RunningMeanStd


class OfflineReplayBuffer:
    def __init__(self, data: Dict[str, np.ndarray], normalize_obs: bool = True):
        self.data = data
        self.size = len(data["obs"])
        self.obs_dim = data["obs"].shape[-1]
        self.action_dim = data["actions"].shape[-1]

        self.obs_rms = RunningMeanStd()
        if normalize_obs:
            self.obs_rms.fit(data["obs"])
            self.obs = self.obs_rms.normalize(data["obs"])
            self.next_obs = self.obs_rms.normalize(data["next_obs"])
        else:
            self.obs = data["obs"].astype(np.float32)
            self.next_obs = data["next_obs"].astype(np.float32)

        self.actions = data["actions"].astype(np.float32)
        self.rewards = data["rewards"].astype(np.float32)
        self.dones = data["dones"].astype(np.float32)
        self.weights = data.get("quality_weight", np.ones((self.size, 1), dtype=np.float32)).astype(np.float32)
        self.support = data.get("support", np.ones((self.size, 1), dtype=np.float32)).astype(np.float32)
        self.uncertainty = data.get("uncertainty", np.zeros((self.size, 1), dtype=np.float32)).astype(np.float32)

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "dones": self.dones[idx],
            "weights": self.weights[idx],
            "support": self.support[idx],
            "uncertainty": self.uncertainty[idx],
        }
        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in batch.items()}

from __future__ import annotations

import os
import random
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def flatten_observation(obs: Any) -> np.ndarray:
    """Flatten one observation or a batched observation.

    Minari observations can be arrays or nested dictionaries, e.g. AntMaze and
    Kitchen have Dict observations. This function recursively concatenates all
    leaves in sorted-key order. If leaves are batched arrays of shape (T, ...),
    the result is (T, D). If leaves are unbatched, the result is (D,).
    """
    if isinstance(obs, dict):
        parts = [flatten_observation(obs[k]) for k in sorted(obs.keys())]
        # Promote all parts to batched or unbatched consistently.
        max_ndim = max(p.ndim for p in parts)
        if max_ndim == 1:
            return np.concatenate([p.reshape(-1) for p in parts], axis=0).astype(np.float32)
        return np.concatenate([p.reshape(p.shape[0], -1) for p in parts], axis=-1).astype(np.float32)
    arr = np.asarray(obs)
    if arr.dtype == object:
        # Some info fields may be object arrays; observations should not be.
        arr = arr.astype(np.float32)
    if arr.ndim == 0:
        return arr.reshape(1).astype(np.float32)
    return arr.astype(np.float32)


def discount_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
    y = np.zeros_like(x, dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(x))):
        running = float(x[t]) + discount * running
        y[t] = running
    return y


def soft_update(source: torch.nn.Module, target: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for p, tp in zip(source.parameters(), target.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * p.data)


def hard_update(source: torch.nn.Module, target: torch.nn.Module) -> None:
    target.load_state_dict(source.state_dict())


class RunningMeanStd:
    def __init__(self, eps: float = 1e-4):
        self.mean = None
        self.var = None
        self.count = eps

    def fit(self, x: np.ndarray) -> None:
        self.mean = x.mean(axis=0).astype(np.float32)
        self.var = (x.var(axis=0) + 1e-6).astype(np.float32)
        self.count = x.shape[0]

    def normalize(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.var is None:
            return x.astype(np.float32)
        return ((x - self.mean) / np.sqrt(self.var)).astype(np.float32)

    def state_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.mean = state.get("mean")
        self.var = state.get("var")
        self.count = state.get("count", 0)



def make_json_safe(x):
    """Convert numpy scalars and non-finite floats into strict-JSON-safe values."""
    import math
    import numpy as np

    if isinstance(x, dict):
        return {str(k): make_json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [make_json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return make_json_safe(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        value = float(x)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return x

def pad_or_truncate_last_dim(x, target_dim: int, pad_value: float = 0.0):
    """Pad or truncate the last dimension of an array to target_dim.

    This is mainly used for Minari Kitchen, where complete/partial/mixed
    datasets may have slightly different flattened observation dimensions.
    """
    import numpy as np

    x = np.asarray(x, dtype=np.float32)

    if x.shape[-1] == target_dim:
        return x.astype(np.float32)

    if x.shape[-1] > target_dim:
        return x[..., :target_dim].astype(np.float32)

    pad_width = [(0, 0)] * x.ndim
    pad_width[-1] = (0, target_dim - x.shape[-1])
    return np.pad(
        x,
        pad_width=pad_width,
        mode="constant",
        constant_values=pad_value,
    ).astype(np.float32)

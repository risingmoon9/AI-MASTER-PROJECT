from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .utils import discount_cumsum, flatten_observation, make_json_safe


ANTMAZE_HF_PREFIX = "hf://lukasz-sawala-goat/"


@dataclass
class EpisodeRecord:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    dones: np.ndarray
    truncations: np.ndarray
    returns_to_go: np.ndarray
    episode_return: float
    episode_length: int
    source_name: str
    source_prior: float
    source_dataset_id: str
    original_episode_id: int


def safe_load_dataset(dataset_id: str, download: bool = False):
    """Load a Minari dataset and try a Hugging Face fallback for community AntMaze.

    Minari's official docs show plain IDs for downloads and `hf://...` IDs for
    some community examples. This helper supports both.
    """
    import minari
    try:
        return minari.load_dataset(dataset_id, download=download)
    except Exception as first_error:
        if dataset_id.startswith("antmaze-"):
            try:
                return minari.load_dataset(ANTMAZE_HF_PREFIX + dataset_id, download=download)
            except Exception:
                raise first_error
        raise


def episode_to_record(
    episode: Any,
    source_name: str,
    source_prior: float,
    source_dataset_id: str,
    done_on_truncation: bool = False,
    discount: float = 0.99,
) -> EpisodeRecord:
    obs_all = flatten_observation(episode.observations)
    actions = np.asarray(episode.actions, dtype=np.float32)
    rewards = np.asarray(episode.rewards, dtype=np.float32).reshape(-1)
    terminations = np.asarray(episode.terminations, dtype=np.float32).reshape(-1)
    truncations = np.asarray(episode.truncations, dtype=np.float32).reshape(-1)

    # Minari stores T+1 observations and T actions/rewards.
    obs = obs_all[:-1].astype(np.float32)
    next_obs = obs_all[1:].astype(np.float32)
    if done_on_truncation:
        dones = np.clip(terminations + truncations, 0, 1).astype(np.float32)
    else:
        dones = terminations.astype(np.float32)

    returns_to_go = discount_cumsum(rewards, discount=discount)
    return EpisodeRecord(
        obs=obs,
        actions=actions,
        rewards=rewards,
        next_obs=next_obs,
        dones=dones,
        truncations=truncations,
        returns_to_go=returns_to_go,
        episode_return=float(rewards.sum()),
        episode_length=int(len(rewards)),
        source_name=source_name,
        source_prior=float(source_prior),
        source_dataset_id=source_dataset_id,
        original_episode_id=int(episode.id),
    )


def load_episode_records(
    dataset_id: str,
    source_name: str,
    source_prior: float,
    n_episodes: Optional[int] = None,
    seed: int = 0,
    download: bool = False,
    done_on_truncation: bool = False,
    discount: float = 0.99,
) -> List[EpisodeRecord]:
    dataset = safe_load_dataset(dataset_id, download=download)
    rng = np.random.default_rng(seed)
    all_indices = np.arange(dataset.total_episodes)
    if n_episodes is not None and n_episodes < len(all_indices):
        indices = rng.choice(all_indices, size=n_episodes, replace=False).tolist()
    else:
        indices = all_indices.tolist()

    records = []
    for ep in tqdm(dataset.iterate_episodes(episode_indices=indices), total=len(indices), desc=dataset_id):
        records.append(
            episode_to_record(
                ep,
                source_name=source_name,
                source_prior=source_prior,
                source_dataset_id=dataset_id,
                done_on_truncation=done_on_truncation,
                discount=discount,
            )
        )
    return records


def records_to_npz(records: List[EpisodeRecord], out_path: str, metadata: Dict[str, Any]) -> None:
    obs, actions, rewards, next_obs, dones, truncs = [], [], [], [], [], []
    rtg, ep_id, t_id, ep_ret, ep_len, src_prior, src_name = [], [], [], [], [], [], []
    source_dataset_id = []

    if len(records) == 0:
        raise ValueError("records_to_npz received an empty records list.")

    obs_dims = [int(rec.obs.shape[-1]) for rec in records]
    next_obs_dims = [int(rec.next_obs.shape[-1]) for rec in records]
    target_obs_dim = max(obs_dims + next_obs_dims)
    if len(set(obs_dims + next_obs_dims)) != 1:
        raise ValueError('Incompatible observation schemas: semantic alignment is required; tail padding is forbidden')

    action_dims = sorted(set(int(rec.actions.shape[-1]) for rec in records))
    if len(action_dims) != 1:
        raise ValueError(f"Action dimensions are inconsistent across records: {action_dims}")

    print(
        "[records_to_npz] observation dimensions found:",
        sorted(set(obs_dims + next_obs_dims)),
        "-> using target_obs_dim =",
        target_obs_dim,
    )

    metadata = dict(metadata)
    metadata["obs_dims_found"] = sorted(set(obs_dims + next_obs_dims))
    metadata["target_obs_dim"] = target_obs_dim
    metadata["obs_alignment"] = "equal_dimensions_required"

    for local_ep_id, rec in enumerate(records):
        n = rec.episode_length

        rec_obs = rec.obs
        rec_next_obs = rec.next_obs

        obs.append(rec_obs)
        actions.append(rec.actions)
        rewards.append(rec.rewards.reshape(-1, 1))
        next_obs.append(rec_next_obs)
        dones.append(rec.dones.reshape(-1, 1))
        truncs.append(rec.truncations.reshape(-1, 1))
        rtg.append(rec.returns_to_go.reshape(-1, 1))
        ep_id.append(np.full((n, 1), local_ep_id, dtype=np.int32))
        t_id.append(np.arange(n, dtype=np.int32).reshape(-1, 1))
        ep_ret.append(np.full((n, 1), rec.episode_return, dtype=np.float32))
        ep_len.append(np.full((n, 1), rec.episode_length, dtype=np.float32))
        src_prior.append(np.full((n, 1), rec.source_prior, dtype=np.float32))
        src_name.extend([rec.source_name] * n)
        source_dataset_id.extend([rec.source_dataset_id] * n)

    arrays = dict(
        obs=np.concatenate(obs, axis=0).astype(np.float32),
        actions=np.concatenate(actions, axis=0).astype(np.float32),
        rewards=np.concatenate(rewards, axis=0).astype(np.float32),
        next_obs=np.concatenate(next_obs, axis=0).astype(np.float32),
        dones=np.concatenate(dones, axis=0).astype(np.float32),
        truncations=np.concatenate(truncs, axis=0).astype(np.float32),
        returns_to_go=np.concatenate(rtg, axis=0).astype(np.float32),
        episode_id=np.concatenate(ep_id, axis=0).astype(np.int32),
        timestep=np.concatenate(t_id, axis=0).astype(np.int32),
        episode_return=np.concatenate(ep_ret, axis=0).astype(np.float32),
        episode_length=np.concatenate(ep_len, axis=0).astype(np.float32),
        source_prior=np.concatenate(src_prior, axis=0).astype(np.float32),
        source_name=np.asarray(src_name),
        source_dataset_id=np.asarray(source_dataset_id),
        metadata=np.asarray(json.dumps(make_json_safe(metadata), sort_keys=True, allow_nan=False)),
    )
    np.savez_compressed(out_path, **arrays)


def load_npz_buffer(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        result = {}
        for k in data.files:
            try:
                result[k] = data[k]
            except ValueError:
                if k != 'metadata':
                    raise
        return result

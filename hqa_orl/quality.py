from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def minmax_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - x.min()) / (x.max() - x.min() + eps)


def zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - x.mean()) / (x.std() + eps)


def build_episode_feature_table(data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Create one feature vector for each episode.

    Features are deliberately simple and environment-agnostic so the method can
    be used across MuJoCo, AntMaze, and Kitchen.
    """
    episode_ids = np.unique(data["episode_id"].reshape(-1))
    features = []
    for eid in episode_ids:
        mask = data["episode_id"].reshape(-1) == eid
        rewards = data["rewards"][mask].reshape(-1)
        actions = data["actions"][mask]
        obs = data["obs"][mask]
        next_obs = data["next_obs"][mask]
        ep_return = float(rewards.sum())
        ep_len = float(mask.sum())
        action_norm = float(np.linalg.norm(actions, axis=-1).mean())
        action_var = float(actions.var(axis=0).mean())
        obs_disp = float(np.linalg.norm(next_obs[-1] - obs[0])) if len(obs) > 0 else 0.0
        reward_mean = float(rewards.mean())
        reward_max = float(rewards.max())
        source_prior = float(data["source_prior"][mask][0])
        features.append([
            ep_return,
            ep_len,
            action_norm,
            action_var,
            obs_disp,
            reward_mean,
            reward_max,
            source_prior,
        ])
    return episode_ids.astype(np.int32), np.asarray(features, dtype=np.float32)


def compute_episode_clusters(
    data: Dict[str, np.ndarray],
    n_clusters: int = 5,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    episode_ids, features = build_episode_feature_table(data)
    n_clusters = min(n_clusters, len(episode_ids))
    scaled = StandardScaler().fit_transform(features)
    if n_clusters <= 1:
        labels = np.zeros(len(episode_ids), dtype=np.int32)
    else:
        labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto").fit_predict(scaled).astype(np.int32)
    transition_labels = np.zeros((len(data["obs"]), 1), dtype=np.int32)
    ep_to_label = {int(eid): int(label) for eid, label in zip(episode_ids, labels)}
    for i, eid in enumerate(data["episode_id"].reshape(-1)):
        transition_labels[i, 0] = ep_to_label[int(eid)]
    return labels, transition_labels


def compute_knn_support(
    obs: np.ndarray,
    actions: np.ndarray,
    max_fit_points: int = 100000,
    k: int = 10,
    batch_size: int = 50000,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Approximate local state-action support with kNN density.

    Smaller kNN distance implies stronger in-dataset support. Returns support in
    [0, 1] and uncertainty in [0, 1].
    """
    rng = np.random.default_rng(seed)
    x = np.concatenate([obs, actions], axis=-1).astype(np.float32)
    if len(x) > max_fit_points:
        fit_idx = rng.choice(len(x), size=max_fit_points, replace=False)
        fit_x = x[fit_idx]
    else:
        fit_x = x

    scaler = StandardScaler()
    fit_scaled = scaler.fit_transform(fit_x)
    nn = NearestNeighbors(n_neighbors=min(k, len(fit_scaled)), algorithm="auto", n_jobs=-1)
    nn.fit(fit_scaled)

    kth_dist = np.zeros(len(x), dtype=np.float32)
    for start in tqdm(range(0, len(x), batch_size), desc="kNN support"):
        end = min(start + batch_size, len(x))
        scaled = scaler.transform(x[start:end])
        dist, _ = nn.kneighbors(scaled, return_distance=True)
        kth_dist[start:end] = dist[:, -1]

    dist_norm = minmax_norm(kth_dist)
    support = 1.0 - dist_norm
    uncertainty = dist_norm
    return support.reshape(-1, 1).astype(np.float32), uncertainty.reshape(-1, 1).astype(np.float32)


def compute_quality_weights(
    data: Dict[str, np.ndarray],
    n_clusters: int = 5,
    seed: int = 0,
    alpha_return: float = 0.55,
    alpha_source: float = 0.30,
    alpha_success: float = 0.15,
    beta_rtg: float = 0.45,
    beta_reward: float = 0.20,
    beta_support: float = 0.25,
    beta_uncertainty: float = 0.10,
    use_knn_support: bool = False,
    max_knn_points: int = 100000,
) -> Dict[str, np.ndarray]:
    n = len(data["obs"])
    _, cluster_labels = compute_episode_clusters(data, n_clusters=n_clusters, seed=seed)

    ep_returns = data["episode_return"].reshape(-1)
    source_prior = data["source_prior"].reshape(-1)
    returns_to_go = data["returns_to_go"].reshape(-1)
    rewards = data["rewards"].reshape(-1)

    ret_norm = minmax_norm(ep_returns)
    source_norm = np.clip(source_prior, 0.0, 1.0)
    # Robust success proxy. For sparse non-negative tasks such as AntMaze,
    # a 75% quantile can be 0, which would incorrectly mark all zero-return
    # episodes as successful. Prefer positive return when possible.
    if float(ep_returns.max()) <= float(ep_returns.min()) + 1e-6:
        success = np.zeros_like(ep_returns, dtype=np.float32)
    else:
        threshold = float(np.quantile(ep_returns, 0.75))
        if threshold <= 1e-6 and float(ep_returns.max()) > 1e-6:
            success = (ep_returns > 1e-6).astype(np.float32)
        else:
            success = (ep_returns > threshold).astype(np.float32)
    traj_quality = alpha_return * ret_norm + alpha_source * source_norm + alpha_success * success

    if use_knn_support:
        support, uncertainty = compute_knn_support(
            data["obs"], data["actions"], max_fit_points=max_knn_points, seed=seed
        )
    else:
        support = np.ones((n, 1), dtype=np.float32)
        uncertainty = np.zeros((n, 1), dtype=np.float32)

    rtg_norm = minmax_norm(returns_to_go)
    reward_norm = minmax_norm(rewards)
    trans_quality = (
        beta_rtg * rtg_norm
        + beta_reward * reward_norm
        + beta_support * support.reshape(-1)
        - beta_uncertainty * uncertainty.reshape(-1)
    )

    # Cluster diversity balance: rare clusters receive a moderate boost.
    cluster_ids = cluster_labels.reshape(-1)
    _, counts = np.unique(cluster_ids, return_counts=True)
    count_map = {c: cnt for c, cnt in zip(np.unique(cluster_ids), counts)}
    balance = np.asarray([1.0 / np.sqrt(count_map[c] + 1e-6) for c in cluster_ids], dtype=np.float32)
    balance = balance / (balance.mean() + 1e-6)

    raw = zscore(traj_quality) + zscore(trans_quality)
    weights = 1.0 / (1.0 + np.exp(-raw))
    weights = weights * balance
    weights = np.clip(weights, 0.05, 5.0)
    weights = weights / (weights.mean() + 1e-6)

    return {
        "quality_weight": weights.reshape(-1, 1).astype(np.float32),
        "trajectory_quality": traj_quality.reshape(-1, 1).astype(np.float32),
        "transition_quality": trans_quality.reshape(-1, 1).astype(np.float32),
        "cluster_id": cluster_labels.astype(np.int32),
        "support": support.astype(np.float32),
        "uncertainty": uncertainty.astype(np.float32),
    }


def save_quality_augmented_npz(input_data: Dict[str, np.ndarray], quality: Dict[str, np.ndarray], out_path: str) -> None:
    merged = {**input_data, **quality}
    np.savez_compressed(out_path, **merged)

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import sys
from tqdm import trange

from .minari_io import safe_load_dataset
from .utils import RunningMeanStd, flatten_observation


def _extract_success(info: Any):
    """Return True/False when success info exists; otherwise return None."""
    if not isinstance(info, dict):
        return None

    for key in ("success", "is_success", "goal_achieved"):
        if key in info:
            value = info[key]
            if isinstance(value, (list, tuple, np.ndarray)):
                return bool(np.asarray(value).astype(bool).any())
            return bool(value)

    return None


@torch.no_grad()
def evaluate_policy(
    actor,
    eval_dataset_id: str,
    obs_rms_state: Optional[Dict] = None,
    episodes: int = 20,
    seed: int = 0,
    device: torch.device = torch.device("cpu"),
    deterministic: bool = True,
    download: bool = False,
    target_obs_dim: Optional[int] = None,
) -> Dict[str, Optional[float]]:
    """Evaluate a policy in the recovered Minari environment.

    Evaluation conventions:
    - download=False by default, so evaluation does not hit Hugging Face again.
    - success_rate uses real environment info only, not `return > 0`.
    - positive_return_rate is reported separately.
    - normalized score is None when Minari cannot compute it, avoiding illegal JSON NaN.
    """
    import minari
    dataset = safe_load_dataset(eval_dataset_id, download=download)
    kitchen = 'kitchen' in eval_dataset_id.lower()
    if kitchen and target_obs_dim != 81:
        raise ValueError('Kitchen evaluation requires canonical 81-dimensional checkpoints')
    try:
        env = dataset.recover_environment(eval_env=True)
    except Exception:
        env = dataset.recover_environment()
    if kitchen:
        base = env.unwrapped
        if base.remove_task_when_completed or base.terminate_on_tasks_completed:
            env.close()
            raise ValueError('Kitchen reward/termination flags do not match the common-task protocol')
        if env.spec.max_episode_steps != 280:
            env.close()
            raise ValueError('Kitchen evaluation horizon must be 280')

    obs_rms = RunningMeanStd()
    if obs_rms_state is not None:
        obs_rms.load_state_dict(obs_rms_state)

    action_low = getattr(env.action_space, "low", None)
    action_high = getattr(env.action_space, "high", None)

    returns = []
    lengths = []
    successes = []
    positive_returns = []
    task_fractions = []
    episode_records = []
    tasks = {'microwave', 'kettle', 'light switch', 'slide cabinet'}

    try:
        actor.eval()
        for ep in trange(episodes, desc="eval", leave=False, disable=not sys.stderr.isatty()):
            obs, info = env.reset(seed=seed + ep)
            done = False
            total = 0.0
            length = 0
            success_flag = _extract_success(info)
            has_success_info = success_flag is not None
            success_flag = bool(success_flag) if success_flag is not None else False
            completed = set()

            while not done:
                obs_vec = flatten_observation(obs).reshape(1, -1)
                if target_obs_dim is not None and obs_vec.shape[-1] != target_obs_dim:
                    raise ValueError(f'Observation schema mismatch: {obs_vec.shape[-1]} vs {target_obs_dim}')
                if kitchen:
                    if set(obs['achieved_goal']) != tasks or set(obs['desired_goal']) != tasks:
                        raise ValueError('Kitchen task schema mismatch')
                obs_vec = obs_rms.normalize(obs_vec)
                obs_t = torch.as_tensor(obs_vec, dtype=torch.float32, device=device)
                action = actor.act(obs_t, deterministic=deterministic).cpu().numpy()[0]

                if action_low is not None and action_high is not None:
                    action = np.clip(action, action_low, action_high)

                obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                total += float(reward)
                length += 1
                if kitchen:
                    # Track exact environment task events across the whole episode.
                    if 'episode_task_completions' not in info:
                        raise ValueError('Environment lacks task-completion instrumentation')
                    completed.update(set(info['episode_task_completions']) & tasks)
                s = _extract_success(info)
                if s is not None:
                    has_success_info = True
                    success_flag = success_flag or bool(s)

            returns.append(total)
            lengths.append(length)
            if kitchen:
                success_flag = completed == tasks
                has_success_info = True
                task_fractions.append(len(completed) / len(tasks))
            if has_success_info:
                successes.append(float(success_flag))
            positive_returns.append(float(total > 0.0))
            episode_records.append({'env_seed':seed+ep,'return':total,'length':length,'completed_tasks':sorted(completed) if kitchen else None,'task_completion_fraction':len(completed)/4 if kitchen else None,'all_tasks_success':bool(success_flag) if has_success_info else None})

            result: Dict[str, Optional[float]] = {
                "return_mean": float(np.mean(returns)),
                "return_std": float(np.std(returns)),
                "length_mean": float(np.mean(lengths)),
                "success_rate": float(np.mean(successes)) if len(successes) > 0 else None,
                "positive_return_rate": float(np.mean(positive_returns)),
            }

        if kitchen:
            result["normalized_score_mean"] = None
            result["normalized_score_std"] = None
            result['task_completion_fraction_mean'] = float(np.mean(task_fractions))
            result['all_tasks_success_rate'] = float(np.mean(successes))
            result['reward_protocol'] = 'kitchen_common81_persistent_v1'
        else:
            try:
                normalized = minari.get_normalized_score(dataset, np.asarray(returns, dtype=np.float32))
                result['normalized_score_mean'] = float(np.mean(normalized))
                result['normalized_score_std'] = float(np.std(normalized))
            except Exception:
                result['normalized_score_mean'] = None
                result['normalized_score_std'] = None
        result['per_episode'] = episode_records
        return result
    finally:
        env.close()

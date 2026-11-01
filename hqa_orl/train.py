from __future__ import annotations

import json
import math
import os
import copy
import hashlib
import random
import tempfile
import sys
from dataclasses import fields
from typing import Dict, Optional

import numpy as np
import torch
from tqdm import trange

from .evaluation import evaluate_policy
from .iql import IQLAgent, IQLConfig
from .minari_io import load_npz_buffer
from .replay_buffer import OfflineReplayBuffer
from .utils import ensure_dir, get_device, load_yaml, make_json_safe, set_seed


def make_iql_config(cfg: Dict) -> IQLConfig:
    tr = cfg["training"]
    hqa = cfg.get("hqa", {})
    return IQLConfig(
        discount=tr.get("discount", 0.99),
        tau=tr.get("tau", 0.005),
        expectile=tr.get("expectile", 0.7),
        temperature=tr.get("temperature", 3.0),
        max_adv_weight=tr.get("max_adv_weight", 20.0),
        actor_lr=tr.get("actor_lr", 1e-4),
        critic_lr=tr.get("critic_lr", 3e-4),
        value_lr=tr.get("value_lr", 3e-4),
        hidden_dim=tr.get("hidden_dim", 256),
        hidden_depth=tr.get("hidden_depth", 2),
        target_update_period=tr.get("target_update_period", 1),
        grad_clip_norm=tr.get("grad_clip_norm", 10.0),
        actor_loss_normalization=tr.get('actor_loss_normalization', 'self_normalized'),
        use_quality_weights=hqa.get("use_quality_weights", True),
        use_adaptive_bc_regularization=hqa.get("use_adaptive_bc_regularization", True),
        lambda_base=hqa.get("lambda_base", 0.05),
        lambda_rho_uncertainty=hqa.get("lambda_rho_uncertainty", 1.0),
        lambda_eta_support=hqa.get("lambda_eta_support", 0.5),
        lambda_min=hqa.get("lambda_min", 0.0),
        lambda_max=hqa.get("lambda_max", 2.0),
        regularization_mode=hqa.get('regularization_mode', 'adaptive'),
        lambda_constant=hqa.get('lambda_constant', 0.0),
    )


def build_agent(obs_dim, action_dim, cfg, device):
    """One explicit factory shared by training and checkpoint evaluation."""
    algorithm = cfg.get('algorithm', 'iql')
    tr = cfg['training']
    if algorithm == 'iql':
        return IQLAgent(obs_dim, action_dim, make_iql_config(cfg), device)
    if algorithm == 'bc':
        from .bc import BCAgent, BCConfig
        bc_cfg = BCConfig(**{f.name: tr[f.name] for f in fields(BCConfig) if f.name in tr})
        return BCAgent(obs_dim, action_dim, bc_cfg, device)
    if algorithm == 'reference_iql':
        from .reference_iql import ReferenceIQLAgent, ReferenceIQLConfig
        ref_kwargs = {f.name: tr[f.name] for f in fields(ReferenceIQLConfig) if f.name in tr}
        ref_kwargs['max_steps'] = int(tr['updates'])
        return ReferenceIQLAgent(obs_dim, action_dim, ReferenceIQLConfig(**ref_kwargs), device)
    raise ValueError(f'Unknown algorithm: {algorithm}')


def _optimizer_dict(agent):
    return {name: getattr(agent, name + '_opt') for name in ('actor', 'q', 'v')
            if hasattr(agent, name + '_opt')}


def _apply_overlay(data, cfg, base_sha256):
    """Only permit quality diagnostics, never silently replace transitions."""
    path = cfg.get('data_overlay')
    if not path:
        return None
    expected = cfg.get('data_overlay_sha256')
    checksum = _file_sha256(path)
    if not expected or checksum != expected:
        raise ValueError('Overlay SHA256 mismatch or missing declared hash')
    from scripts.prepare_data_variants import load_overlay
    merged, _ = load_overlay(data, path, expected_sha256=base_sha256)
    data.update(merged)
    return checksum


def _metric_value(metrics: Dict[str, object], key: str) -> Optional[float]:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _select_eval_score(eval_metrics: Dict[str, object], eval_dataset_id: str) -> float:
    """Choose a stable metric for best-checkpoint selection.

    Minari normalized score is not available for some MuJoCo datasets, so fall
    back to return. For AntMaze/Kitchen, normalized score is preferred when it
    exists; success_rate can be inspected separately.
    """
    normalized = _metric_value(eval_metrics, "normalized_score_mean")
    if normalized is not None:
        return normalized
    ret = _metric_value(eval_metrics, "return_mean")
    if ret is not None:
        return ret
    return -float("inf")


def _checkpoint_payload(agent: IQLAgent, buffer: OfflineReplayBuffer, cfg: Dict, data_path: str,
                        eval_dataset_id: str, step: int, extra: Optional[Dict] = None) -> Dict:
    payload = {
        "agent": agent.state_dict(),
        "obs_rms": buffer.obs_rms.state_dict(),
        "obs_dim": buffer.obs_dim,
        "action_dim": buffer.action_dim,
        "config": cfg,
        "data_path": data_path,
        "eval_dataset_id": eval_dataset_id,
        "step": step,
    }
    if extra:
        payload.update(extra)
    return payload


def _atomic_save(payload: Dict, path: str) -> None:
    """Commit only complete checkpoints, retaining the previous file on failure."""
    directory = os.path.dirname(os.path.abspath(path))
    fd, temporary = tempfile.mkstemp(prefix='.checkpoint-', suffix='.tmp', dir=directory)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _save_checkpoint(path: str, agent: IQLAgent, buffer: OfflineReplayBuffer, cfg: Dict, data_path: str,
                     eval_dataset_id: str, step: int, extra: Optional[Dict] = None) -> None:
    _atomic_save(_checkpoint_payload(agent, buffer, cfg, data_path, eval_dataset_id, step, extra), path)


def _restore_best_checkpoint(payload: Dict, path: str) -> None:
    """Restore exact serialized best bytes; CUDA-to-CPU loading must not change its identity."""
    content = payload.get('best_checkpoint_bytes')
    expected = payload.get('best_sha256')
    if payload['best_payload'] is None:
        if os.path.exists(path):
            os.remove(path)
        return
    if not isinstance(content, bytes) or hashlib.sha256(content).hexdigest() != expected:
        raise ValueError('Resume checkpoint lacks a valid serialized best checkpoint')
    if os.path.isfile(path) and _file_sha256(path) == expected:
        return
    directory = os.path.dirname(os.path.abspath(path))
    fd, temporary = tempfile.mkstemp(prefix='.best-checkpoint-', suffix='.tmp', dir=directory)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _file_sha256(path: str, limit: Optional[int] = None) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        remaining = limit
        while remaining is None or remaining > 0:
            chunk = stream.read(1024 * 1024 if remaining is None else min(1024 * 1024, remaining))
            if not chunk:
                if remaining:
                    raise ValueError('Metrics log is shorter than the committed checkpoint offset')
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return digest.hexdigest()


def _rng_state() -> Dict:
    return {
        'python': random.getstate(), 'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state: Dict) -> None:
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    if state['torch_cuda']:
        if not torch.cuda.is_available() or len(state['torch_cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA device count differs from the resume checkpoint')
        torch.cuda.set_rng_state_all([value.cpu() for value in state['torch_cuda']])


def train_iql(config_path: str, data_path: str, eval_dataset_id: str, run_name: str,
              resume: bool = False) -> str:
    cfg = load_yaml(config_path)
    seed = int(cfg.get("seed", 0))
    device = get_device(cfg.get("device", "cuda"))

    data = load_npz_buffer(data_path)
    if 'kitchen' in eval_dataset_id.lower():
        metadata = json.loads(str(data['metadata']))
        if metadata.get('schema') != 'kitchen_common81_persistent_v1' or data['obs'].shape[1] != 81:
            raise ValueError('Kitchen requires canonical 81-dimensional common-task data')
        if cfg.get('data_protocol') != metadata['schema']:
            raise ValueError('Configuration/data protocol mismatch')
        if cfg.get('hqa', {}).get('use_quality_weights', True) or cfg.get('hqa', {}).get('use_adaptive_bc_regularization', True):
            if metadata.get('quality_status') != 'recomputed':
                raise ValueError('Compute quality on common-task data before HQA training')
    data_sha256 = _file_sha256(data_path)
    overlay_sha256 = _apply_overlay(data, cfg, data_sha256)
    # Rebuild after applying weights; the transition arrays stay unchanged.
    buffer = OfflineReplayBuffer(data, normalize_obs=cfg["training"].get("normalize_obs", True))

    ckpt_dir = cfg.get("paths", {}).get("checkpoint_dir", "checkpoints")
    result_dir = cfg.get("paths", {}).get("result_dir", "results")
    ensure_dir(ckpt_dir)
    ensure_dir(result_dir)

    final_ckpt_path = os.path.join(ckpt_dir, f"{run_name}.pt")
    best_ckpt_path = os.path.join(ckpt_dir, f"{run_name}_best.pt")
    metrics_path = os.path.join(result_dir, f"{run_name}.jsonl")

    tr = cfg["training"]
    batch_size = int(tr.get("batch_size", 256))
    updates = int(tr.get("updates", 300000))
    eval_interval = int(tr.get("eval_interval", 5000))
    log_interval = int(tr.get("log_interval", 1000))
    save_interval = int(tr.get("save_interval", 50000))
    eval_episodes = int(tr.get("eval_episodes", 20))
    save_step_checkpoints = bool(tr.get("save_step_checkpoints", False))

    best_score = -float("inf")
    best_step = -1
    best_payload = None
    best_checkpoint_bytes = None
    best_sha256 = None
    start_step = 0
    resume_payload = None
    if resume:
        if not os.path.isfile(final_ckpt_path):
            raise FileNotFoundError(f'Resume checkpoint does not exist: {final_ckpt_path}')
        # Only load checkpoints created by this trusted project, never arbitrary uploads.
        resume_payload = torch.load(final_ckpt_path, map_location='cpu', weights_only=False)
        if resume_payload.get('resume_format') not in {'hqa_resume_v1', 'hqa_supplement_v1'}:
            raise ValueError('Checkpoint lacks complete optimizer/RNG resume state')
        if resume_payload['config'] != cfg:
            raise ValueError('Resume configuration does not exactly match the checkpoint')
        if resume_payload['data_sha256'] != data_sha256:
            raise ValueError('Resume dataset SHA256 does not match the checkpoint')
        if resume_payload.get('overlay_sha256') != overlay_sha256:
            raise ValueError('Resume overlay SHA256 does not match the checkpoint')
        if resume_payload['eval_dataset_id'] != eval_dataset_id or resume_payload['run_name'] != run_name:
            raise ValueError('Resume experiment identity does not match the checkpoint')
        if resume_payload['device_type'] != device.type:
            raise ValueError('Resume device type must match the original run')
        if resume_payload['torch_version'] != str(torch.__version__):
            raise ValueError('Resume requires the same PyTorch version')
        start_step = int(resume_payload['step'])
        if not 0 <= start_step <= updates:
            raise ValueError('Resume checkpoint has an invalid training step')
        if not os.path.isfile(metrics_path):
            raise FileNotFoundError(f'Resume metrics log does not exist: {metrics_path}')
        committed_offset = int(resume_payload['metrics_offset'])
        if committed_offset < 0 or _file_sha256(metrics_path, committed_offset) != resume_payload['metrics_sha256']:
            raise ValueError('Resume metrics log does not match the committed checkpoint')
        if resume_payload.get('training_complete') and start_step != updates:
            raise ValueError('Completed checkpoint step differs from the configured update budget')
        _restore_best_checkpoint(resume_payload, best_ckpt_path)
        if resume_payload.get('training_complete'):
            # Preserve checkpoint byte identity so cached final evaluation remains usable.
            with open(metrics_path, 'r+b') as stream:
                stream.truncate(committed_offset)
                stream.flush()
                os.fsync(stream.fileno())
            return final_ckpt_path

    # A completed resume above does not initialize a model or modify global RNGs.
    set_seed(seed)
    agent = build_agent(buffer.obs_dim, buffer.action_dim, cfg, device)
    if resume:
        agent.load_state_dict(resume_payload['agent'])
        for name, opt in _optimizer_dict(agent).items():
            opt.load_state_dict(resume_payload['optimizers'][name])
        buffer.obs_rms.load_state_dict(resume_payload['obs_rms'])
        best_score = resume_payload['best_score']
        best_step = resume_payload['best_step']
        best_payload = resume_payload['best_payload']
        best_checkpoint_bytes = resume_payload.get('best_checkpoint_bytes')
        best_sha256 = resume_payload.get('best_sha256')
        _restore_rng(resume_payload['rng'])

    with open(metrics_path, 'r+b' if resume else 'w+b') as mf:
        if resume:
            mf.seek(committed_offset)
            mf.truncate()

        def write_row(row: Dict) -> None:
            mf.write((json.dumps(make_json_safe(row), ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8'))
            mf.flush()

        def save_latest(step: int, complete: bool = False) -> Dict:
            mf.flush()
            os.fsync(mf.fileno())
            payload = _checkpoint_payload(agent, buffer, cfg, data_path, eval_dataset_id, step, {
                'resume_format': 'hqa_resume_v1', 'run_name': run_name,
                'data_sha256': data_sha256, 'device_type': device.type,
                'overlay_sha256': overlay_sha256,
                'torch_version': str(torch.__version__),
                'optimizers': {name: opt.state_dict() for name, opt in _optimizer_dict(agent).items()},
                'rng': _rng_state(), 'best_score': best_score, 'best_step': best_step,
                'best_payload': best_payload, 'best_ckpt_path': best_ckpt_path,
                'best_checkpoint_bytes': best_checkpoint_bytes, 'best_sha256': best_sha256,
                'metrics_offset': mf.tell(), 'metrics_sha256': _file_sha256(metrics_path, mf.tell()),
                'training_complete': complete,
            })
            _atomic_save(payload, final_ckpt_path)
            return payload

        if not resume:
            save_latest(0)
        for step in trange(start_step + 1, updates + 1, desc=run_name, disable=not sys.stderr.isatty()):
            agent.actor.train()
            batch = buffer.sample(batch_size, device)
            metrics = agent.update(batch, collect_metrics=(step % log_interval == 0))

            if step % log_interval == 0:
                if not all(math.isfinite(float(value)) for value in metrics.values()):
                    raise FloatingPointError(f'Nonfinite training diagnostic at update {step}')
                row = make_json_safe({"step": step, **metrics})
                write_row(row)

            if step % eval_interval == 0:
                eval_metrics = evaluate_policy(
                    agent.actor,
                    eval_dataset_id=eval_dataset_id,
                    obs_rms_state=buffer.obs_rms.state_dict(),
                    episodes=eval_episodes,
                    seed=int(tr.get('validation_seed_start', 10000)),
                    device=device,
                    download=False,
                    target_obs_dim=buffer.obs_dim,
                )
                score = _select_eval_score(eval_metrics, eval_dataset_id)
                if not math.isfinite(score):
                    raise FloatingPointError(f'Nonfinite validation return at update {step}')
                is_best = score > best_score
                if is_best:
                    best_score = score
                    best_step = step
                    best_payload = copy.deepcopy(_checkpoint_payload(
                        agent,
                        buffer,
                        cfg,
                        data_path,
                        eval_dataset_id,
                        step,
                        extra={
                            "best_score": best_score,
                            "best_step": best_step,
                            "best_eval_metrics": eval_metrics,
                        },
                    ))
                    _atomic_save(best_payload, best_ckpt_path)
                    with open(best_ckpt_path, 'rb') as stream:
                        best_checkpoint_bytes = stream.read()
                    best_sha256 = hashlib.sha256(best_checkpoint_bytes).hexdigest()

                row = make_json_safe({
                    "step": step,
                    "best_step": best_step,
                    "best_score": best_score,
                    "is_best": is_best,
                    **{f"eval/{k}": v for k, v in eval_metrics.items()},
                })
                write_row(row)
                print(json.dumps({'step': step, 'return_mean': eval_metrics['return_mean'],
                                  'best_step': best_step, 'best_score': best_score}), flush=True)

            if step % save_interval == 0 or step % eval_interval == 0:
                payload = save_latest(step)
                if save_step_checkpoints and step % save_interval == 0:
                    step_ckpt_path = os.path.join(ckpt_dir, f"{run_name}_step{step}.pt")
                    _atomic_save(payload, step_ckpt_path)

        save_latest(updates, complete=True)
    return final_ckpt_path

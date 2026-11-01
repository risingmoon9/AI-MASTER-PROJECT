"""Verify the frozen data, experiment configurations and local Kitchen simulator."""
import argparse
import importlib.metadata
import json
import platform
import sys

try:
    from .runtime import ROOT, initialize
except ImportError:
    from runtime import ROOT, initialize


def preflight(require_cuda=True, check_versions=True):
    initialize()
    import numpy as np
    import torch
    from hqa_orl.minari_io import safe_load_dataset
    from hqa_orl.utils import flatten_observation
    from scripts.validate_data import validate
    from scripts.prepare_data_variants import load_overlay
    from scripts.experiment_common import (atomic_json, load_protocol, project_path,
                                           validate_protocol, verify_declared_data)

    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        raise RuntimeError('Use Python 3.10, 3.11, or 3.12.')
    pinned = {'numpy': '1.26.4', 'h5py': '3.11.0', 'scikit-learn': '1.5.1',
              'gymnasium': '1.3.0', 'gymnasium-robotics': '1.4.2',
              'minari': '0.5.3', 'mujoco': '3.10.0', 'torch': '2.7.1'}
    versions = {name: importlib.metadata.version(name) for name in pinned}
    mismatch = {name: version for name, version in versions.items()
                if version.split('+')[0] != pinned[name]}
    if check_versions and mismatch:
        raise RuntimeError(f'Version mismatch: {mismatch}. Use the pinned environment in README.md.')
    protocol = load_protocol()
    validate_protocol()
    hashes = verify_declared_data()
    data_paths = sorted({row['data'] for row in protocol['runs']})
    data_reports = {path: validate(project_path(path)) for path in data_paths}
    overlay_reports = []
    for row in protocol['runs']:
        if row.get('overlay') and row['overlay'] not in overlay_reports:
            with np.load(project_path(row['data']), allow_pickle=False) as archive:
                base = {key: archive[key] for key in archive.files}
            load_overlay(base, project_path(row['overlay']), expected_sha256=hashes[row['data']])
            overlay_reports.append(row['overlay'])
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Use a GPU instance and the matching PyTorch build.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(32, 32, device=device, requires_grad=True)
    loss = (x @ x.T).square().mean()
    loss.backward()
    if not torch.isfinite(loss).item() or not torch.isfinite(x.grad).all().item():
        raise RuntimeError('Device arithmetic check failed')
    if device.type == 'cuda':
        torch.cuda.synchronize()
    dataset = safe_load_dataset('D4RL/kitchen/complete-v2', download=False)
    env = dataset.recover_environment()
    try:
        assert env.spec.max_episode_steps == 280, 'Wrong episode horizon'
        assert not env.unwrapped.remove_task_when_completed
        assert not env.unwrapped.terminate_on_tasks_completed
        tasks = {'kettle', 'light switch', 'microwave', 'slide cabinet'}
        obs, _ = env.reset(seed=12000)
        assert flatten_observation(obs).shape == (81,)
        assert set(obs['achieved_goal']) == set(obs['desired_goal']) == tasks
        assert env.action_space.shape == (9,)
        for step in range(1, 281):
            obs, reward, terminated, truncated, info = env.step(np.zeros(9))
            expected = sum(np.linalg.norm(obs['achieved_goal'][key] - obs['desired_goal'][key]) < .3
                           for key in tasks)
            assert float(reward) == float(expected), 'Reward definition differs from the data'
            assert 'episode_task_completions' in info
            assert np.isfinite(flatten_observation(obs)).all()
            if terminated or truncated:
                assert step == 280 and truncated and not terminated
                break
        else:
            raise AssertionError('Environment did not stop at the required horizon')
    finally:
        env.close()
    report = {'status': 'PASS', 'platform': platform.platform(), 'python': sys.version,
              'versions': versions, 'version_mismatches': mismatch,
              'pinned_versions_enforced': check_versions, 'cuda': torch.version.cuda,
              'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
              'protocol_runs': len(protocol['runs']), 'dataset_checks': data_reports,
              'overlay_checks': overlay_reports,
              'local_environment': '280-step reward/schema/horizon PASS; engineering seed 12000'}
    atomic_json(ROOT / 'runs/audit/preflight.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-cpu', action='store_true', help='CPU diagnostic; training launcher still requires CUDA')
    parser.add_argument('--skip-version-check', action='store_true', help='Diagnostic only; records mismatches in report')
    args = parser.parse_args()
    preflight(require_cuda=not args.allow_cpu, check_versions=not args.skip_version_check)

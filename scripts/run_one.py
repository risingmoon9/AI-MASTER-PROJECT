"""Execute one immutable experiment run and its selected/endpoint evaluations."""
import argparse
import copy
import importlib.metadata
import json
import math
import os
import platform
import signal
import sys
import traceback

if __package__:
    from .experiment_common import (ROOT, atomic_json, canonical_hash, load_protocol, output_paths, project_path, run_lock, sha256, source_hashes, utc_now, verify_completed, verify_declared_data, verify_identity)
else:
    from experiment_common import (ROOT, atomic_json, canonical_hash, load_protocol, output_paths, project_path, run_lock, sha256, source_hashes, utc_now, verify_completed, verify_declared_data, verify_identity)


def config_for(row, protocol, smoke=False, smoke_updates=100, root=ROOT):
    """Load and validate a formal config, then derive isolated engineering settings."""
    import yaml
    cfg = copy.deepcopy(yaml.safe_load(project_path(row['config'], root).read_text(encoding='utf-8')))
    training = cfg['training']
    expected_algorithm = {'bc': 'bc', 'reference_iql': 'reference_iql'}.get(row['method'], 'iql')
    checks = {
        'seed': cfg['seed'] == row['seed'],
        'CUDA device': cfg['device'] == 'cuda',
        'algorithm': cfg['algorithm'] == expected_algorithm,
        'training budget': training['updates'] == protocol['updates'] == 200000,
        'batch size': training['batch_size'] == 256,
        'evaluation episode count': training['eval_episodes'] == row['validation_episodes'] == 20,
        'evaluation interval': training['eval_interval'] == 5000,
        'save interval': training['save_interval'] == (25000 if row['comparison'] == 'ablations' else 5000),
        'validation seed': training['validation_seed_start'] == row['validation_seed_start'],
        'overlay': cfg.get('data_overlay') == row.get('overlay'),
        'data protocol': cfg['data_protocol'] == protocol['data_protocol'],
        'experiment protocol': cfg['experiment_protocol'] == protocol['protocol_id'],
    }
    for field, passed in checks.items():
        if not passed:
            raise ValueError(f"Configuration/protocol mismatch for {row['name']}: {field}")
    output_paths(row, cfg, root=root)
    declared = json.loads((root / 'DATA_SHA256.json').read_text(encoding='utf-8'))
    if row['data'] not in declared:
        raise ValueError('Base dataset is not declared in DATA_SHA256.json')
    if row.get('overlay') and cfg.get('data_overlay_sha256') != declared.get(row['overlay']):
        raise ValueError('Overlay checksum differs from DATA_SHA256.json')
    if smoke:
        if smoke_updates not in (20, 100):
            raise ValueError('Smoke budget must be 20 or 100 updates')
        training.update(updates=smoke_updates, eval_interval=smoke_updates, eval_episodes=2,
                        validation_seed_start=12000, log_interval=smoke_updates,
                        save_interval=smoke_updates)
        cfg['paths'] = {'checkpoint_dir': 'runs/smoke/checkpoints', 'result_dir': 'runs/smoke/metrics'}
    return cfg


def make_identity(row, protocol, cfg, smoke=False):
    test_start, test_count = (12002, 2) if smoke else (row['test_seed_start'], row['test_episodes'])
    validation_start = cfg['training']['validation_seed_start']
    validation_seeds = list(range(validation_start, validation_start + cfg['training']['eval_episodes']))
    test_seeds = list(range(test_start, test_start + test_count))
    if set(validation_seeds) & set(test_seeds):
        raise ValueError('Validation/test seeds overlap')
    return {'protocol_id': protocol['protocol_id'], 'dataset': row['dataset'],
            'method': row['method'], 'seed': row['seed'], 'config': cfg,
            'dataset_path': row['data'], 'dataset_sha256': sha256(project_path(row['data'])),
            'overlay_path': row.get('overlay'),
            'overlay_sha256': sha256(project_path(row['overlay'])) if row.get('overlay') else None,
            'environment_data_sha256': verify_declared_data(), 'code_sha256': source_hashes(),
            'validation_seeds': validation_seeds, 'test_seeds': test_seeds, 'smoke_only': smoke}


def execute(row, protocol, args):
    # Every child receives thread limits before numerical libraries are imported.
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(args.threads)
    if __package__:
        from .runtime import initialize
    else:
        from runtime import initialize
    initialize()
    import torch
    import yaml
    from hqa_orl.train import build_agent, train_iql
    from hqa_orl.evaluation import evaluate_policy
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('Formal and smoke runs require working CUDA; run autodl/setup.sh first')
    cfg = config_for(row, protocol, args.smoke, args.smoke_updates)
    paths = output_paths(row, cfg, args.smoke)
    identity = make_identity(row, protocol, cfg, args.smoke)
    versions = {name: importlib.metadata.version(name) for name in
                ('numpy', 'torch', 'mujoco', 'gymnasium', 'gymnasium-robotics', 'minari', 'h5py')}
    hardware = {'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
                'python_minor': list(sys.version_info[:2]), 'versions': versions, 'threads': args.threads}
    manifest_path = paths['manifest']
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        verify_identity(manifest, identity)
        if manifest.get('status') == 'complete':
            verify_completed(manifest, paths)
            print(f'SKIP complete (hashes verified): {paths["name"]}', flush=True)
            return
        if args.no_resume:
            raise FileExistsError('Run already exists; omit --no-resume to continue it')
        if manifest['hardware'] != hardware:
            raise RuntimeError('Resume hardware/dependencies changed; restore original environment')
        if not paths['endpoint'].exists() and any(paths[key].exists() for key in ('best', 'metrics')):
            raise RuntimeError('Incomplete run has logs/best model but no resumable endpoint; preserve it and diagnose')
    else:
        if any(paths[key].exists() for key in ('endpoint', 'best', 'metrics', 'config')):
            raise FileExistsError('Orphan outputs found without a manifest; preserve them and diagnose')
        manifest = {'run': paths['name'], 'identity': identity, 'identity_sha256': canonical_hash(identity),
                    'hardware': hardware, 'platform': platform.platform(), 'python': sys.version,
                    'started_utc': utc_now(), 'status': 'started', 'execution_attempts': []}
        # Write identity before any output; a crash during config creation can be resumed.
        atomic_json(manifest_path, manifest)
    paths['config'].parent.mkdir(parents=True, exist_ok=True)
    paths['config'].write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    manifest['run_config_sha256'] = sha256(paths['config'])
    manifest.setdefault('execution_attempts', []).append({'started_utc': utc_now(), 'pid': os.getpid(),
                                                       'threads': args.threads,
                                                       'resumed_from_checkpoint': paths['endpoint'].exists()})
    manifest['status'] = 'training'
    atomic_json(manifest_path, manifest)
    print(f'{"RESUME" if paths["endpoint"].exists() else "START"}: {paths["name"]}', flush=True)
    try:
        train_iql(str(paths['config']), str(project_path(row['data'])),
                  'D4RL/kitchen/complete-v2', paths['name'], resume=paths['endpoint'].exists())
        manifest['status'] = 'final_evaluation'
        atomic_json(manifest_path, manifest)
        final = manifest.setdefault('final_evaluation', {})
        test_seeds = identity['test_seeds']
        for label in ('best', 'endpoint'):
            checksum = sha256(paths[label])
            if label in final and final[label]['checkpoint_sha256'] == checksum:
                continue
            # Only locally generated checkpoints guarded by the run identity are loaded.
            saved = torch.load(paths[label], map_location='cuda', weights_only=False)
            if saved['config'] != cfg:
                raise RuntimeError('Checkpoint configuration differs from the run identity')
            if label == 'endpoint' and (saved['step'] != cfg['training']['updates'] or not saved.get('training_complete')):
                raise RuntimeError('Endpoint checkpoint did not complete the prescribed budget')
            agent = build_agent(saved['obs_dim'], saved['action_dim'], saved['config'], torch.device('cuda'))
            agent.load_state_dict(saved['agent'])
            evaluation = evaluate_policy(agent.actor, 'D4RL/kitchen/complete-v2', saved['obs_rms'],
                                         episodes=len(test_seeds), seed=test_seeds[0], device=torch.device('cuda'),
                                         target_obs_dim=81)
            episodes = evaluation['per_episode']
            if [episode['env_seed'] for episode in episodes] != test_seeds:
                raise RuntimeError('Final evaluation seed list does not match protocol')
            if not all(math.isfinite(episode['return']) for episode in episodes):
                raise RuntimeError('Nonfinite final evaluation return')
            final[label] = {'checkpoint_step': saved['step'], 'checkpoint_sha256': checksum, 'metrics': evaluation}
            atomic_json(manifest_path, manifest)
            del agent, saved
            torch.cuda.empty_cache()
        manifest.update(status='complete', finished_utc=utc_now(), metrics_sha256=sha256(paths['metrics']))
        manifest.pop('last_error', None)
        atomic_json(manifest_path, manifest)
        print(f'COMPLETE: {paths["name"]}', flush=True)
    except BaseException:
        manifest['status'] = 'interrupted'
        manifest['last_error'] = traceback.format_exc()
        manifest['interrupted_utc'] = utc_now()
        atomic_json(manifest_path, manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-updates', type=int, choices=(20, 100), default=100)
    parser.add_argument('--no-resume', action='store_true')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    protocol = load_protocol()
    rows = [row for row in protocol['runs'] if row['name'] == args.run]
    if len(rows) != 1:
        parser.error('Unknown run; choose a name in protocol/experiments.json')
    def terminate(signum, frame):
        raise KeyboardInterrupt('Termination requested; last atomic checkpoint is retained')
    signal.signal(signal.SIGTERM, terminate)
    with run_lock(('smoke_' if args.smoke else '') + args.run):
        execute(rows[0], protocol, args)


if __name__ == '__main__':
    main()

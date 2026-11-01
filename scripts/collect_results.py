"""Export only the experiment evidence, including all model checkpoints."""
import argparse
from contextlib import ExitStack
import csv
import json
import os
from pathlib import Path
import tarfile
from datetime import datetime, timezone

if __package__:
    from .experiment_common import (ROOT, atomic_json, load_protocol, output_paths, project_path, run_lock, sha256, source_hashes, verify_completed, verify_declared_data, verify_identity)
else:
    from experiment_common import (ROOT, atomic_json, load_protocol, output_paths, project_path, run_lock, sha256, source_hashes, verify_completed, verify_declared_data, verify_identity)
if __package__:
    from .run_one import config_for
else:
    from run_one import config_for


def collect(include_smoke=False):
    protocol = load_protocol()
    # Hold every planned per-run lock throughout collection. This prevents a torn
    # snapshot without adding a directory-wide training lock.
    with ExitStack() as locks:
        for row in sorted(protocol['runs'], key=lambda item: item['name']):
            locks.enter_context(run_lock(row['name']))
            if include_smoke:
                locks.enter_context(run_lock('smoke_' + row['name']))
        declared = verify_declared_data()
        code = source_hashes()
        file_hashes = {}
        def cached_hash(relative):
            if relative not in file_hashes:
                file_hashes[relative] = sha256(project_path(relative))
            return file_hashes[relative]
        states, summary = [], []
        for smoke in ([False, True] if include_smoke else [False]):
            for row in protocol['runs']:
                name = ('smoke_' if smoke else '') + row['name']
                manifest_path = ROOT / ('runs/smoke/audit/runs' if smoke else 'runs/audit/runs') / (name + '.json')
                if not manifest_path.exists():
                    if not smoke:
                        states.append({'run': name, 'status': 'not_started', 'smoke_only': False})
                    continue
                manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                identity = manifest['identity']
                smoke_updates = identity['config']['training']['updates'] if smoke else 100
                cfg = config_for(row, protocol, smoke, smoke_updates)
                expected = dict(identity)
                validation_start = cfg['training']['validation_seed_start']
                test_start, test_count = (12002, 2) if smoke else (row['test_seed_start'], row['test_episodes'])
                expected.update(protocol_id=protocol['protocol_id'], dataset=row['dataset'],
                                method=row['method'], seed=row['seed'], config=cfg,
                                dataset_path=row['data'], dataset_sha256=cached_hash(row['data']),
                                overlay_path=row.get('overlay'),
                                overlay_sha256=cached_hash(row['overlay']) if row.get('overlay') else None,
                                environment_data_sha256=declared, code_sha256=code,
                                validation_seeds=list(range(validation_start, validation_start + cfg['training']['eval_episodes'])),
                                test_seeds=list(range(test_start, test_start + test_count)), smoke_only=smoke)
                verify_identity(manifest, expected)
                paths = output_paths(row, cfg, smoke)
                states.append({'run': name, 'status': manifest['status'], 'smoke_only': smoke})
                if manifest['status'] != 'complete':
                    continue
                verify_completed(manifest, paths)
                for label in ('best', 'endpoint'):
                    entry = manifest['final_evaluation'][label]
                    metrics = entry['metrics']
                    summary.append({'run': name, 'comparison': row['comparison'], 'dataset': row['dataset'], 'method': row['method'],
                                    'training_seed': row['seed'], 'smoke_only': smoke, 'checkpoint': label,
                                    'step': entry['checkpoint_step'], 'return_mean': metrics['return_mean'],
                                    'within_policy_episode_return_std': metrics['return_std'],
                                    'task_completion_fraction_mean': metrics['task_completion_fraction_mean'],
                                    'all_tasks_success_rate': metrics['all_tasks_success_rate'],
                                    'n_test_episodes': len(metrics['per_episode'])})
        destination = ROOT / 'runs/exports'
        destination.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        export_dir = destination / ('evidence_' + stamp)
        export_dir.mkdir()
        fields = ['run', 'comparison', 'dataset', 'method', 'training_seed', 'smoke_only', 'checkpoint', 'step',
                  'return_mean', 'within_policy_episode_return_std', 'task_completion_fraction_mean',
                  'all_tasks_success_rate', 'n_test_episodes']
        with (export_dir / 'run_summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary)
        atomic_json(export_dir / 'run_status.json', states)
        included = ['runs/audit', 'runs/configs', 'runs/metrics',
                    'runs/checkpoints', 'runs/logs', 'hqa_orl', 'scripts', 'tests',
                    'configs/experiments', 'autodl', 'protocol', 'reference', 'docs', 'requirements.txt', 'requirements-dev.txt',
                    'README.md', 'README_CN.md', 'DATA_SHA256.json', 'pyproject.toml', 'LICENSE', 'THIRD_PARTY_NOTICES.md']
        if include_smoke:
            included.append('runs/smoke')
        files = []
        for relative in included:
            path = ROOT / relative
            candidates = path.rglob('*') if path.is_dir() else [path]
            for item in candidates:
                if item.is_file() and not item.is_symlink() and '__pycache__' not in item.parts \
                        and item.suffix not in ('.tmp', '.pyc', '.lock', '.npz', '.hdf5'):
                    files.append(item)
        files.extend([export_dir / 'run_summary.csv', export_dir / 'run_status.json'])
        files = sorted(set(files))
        inventory = {'protocol_id': protocol['protocol_id'], 'source_code_sha256': code,
                     'verified_data_sha256': declared, 'verified_overlay_sha256': file_hashes,
                     'includes_checkpoints': True, 'includes_smoke': include_smoke,
                     'files_sha256': {path.relative_to(ROOT).as_posix(): sha256(path) for path in files}}
        atomic_json(export_dir / 'export_inventory.json', inventory)
        files.append(export_dir / 'export_inventory.json')
        archive = destination / ('HQA_IQL_results_' + stamp + '.tar.gz')
        temporary = archive.with_suffix(archive.suffix + '.tmp')
        try:
            with tarfile.open(temporary, 'w:gz') as bundle:
                for path in files:
                    bundle.add(path, arcname='HQA_IQL_results/' + path.relative_to(ROOT).as_posix(), recursive=False)
            os.replace(temporary, archive)
        finally:
            if temporary.exists():
                temporary.unlink()
        result = {'archive': str(archive), 'bytes': archive.stat().st_size, 'sha256': sha256(archive),
                  'completed_formal_runs': sum(item['status'] == 'complete' and not item['smoke_only'] for item in states),
                  'planned_formal_runs': len(protocol['runs']),
                  'checkpoint_files': sum(path.suffix == '.pt' for path in files),
                  'unfinished_formal_runs': [item for item in states if item['status'] != 'complete' and not item['smoke_only']]}
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-smoke', action='store_true', help='Include isolated smoke evidence for installation diagnosis')
    args = parser.parse_args()
    collect(args.include_smoke)


if __name__ == '__main__':
    main()

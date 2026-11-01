"""GPU-free process/integrity tests; these exercise real OS locks and subprocesses."""
from argparse import Namespace
from contextlib import redirect_stdout
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import tarfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import experiment_common as common
import run_experiments as runner


class IntegrityTests(unittest.TestCase):
    def test_identity_rejects_changed_config_code_or_data(self):
        identity = {'config': {'training': {'updates': 200000}}, 'dataset_sha256': 'original',
                    'code_sha256': {'hqa_orl/train.py': 'original'}}
        manifest = {'identity': copy.deepcopy(identity), 'identity_sha256': common.canonical_hash(identity)}
        common.verify_identity(manifest, identity)
        for field in ('config', 'dataset_sha256', 'code_sha256'):
            changed = copy.deepcopy(identity)
            changed[field] = 'modified'
            with self.assertRaises(RuntimeError):
                common.verify_identity(manifest, changed)
        manifest['identity_sha256'] = 'tampered'
        with self.assertRaises(RuntimeError):
            common.verify_identity(manifest, identity)

    def test_completed_artifacts_must_all_match(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = {key: Path(directory) / key for key in ('best', 'endpoint', 'metrics', 'config')}
            for key, path in paths.items():
                path.write_text(key)
            manifest = {'status': 'complete', 'final_evaluation': {
                label: {'checkpoint_sha256': common.sha256(paths[label])} for label in ('best', 'endpoint')},
                'metrics_sha256': common.sha256(paths['metrics']), 'run_config_sha256': common.sha256(paths['config'])}
            common.verify_completed(manifest, paths)
            for key, path in paths.items():
                path.write_text('corrupted')
                with self.assertRaises(RuntimeError):
                    common.verify_completed(manifest, paths)
                path.write_text(key)

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in ('../other/file', '/tmp/file', 'folder\\file'):
                with self.assertRaises(ValueError):
                    common.project_path(relative, root)

    def test_filter_and_smoke_one_seed_per_method_dataset(self):
        protocol = {'runs': [dict(name=f'{dataset}_{method}_{seed}', dataset=dataset, method=method,
                                 seed=seed, comparison='comparisons', presets=['comparisons'])
                             for dataset in ('low', 'complete') for method in ('iql', 'bc') for seed in (3, 4, 5)]}
        selected = common.select_runs(protocol, smoke=True)
        self.assertEqual(len(selected), 4)
        self.assertEqual({row['seed'] for row in selected}, {3})
        selected = common.select_runs(protocol, seeds=[4], methods=['bc'], datasets=['low'])
        self.assertEqual([row['name'] for row in selected], ['low_bc_4'])


class ProtocolTests(unittest.TestCase):
    def test_recorded_design_and_configurations(self):
        protocol = common.validate_protocol()
        self.assertEqual(len(protocol['runs']), 60)
        for comparison, count, seeds, validation_start, test_start in (
                ('comparisons', 42, {3, 4, 5}, 11000, 30000),
                ('ablations', 18, {0, 1, 2}, 10000, 20000)):
            rows = common.select_runs(protocol, comparison)
            self.assertEqual(len(rows), count)
            self.assertEqual({row['seed'] for row in rows}, seeds)
            self.assertEqual({row['validation_seed_start'] for row in rows}, {validation_start})
            self.assertEqual({row['test_seed_start'] for row in rows}, {test_start})
            self.assertEqual({row['test_episodes'] for row in rows}, {50})
        from run_one import config_for
        row = protocol['runs'][0]
        formal = config_for(row, protocol)
        smoke = config_for(row, protocol, smoke=True, smoke_updates=20)
        self.assertEqual(formal['training']['updates'], 200000)
        self.assertEqual(smoke['training']['updates'], 20)
        self.assertEqual(smoke['training']['validation_seed_start'], 12000)
        self.assertNotEqual(common.output_paths(row, formal)['endpoint'],
                            common.output_paths(row, smoke, smoke=True)['endpoint'])

    def test_help_and_dry_run_without_site_packages_or_data(self):
        for script in ('run_experiments', 'run_one', 'collect_results', 'status'):
            child = subprocess.run([sys.executable, '-S', str(ROOT / 'scripts' / (script + '.py')), '--help'],
                                   cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(child.returncode, 0, child.stderr)
        child = subprocess.run([sys.executable, '-S', str(ROOT / 'scripts/run_experiments.py'),
                                '--preset', 'all', '--dry-run'], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertEqual(json.loads(child.stdout)['run_count'], 60)

    def test_package_import_and_module_help(self):
        child = subprocess.run([sys.executable, '-S', '-m', 'scripts.run_experiments', '--help'],
                               cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(child.returncode, 0, child.stderr)
        child = subprocess.run([sys.executable, '-S', '-m', 'scripts.collect_results', '--help'],
                               cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(child.returncode, 0, child.stderr)


class LockTests(unittest.TestCase):
    def test_same_run_conflict_distinct_runs_and_crash_release(self):
        with tempfile.TemporaryDirectory() as directory:
            code = ('import sys,time; from pathlib import Path; '
                    f'sys.path.insert(0,{str(ROOT / "scripts")!r}); '
                    'from experiment_common import run_lock; '
                    f'context=run_lock("same",Path({directory!r})); '
                    'context.__enter__(); print("locked",flush=True); time.sleep(60)')
            child = subprocess.Popen([sys.executable, '-u', '-c', code], stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True)
            try:
                self.assertEqual(child.stdout.readline().strip(), 'locked')
                with self.assertRaises(RuntimeError):
                    with common.run_lock('same', Path(directory)):
                        pass
                with common.run_lock('different', Path(directory)):
                    pass
            finally:
                child.kill()
                child.communicate(timeout=10)
            with common.run_lock('same', Path(directory)):
                pass


class SchedulerTests(unittest.TestCase):
    def test_concurrent_children_failure_does_not_cancel_other_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / 'fake_worker.py'
            worker.write_text('''import argparse,json,os,time
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--run'); parser.add_argument('--threads')
args=parser.parse_args()
start=time.monotonic()
time.sleep(0.45)
Path(args.run+'.json').write_text(json.dumps({'pid':os.getpid(),'start':start,'end':time.monotonic(),
    'omp':os.environ['OMP_NUM_THREADS'],'seed':os.environ['PYTHONHASHSEED']}))
raise SystemExit(7 if args.run=='fail' else 0)
''', encoding='utf-8')
            rows = [{'name': name, 'seed': index + 3} for index, name in enumerate(('first', 'fail', 'third', 'fourth'))]
            args = Namespace(smoke=False, preset='all', jobs=2, threads=1, smoke_updates=100, no_resume=False)
            with patch.object(runner, 'ROOT', root), redirect_stdout(io.StringIO()):
                code = runner.schedule(rows, args, worker_script=worker)
            self.assertEqual(code, 1)
            records = [json.loads((root / (row['name'] + '.json')).read_text()) for row in rows]
            self.assertEqual(len({row['pid'] for row in records}), 4)
            self.assertEqual([row['seed'] for row in records], ['3', '4', '5', '6'])
            self.assertTrue(all(row['omp'] == '1' for row in records))
            events = sorted([(row['start'], 1) for row in records] + [(row['end'], -1) for row in records])
            active, peak = 0, 0
            for _, change in events:
                active += change
                peak = max(peak, active)
            self.assertEqual(peak, 2)
            session = json.loads(next((root / 'runs/audit/sessions').glob('*.json')).read_text())
            self.assertEqual(session['status'], 'failed')
            self.assertEqual(len(session['finished_runs']), 4)
            self.assertFalse(session['not_started'])


class CollectorTests(unittest.TestCase):
    def test_export_contains_both_models_and_no_old_results_or_large_data(self):
        import collect_results as collector
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = dict(name='comparisons_low_s3', dataset='low', method='baseline', seed=3, comparison='comparisons', validation_seed_start=11000, validation_episodes=20, test_seed_start=30000, test_episodes=50,
                       data='data_repaired/data.npz', overlay=None)
            protocol = {'protocol_id': 'test', 'validation_seed_start': 11000,
                        'test_seed_start': 30000, 'test_episodes': 50, 'runs': [row]}
            cfg = {'paths': {'checkpoint_dir': 'runs/checkpoints', 'result_dir': 'runs/metrics'},
                   'training': {'validation_seed_start': 11000, 'eval_episodes': 20, 'updates': 200000}}
            paths = common.output_paths(row, cfg, root=root)
            for key in ('best', 'endpoint', 'metrics', 'config'):
                paths[key].parent.mkdir(parents=True, exist_ok=True)
                paths[key].write_text(key)
            for relative in ('data_repaired/data.npz', 'minari_datasets/raw.hdf5',
                             'checkpoints_repaired/old.pt', 'results_repaired/old.jsonl'):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('should not be exported')
            code = {'hqa_orl/train.py': 'sourcehash'}
            identity = dict(protocol_id='test', dataset='low', method='baseline', seed=3, config=cfg,
                            dataset_path=row['data'], dataset_sha256=common.sha256(root / row['data']),
                            overlay_path=None, overlay_sha256=None, environment_data_sha256={},
                            code_sha256=code, validation_seeds=list(range(11000, 11020)),
                            test_seeds=list(range(30000, 30050)), smoke_only=False)
            metrics = dict(return_mean=0, return_std=0, task_completion_fraction_mean=0,
                           all_tasks_success_rate=0, per_episode=[{'env_seed': seed} for seed in range(30000, 30050)])
            manifest = {'run': row['name'], 'identity': identity, 'identity_sha256': common.canonical_hash(identity),
                        'status': 'complete', 'metrics_sha256': common.sha256(paths['metrics']),
                        'run_config_sha256': common.sha256(paths['config']), 'final_evaluation': {
                            label: {'checkpoint_step': 200000, 'checkpoint_sha256': common.sha256(paths[label]),
                                    'metrics': metrics} for label in ('best', 'endpoint')}}
            common.atomic_json(paths['manifest'], manifest)
            with patch.object(collector, 'ROOT', root), \
                    patch.object(collector, 'load_protocol', lambda: protocol), \
                    patch.object(collector, 'run_lock', lambda name: common.run_lock(name, root)), \
                    patch.object(collector, 'verify_declared_data', lambda: {}), \
                    patch.object(collector, 'source_hashes', lambda: code), \
                    patch.object(collector, 'project_path', lambda relative: common.project_path(relative, root)), \
                    patch.object(collector, 'config_for', lambda *args: cfg), \
                    patch.object(collector, 'output_paths', lambda row, cfg, smoke: common.output_paths(row, cfg, smoke, root)), \
                    redirect_stdout(io.StringIO()):
                result = collector.collect()
            self.assertEqual(result['completed_formal_runs'], 1)
            self.assertEqual(result['checkpoint_files'], 2)
            with tarfile.open(result['archive'], 'r:gz') as bundle:
                names = bundle.getnames()
            self.assertEqual(sum(name.endswith('.pt') for name in names), 2)
            self.assertTrue(any('run_summary.csv' in name for name in names))
            self.assertTrue(any('export_inventory.json' in name for name in names))
            self.assertFalse(any('checkpoints_repaired' in name or 'results_repaired' in name for name in names))
            self.assertFalse(any(name.endswith(('.npz', '.hdf5', '.lock')) for name in names))


if __name__ == '__main__':
    unittest.main()

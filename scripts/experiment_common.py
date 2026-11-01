"""Shared protocol, integrity and per-run locking helpers (stdlib only)."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PRESETS = ('comparisons', 'ablations', 'all')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def project_path(relative, root=ROOT):
    path = Path(relative)
    if path.is_absolute() or '\\' in str(relative) or '..' in path.parts:
        raise ValueError(f'Expected a project-relative path: {relative}')
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f'Path leaves project root: {relative}')
    return resolved


def load_protocol(root=ROOT):
    """Read safe metadata using only the standard library (also used by dry-run)."""
    protocol = json.loads((root / 'protocol/experiments.json').read_text(encoding='utf-8'))
    names, settings = set(), set()
    for row in protocol['runs']:
        name = row['name']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name in names:
            raise ValueError(f'Unsafe or duplicate run name: {name}')
        names.add(name)
        setting = (row['comparison'], row['dataset'], row['method'], row['seed'])
        if setting in settings:
            raise ValueError(f'Duplicate experiment setting: {setting}')
        settings.add(setting)
        if row['comparison'] not in ('comparisons', 'ablations') or row['dataset'] not in ('mixture', 'complete'):
            raise ValueError(f'Unknown comparison or dataset for {name}')
        if row['comparison'] not in row['presets'] or 'all' not in row['presets']:
            raise ValueError(f'Preset membership inconsistent for {name}')
        for field in ('config', 'data', 'overlay'):
            if row.get(field):
                project_path(row[field], root)
        validation = set(range(row['validation_seed_start'], row['validation_seed_start'] + row['validation_episodes']))
        test = set(range(row['test_seed_start'], row['test_seed_start'] + row['test_episodes']))
        if len(validation) != 20 or len(test) != 50 or validation & test:
            raise ValueError(f'Invalid validation/test partition for {name}')
    if not names:
        raise ValueError('Protocol contains no runs')
    return protocol


def validate_protocol(root=ROOT):
    """Validate all saved configurations without loading data, torch, or MuJoCo.

    Returns the loaded protocol. Requires PyYAML; CLI help and dry-run do not.
    """
    if __package__:
        from .run_one import config_for
    else:
        from run_one import config_for
    protocol = load_protocol(root)
    for row in protocol['runs']:
        config_for(row, protocol, root=root)
    return protocol


def select_runs(protocol, preset='comparisons', seeds=None, methods=None, datasets=None, smoke=False):
    rows = [row for row in protocol['runs'] if preset == 'all' or preset in row['presets']]
    rows = [row for row in rows if (seeds is None or row['seed'] in seeds)
            and (methods is None or row['method'] in methods)
            and (datasets is None or row['dataset'] in datasets)]
    if smoke:
        seen = set()
        selected = []
        for row in sorted(rows, key=lambda item: (item['comparison'], item['dataset'], item['method'], item['seed'])):
            key = (row['comparison'], row['dataset'], row['method'])
            if key not in seen:
                seen.add(key)
                selected.append(row)
        rows = selected
    return rows


def output_paths(row, cfg, smoke=False, root=ROOT):
    name = ('smoke_' if smoke else '') + row['name']
    expected = {'checkpoint_dir': 'runs/smoke/checkpoints' if smoke else 'runs/checkpoints',
                'result_dir': 'runs/smoke/metrics' if smoke else 'runs/metrics'}
    if cfg['paths'] != expected:
        raise ValueError(f'Output paths must be the declared experiment directories: {expected}')
    return {'name': name,
            'manifest': root / ('runs/smoke/audit/runs' if smoke else 'runs/audit/runs') / (name + '.json'),
            'config': root / ('runs/smoke/configs' if smoke else 'runs/configs') / (name + '.yaml'),
            'endpoint': root / expected['checkpoint_dir'] / (name + '.pt'),
            'best': root / expected['checkpoint_dir'] / (name + '_best.pt'),
            'metrics': root / expected['result_dir'] / (name + '.jsonl')}


@contextmanager
def run_lock(name, root=ROOT):
    """Nonblocking OS lock; separate runs never share a lock and crashes release it."""
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('Unsafe run lock name')
    directory = root / 'runs/audit/locks'
    directory.mkdir(parents=True, exist_ok=True)
    handle = (directory / (name + '.lock')).open('a+b')
    try:
        if os.name == 'nt':
            import msvcrt
            handle.seek(0)
            if not handle.read(1):
                handle.write(b'0')
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        handle.close()
        raise RuntimeError(f'Run is already active (or being collected): {name}') from error
    try:
        yield
    finally:
        handle.close()


def source_hashes(root=ROOT):
    result = {}
    for folder in ('hqa_orl', 'scripts', 'reference', 'autodl', 'protocol', 'configs/experiments'):
        for path in sorted((root / folder).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc', '.tmp'):
                result[path.relative_to(root).as_posix()] = sha256(path)
    for name in ('requirements.txt', 'requirements-dev.txt', 'pyproject.toml', 'DATA_SHA256.json'):
        if (root / name).is_file():
            result[name] = sha256(root / name)
    return result


def verify_declared_data(root=ROOT):
    declared = json.loads((root / 'DATA_SHA256.json').read_text(encoding='utf-8'))
    for relative, expected in declared.items():
        path = project_path(relative, root)
        # The runtime accepts an explicitly configured Minari cache. Hash the
        # actual cache used by the environment, not an unused local copy.
        if relative.startswith('minari_datasets/') and os.environ.get('MINARI_DATASETS_PATH'):
            path = Path(os.environ['MINARI_DATASETS_PATH']) / relative.removeprefix('minari_datasets/')
        if sha256(path) != expected:
            raise ValueError(f'Declared data checksum mismatch: {relative}')
    return declared


def verify_identity(manifest, identity):
    expected = canonical_hash(identity)
    if manifest.get('identity') != identity or manifest.get('identity_sha256') != expected:
        raise RuntimeError('Run identity changed: restore original code/config/data or use a new project copy')


def verify_completed(manifest, paths):
    if manifest.get('status') != 'complete':
        raise ValueError('Run is not complete')
    for label in ('best', 'endpoint'):
        entry = manifest['final_evaluation'][label]
        if not paths[label].is_file() or sha256(paths[label]) != entry['checkpoint_sha256']:
            raise RuntimeError(f'Completed checkpoint missing or changed: {paths[label]}')
    if not paths['metrics'].is_file() or sha256(paths['metrics']) != manifest['metrics_sha256']:
        raise RuntimeError(f'Completed metrics log missing or changed: {paths["metrics"]}')
    if not paths['config'].is_file() or sha256(paths['config']) != manifest['run_config_sha256']:
        raise RuntimeError(f'Completed runtime config missing or changed: {paths["config"]}')

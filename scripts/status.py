"""Show experiment completion and latest recorded update without importing torch."""
import argparse
import json

if __package__:
    from .experiment_common import (ROOT, PRESETS, load_protocol, select_runs)
else:
    from experiment_common import (ROOT, PRESETS, load_protocol, select_runs)


def last_step(path):
    if not path.exists():
        return '-'
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 128 * 1024))
        tail = stream.read().decode('utf-8', errors='replace')
    for line in reversed(tail.splitlines()):
        try:
            return json.loads(line).get('step', '-')
        except (ValueError, AttributeError):
            continue
    return '-'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', choices=PRESETS, default='all')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    counts = {}
    base = ROOT / ('runs/smoke' if args.smoke else 'runs')
    for row in select_runs(load_protocol(), preset=args.preset):
        name = ('smoke_' if args.smoke else '') + row['name']
        path = base / 'audit/runs' / (name + '.json')
        if args.smoke and not path.exists():
            continue
        manifest = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        status = manifest.get('status', 'not_started')
        counts[status] = counts.get(status, 0) + 1
        step = last_step(base / 'metrics' / (name + '.jsonl'))
        print(f'{name:65s} {status:18s} step={step}')
    print(json.dumps(counts, indent=2))
    print('Status is the last recorded state; an abrupt kill may leave "training" until the next resume.')


if __name__ == '__main__':
    main()

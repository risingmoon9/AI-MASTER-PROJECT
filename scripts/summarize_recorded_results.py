"""Recompute study tables from the committed per-model numerical records (stdlib)."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('return_mean', 'task_completion_percent', 'all_tasks_success_percent')


def summarize(source):
    groups = defaultdict(list)
    seen = set()
    with Path(source).open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            key = tuple(row[k] for k in ('comparison', 'dataset', 'method', 'checkpoint'))
            identity = key + (int(row['seed']),)
            if identity in seen:
                raise ValueError(f'Duplicate model/checkpoint: {identity}')
            seen.add(identity)
            if int(row['episode_count']) != 50:
                raise ValueError('Expected 50 final evaluation episodes per model')
            groups[key].append(row)
    output = []
    for key, rows in sorted(groups.items()):
        expected = {3, 4, 5} if key[0] == 'comparisons' else {0, 1, 2}
        if {int(row['seed']) for row in rows} != expected:
            raise ValueError(f'Incomplete training-seed group: {key}')
        entry = dict(zip(('comparison', 'dataset', 'method', 'checkpoint'), key))
        entry.update(n_training_seeds=len(rows), n_episodes_per_seed=50)
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            entry[metric + '_mean'] = mean(values)
            entry[metric + '_sample_sd'] = stdev(values)
        output.append(entry)
    if not output:
        raise ValueError('No recorded results found')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'results/recorded_runs.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/exports/recorded_summary.csv')
    args = parser.parse_args()
    rows = summarize(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(f'{len(rows)} groups written to {args.output}')


if __name__ == '__main__':
    main()

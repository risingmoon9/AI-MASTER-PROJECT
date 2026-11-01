"""Schedule independent experiment processes with per-run locks and bounded concurrency."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import uuid

if __package__:
    from .experiment_common import (ROOT, PRESETS, atomic_json, load_protocol, select_runs, utc_now)
else:
    from experiment_common import (ROOT, PRESETS, atomic_json, load_protocol, select_runs, utc_now)


def schedule(rows, args, worker_script=None):
    session_id = utc_now().replace(':', '').replace('-', '') + '_' + uuid.uuid4().hex[:8]
    session_dir = ROOT / ('runs/smoke/sessions' if args.smoke else 'runs/audit/sessions')
    log_dir = ROOT / ('runs/smoke/logs' if args.smoke else 'runs/logs') / session_id
    log_dir.mkdir(parents=True, exist_ok=False)
    session = {'session_id': session_id, 'started_utc': utc_now(), 'status': 'running',
               'preset': args.preset, 'smoke_only': args.smoke, 'jobs': args.jobs, 'threads': args.threads,
               'planned_runs': [row['name'] for row in rows], 'finished_runs': []}
    session_path = session_dir / (session_id + '.json')
    atomic_json(session_path, session)
    active = {}
    queue = list(rows)
    stopping = False
    stop_started = None
    def request_stop(signum, frame):
        nonlocal stopping, stop_started
        if not stopping:
            stopping, stop_started = True, time.monotonic()
            print('Stopping launch queue; terminating children and retaining atomic checkpoints.', flush=True)
            for process, stream, row in active.values():
                process.terminate()
    previous = {number: signal.signal(number, request_stop) for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        while queue or active:
            while queue and len(active) < args.jobs and not stopping:
                row = queue.pop(0)
                log_path = log_dir / (row['name'] + '.log')
                stream = log_path.open('wb')
                command = [sys.executable, '-u', str(worker_script or ROOT / 'scripts/run_one.py'),
                           '--run', row['name'], '--threads', str(args.threads)]
                if args.smoke:
                    command.extend(['--smoke', '--smoke-updates', str(args.smoke_updates)])
                if args.no_resume:
                    command.append('--no-resume')
                environment = os.environ.copy()
                environment.update({name: str(args.threads) for name in
                                    ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
                environment['PYTHONUNBUFFERED'] = '1'
                environment['PYTHONHASHSEED'] = str(row['seed'])
                try:
                    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream,
                                               stderr=subprocess.STDOUT, start_new_session=os.name != 'nt')
                except BaseException:
                    stream.close()
                    raise
                active[process.pid] = (process, stream, row)
                print(f'START {row["name"]} pid={process.pid} log={log_path.relative_to(ROOT)}', flush=True)
            for pid, (process, stream, row) in list(active.items()):
                code = process.poll()
                if code is None:
                    if stopping and time.monotonic() - stop_started > 30:
                        process.kill()
                    continue
                stream.close()
                del active[pid]
                session['finished_runs'].append({'run': row['name'], 'exit_code': code, 'finished_utc': utc_now()})
                atomic_json(session_path, session)
                print(f'{"DONE" if code == 0 else "FAILED"} {row["name"]} exit={code}', flush=True)
            if stopping and not active:
                break
            if active:
                time.sleep(0.25)
        failed = [entry for entry in session['finished_runs'] if entry['exit_code'] != 0]
        session.update(status='interrupted' if stopping else ('failed' if failed else 'complete'),
                       finished_utc=utc_now(), not_started=[row['name'] for row in queue])
        atomic_json(session_path, session)
        print(json.dumps({'status': session['status'], 'finished': len(session['finished_runs']),
                          'failed': failed, 'not_started': session['not_started'],
                          'session': str(session_path)}, indent=2), flush=True)
        return 130 if stopping else (1 if failed else 0)
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
        for process, stream, row in active.values():
            if process.poll() is None:
                process.terminate()
        deadline = time.monotonic() + 30
        for process, stream, row in active.values():
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', choices=PRESETS, default='comparisons')
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--methods', nargs='+')
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--jobs', type=int, default=6)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-updates', type=int, choices=(20, 100), default=100)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--no-resume', action='store_true')
    args = parser.parse_args()
    if args.jobs < 1 or args.threads < 1:
        parser.error('--jobs and --threads must be positive')
    protocol = load_protocol()
    rows = select_runs(protocol, args.preset, args.seeds, args.methods, args.datasets, args.smoke)
    if not rows:
        parser.error('No runs match the selected preset/filters')
    if args.dry_run:
        print(json.dumps({'protocol_id': protocol['protocol_id'], 'preset': args.preset,
                          'smoke_only': args.smoke, 'jobs': args.jobs, 'threads': args.threads,
                          'updates_each': args.smoke_updates if args.smoke else 200000,
                          'evaluation_blocks': ({'smoke': {'validation_seeds': [12000, 12001],
                                                       'test_seeds': [12002, 12003]}} if args.smoke else
                              {row['comparison']: {'validation_seeds': [row['validation_seed_start'],
                                  row['validation_seed_start'] + row['validation_episodes'] - 1],
                                  'test_seeds': [row['test_seed_start'],
                                  row['test_seed_start'] + row['test_episodes'] - 1]} for row in rows}),
                          'run_count': len(rows), 'runs': rows}, indent=2, ensure_ascii=False))
        return 0
    return schedule(rows, args)


if __name__ == '__main__':
    sys.exit(main())

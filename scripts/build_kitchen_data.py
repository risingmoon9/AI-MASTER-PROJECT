"""Reconstruct common-task buffers from local Minari Kitchen HDF5 files.

Writes to data/rebuilt, never over frozen study buffers. The committed selection
preserves episode order and multiplicity. Use import_data.py for exact study bytes.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TASKS = ["kettle", "light switch", "microwave", "slide cabinet"]
GOALS = {"kettle": [-.23, .75, 1.62, .99, 0., 0., -.06],
         "light switch": [-.69, -.05], "microwave": [-.75], "slide cabinet": [.37]}
PRIORS = {"complete": 1.0, "partial": .55, "mixed": .30}


def physical_identity(physical, actions):
    h = hashlib.sha256()
    h.update(np.asarray(physical, dtype=np.float32).tobytes(order="C"))
    h.update(np.asarray(actions, dtype=np.float32).tobytes(order="C"))
    return h.hexdigest()


def canonical_observations(episode, peers):
    """Recover missing goal channels only from byte-identical physical peers.

    The mixed source omits slide-cabinet achieved goals. Its physical observation
    is noisy and must not be substituted for the clean achieved-goal channel.
    Matching partial trajectories supply the channel with exact physical/action
    identity, rather than guessed coordinate padding.
    """
    physical = np.asarray(episode["observations/observation"], np.float32)
    achieved = []
    for task in TASKS:
        path = "observations/achieved_goal/" + task
        candidates = [np.asarray(peer[path], np.float32) for peer in peers if path in peer]
        if not candidates:
            raise ValueError(f"No matching trajectory provides achieved goal {task!r}")
        for other in candidates[1:]:
            np.testing.assert_array_equal(candidates[0], other)
        achieved.append(candidates[0])
    desired = np.broadcast_to(np.concatenate([GOALS[t] for t in TASKS]).astype(np.float32), (len(physical), 11))
    return np.concatenate([*achieved, desired, physical], axis=1).astype(np.float32)


def record_arrays(observations, actions, truncations, local_id, source):
    n = len(actions)
    if observations.shape != (n+1, 81) or actions.shape != (n, 9):
        raise ValueError("Invalid source trajectory dimensions")
    nxt = observations[1:]
    reward = sum((np.linalg.norm(nxt[:, lo:hi] - nxt[:, 11+lo:11+hi], axis=1) < .3).astype(np.float32)
                 for lo, hi in [(0, 7), (7, 9), (9, 10), (10, 11)])
    rtg = np.empty(n, dtype=np.float32)
    acc = 0.0
    for i in range(n-1, -1, -1):
        acc = float(reward[i]) + .99 * acc
        rtg[i] = acc
    column = lambda value, dtype=np.float32: np.full((n, 1), value, dtype=dtype)
    return {"obs": observations[:-1], "actions": actions.astype(np.float32), "rewards": reward[:, None],
            "next_obs": nxt, "dones": column(0), "truncations": np.asarray(truncations, np.float32).reshape(n, 1),
            "returns_to_go": rtg[:, None], "episode_id": column(local_id, np.int32),
            "timestep": np.arange(n, dtype=np.int32)[:, None], "episode_return": column(float(reward.sum())),
            "episode_length": column(n), "source_prior": column(PRIORS[source]),
            "source_name": np.asarray([source]*n), "source_dataset_id": np.asarray([f"D4RL/kitchen/{source}-v2"]*n)}


def build(minari_root, selection, output, with_quality=False, threads=1):
    output.mkdir(parents=True, exist_ok=True)
    needed = {row["physical_sha256"] for data in selection["datasets"].values() for row in data["episode_rows"]}
    peer_index = {}
    report = {}
    with ExitStack() as stack:
        sources = {name: stack.enter_context(h5py.File(minari_root / f"D4RL/kitchen/{name}-v2/data/main_data.hdf5", "r")) for name in PRIORS}
        for name, source in sources.items():
            for key in source:
                ep = source[key]
                identity = physical_identity(ep["observations/observation"], ep["actions"])
                if identity in needed:
                    peer_index.setdefault(identity, []).append((name, key))
        for dataset, info in selection["datasets"].items():
            records = []
            for row in info["episode_rows"]:
                identity = row["physical_sha256"]
                candidates = peer_index.get(identity, [])
                own = [(source, key) for source, key in candidates if source == row["source"] and key in row["candidate_original_episode_ids"]]
                if not own:
                    raise ValueError(f"Cannot verify selected episode {dataset}:{row['local_episode_id']}")
                source, key = own[0]
                ep = sources[source][key]
                if len(ep["actions"]) != row["length"]:
                    raise ValueError("Selected episode length mismatch")
                # Retain ambiguous original IDs in the selection manifest; do
                # not claim one historical source ID merely because it matched.
                peers = [sources[src][candidate] for src, candidate in candidates]
                obs = canonical_observations(ep, peers)
                records.append(record_arrays(obs, np.asarray(ep["actions"], np.float32), ep["truncations"], row["local_episode_id"], source))
            data = {key: np.concatenate([record[key] for record in records], axis=0) for key in records[0]}
            if len(data["obs"]) != info["transitions"]:
                raise ValueError("Selected transition total mismatch")
            metadata = {"schema": "kitchen_common81_persistent_v1", "tasks": TASKS, "goal_order": TASKS,
                        "obs_dim": 81, "reward": "sum of next-state goal distances < 0.3; repeated each step",
                        "remove_task_when_completed": False, "terminate_on_tasks_completed": False,
                        "discount": .99, "sampling": "original episode order and multiplicity preserved",
                        "quality_status": "recomputed" if with_quality else "not_computed"}
            if with_quality:
                from hqa_orl.quality import compute_quality_weights
                from threadpoolctl import threadpool_limits
                with threadpool_limits(limits=threads):
                    data.update(compute_quality_weights(data, n_clusters=5, seed=0, use_knn_support=True, max_knn_points=100000))
                metadata["quality"] = {"n_clusters": 5, "seed": 0, "use_knn_support": True, "max_knn_points": 100000, "k": 10}
            data["metadata"] = np.asarray(json.dumps(metadata))
            target = output / f"kitchen_{dataset}{'_quality' if with_quality else ''}.npz"
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite existing reconstruction: {target}")
            np.savez_compressed(target, **data)
            from scripts.validate_data import validate
            report[dataset] = validate(target)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minari-root", required=True, type=Path, help="Directory containing D4RL/kitchen/{complete,mixed,partial}-v2")
    parser.add_argument("--selection", type=Path, default=ROOT / "protocol/kitchen_episode_selection.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/rebuilt")
    parser.add_argument("--with-quality", action="store_true", help="Also recompute KMeans and kNN quality; this can take substantial CPU time")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.output_dir.resolve() == (ROOT / "data/processed").resolve():
        parser.error("Reconstruction must use a separate directory from frozen study buffers")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    report = build(args.minari_root, selection, args.output_dir, args.with_quality, args.threads)
    (args.output_dir / "reconstruction.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

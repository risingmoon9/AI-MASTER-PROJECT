"""Check the common Kitchen observation, reward, trajectory and quality contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

TASKS = ["kettle", "light switch", "microwave", "slide cabinet"]
REQUIRED = {"obs", "next_obs", "actions", "rewards", "dones", "truncations",
            "returns_to_go", "episode_id", "timestep", "episode_return",
            "episode_length", "source_prior", "source_name", "source_dataset_id", "metadata"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        d = {key: z[key] for key in z.files}
    require(REQUIRED <= set(d), f"Missing fields: {sorted(REQUIRED - set(d))}")
    meta = json.loads(str(d["metadata"]))
    require(meta.get("schema") == "kitchen_common81_persistent_v1", "Wrong observation/reward schema")
    require(meta.get("tasks") == TASKS and meta.get("goal_order") == TASKS, "Wrong task/goal order")
    require(meta.get("remove_task_when_completed") is False and meta.get("terminate_on_tasks_completed") is False, "Wrong persistent task settings")
    require(d["obs"].ndim == 2 and d["obs"].shape[1] == 81, "Expected 81-dimensional observations")
    n = len(d["obs"])
    require(n > 0 and d["next_obs"].shape == d["obs"].shape, "Empty or mismatched observations")
    require(d["actions"].shape == (n, 9), "Expected nine-dimensional actions")
    for key, value in d.items():
        if key == "metadata":
            continue
        require(value.ndim >= 1 and len(value) == n, f"Wrong transition count: {key}")
        if value.dtype.kind in "fiu":
            require(np.isfinite(value).all(), f"Non-finite values: {key}")
    for key in ["rewards", "dones", "truncations", "returns_to_go", "episode_id", "timestep", "episode_return", "episode_length"]:
        require(d[key].shape == (n, 1), f"Expected column vector: {key}")
    require(np.isin(d["dones"], [0, 1]).all() and np.isin(d["truncations"], [0, 1]).all(), "Invalid terminal flags")
    # The frozen common task has no true terminations. Time limits are separate.
    require((d["dones"] == 0).all(), "Unexpected terminal in persistent common task")
    reward = sum((np.linalg.norm(d["next_obs"][:, lo:hi] - d["next_obs"][:, 11+lo:11+hi], axis=1) < .3).astype(np.float32)
                 for lo, hi in [(0, 7), (7, 9), (9, 10), (10, 11)])
    np.testing.assert_array_equal(reward, d["rewards"].ravel())
    ids = d["episode_id"].ravel()
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    require(len(starts) == len(np.unique(ids)), "Episode IDs must occur in contiguous blocks")
    discount = float(meta["discount"])
    require(0 <= discount <= 1, "Invalid discount")
    for start, end in zip(starts, np.r_[starts[1:], n]):
        np.testing.assert_array_equal(d["timestep"][start:end].ravel(), np.arange(end-start))
        np.testing.assert_array_equal(d["obs"][start+1:end], d["next_obs"][start:end-1])
        np.testing.assert_allclose(d["episode_return"][start:end], reward[start:end].sum(), rtol=0, atol=1e-5)
        np.testing.assert_array_equal(d["episode_length"][start:end], np.full((end-start, 1), end-start))
        acc = 0.0
        rtg = np.empty(end-start, np.float32)
        for i in range(end-start-1, -1, -1):
            acc = float(reward[start+i]) + discount * acc
            rtg[i] = acc
        np.testing.assert_array_equal(d["returns_to_go"][start:end, 0], rtg)
    if "quality_weight" in d:
        require((d["quality_weight"] > 0).all(), "Non-positive quality weight")
        require(abs(float(d["quality_weight"].mean()) - 1) < 1e-5, "Quality weights are not mean normalized")
        np.testing.assert_allclose(d["support"] + d["uncertainty"], 1, rtol=0, atol=1e-6)
        require(((d["support"] >= 0) & (d["support"] <= 1)).all(), "Support outside [0,1]")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"file": path.name, "transitions": n, "episodes": len(starts),
            "schema_reward_rtg_continuity_finite_checks": "PASS", "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    paths = args.paths or [args.root / "data/processed" / name for name in ["kitchen_mixture_quality.npz", "kitchen_complete_quality.npz"]]
    results = [validate(path) for path in paths]
    out = args.root / "runs/audit/data_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

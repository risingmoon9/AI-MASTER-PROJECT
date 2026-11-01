"""Prepare compact quality overlays without changing the common-task base buffers.

Run from the project root: python scripts/prepare_data_variants.py
The delivered overlays are authoritative; regeneration may change KMeans labels
between scikit-learn versions and must be treated as a new preprocessing build.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

SCHEMA = "hqa_quality_overlay_v1"
BASES = {
    "low": "data/processed/kitchen_mixture_quality.npz",
    "complete": "data/processed/kitchen_complete_quality.npz",
}
FEATURE_NAMES = ["episode_return", "episode_length", "mean_action_norm", "mean_action_variance", "observation_displacement", "mean_reward", "max_reward"]
ALLOWED_FIELDS = {"quality_weight", "trajectory_quality", "transition_quality", "cluster_id", "source_prior"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def minmax(x):
    x = np.asarray(x, dtype=np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-6)


def zscore(x):
    x = np.asarray(x, dtype=np.float32)
    return (x - x.mean()) / (x.std() + 1e-6)


def success_proxy(ep_returns):
    """Retain the base implementation's transition-weighted return threshold."""
    if float(ep_returns.max()) <= float(ep_returns.min()) + 1e-6:
        return np.zeros_like(ep_returns, dtype=np.float32)
    threshold = float(np.quantile(ep_returns, 0.75))
    if threshold <= 1e-6 and float(ep_returns.max()) > 1e-6:
        return (ep_returns > 1e-6).astype(np.float32)
    return (ep_returns > threshold).astype(np.float32)


def normalize_weight(raw, balance=None):
    weights = 1.0 / (1.0 + np.exp(-raw))
    if balance is not None:
        weights = weights * balance
    weights = np.clip(weights, 0.05, 5.0)
    return (weights / (weights.mean() + 1e-6)).reshape(-1, 1).astype(np.float32)


def no_source_features(data):
    """Build exactly the seven non-source features; never read source_prior."""
    episode_ids = np.unique(data["episode_id"].reshape(-1))
    features = []
    for eid in episode_ids:
        mask = data["episode_id"].reshape(-1) == eid
        rewards = data["rewards"][mask].reshape(-1)
        actions, obs, nxt = data["actions"][mask], data["obs"][mask], data["next_obs"][mask]
        features.append([
            float(rewards.sum()), float(mask.sum()),
            float(np.linalg.norm(actions, axis=-1).mean()), float(actions.var(axis=0).mean()),
            float(np.linalg.norm(nxt[-1] - obs[0])), float(rewards.mean()), float(rewards.max()),
        ])
    return episode_ids, np.asarray(features, dtype=np.float32)


def make_no_source_prior(data, seed=0, n_clusters=5):
    episode_ids, features = no_source_features(data)
    scaled = StandardScaler().fit_transform(features)
    k = min(n_clusters, len(episode_ids))
    labels = (KMeans(n_clusters=k, random_state=seed, n_init="auto").fit_predict(scaled)
              if k > 1 else np.zeros(len(episode_ids))).astype(np.int32)
    idx = np.searchsorted(episode_ids, data["episode_id"].reshape(-1))
    clusters = labels[idx]
    ep_returns = data["episode_return"].reshape(-1)
    # Drop the alpha_source term, retaining the original other coefficients.
    trajectory = 0.55 * minmax(ep_returns) + 0.15 * success_proxy(ep_returns)
    _, counts = np.unique(clusters, return_counts=True)
    count_map = dict(zip(np.unique(clusters), counts))
    balance = np.asarray([1.0 / np.sqrt(count_map[c] + 1e-6) for c in clusters], dtype=np.float32)
    balance = balance / (balance.mean() + 1e-6)
    raw = zscore(trajectory) + zscore(data["transition_quality"].reshape(-1))
    return {
        "quality_weight": normalize_weight(raw, balance),
        "trajectory_quality": trajectory.reshape(-1, 1).astype(np.float32),
        "cluster_id": clusters.reshape(-1, 1).astype(np.int32),
        "source_prior": np.zeros_like(data["source_prior"]),
    }


def make_return_only(data):
    """Depend only on repeated episode returns; no support, RTG or source use."""
    returns = np.asarray(data["episode_return"], dtype=np.float32).reshape(-1)
    return {
        "quality_weight": normalize_weight(zscore(returns)),
        "trajectory_quality": minmax(returns).reshape(-1, 1).astype(np.float32),
        "transition_quality": np.zeros((len(returns), 1), dtype=np.float32),
        "cluster_id": np.full((len(returns), 1), -1, dtype=np.int32),
    }


def load_overlay(base, path, *, expected_sha256):
    """Validate and return a new merged mapping; leave the input mapping intact.

    expected_sha256 must be the SHA-256 independently computed for the base NPZ
    that was loaded. The base metadata is preserved; overlay metadata is returned
    separately. Caller must also verify the overlay file hash from its manifest.
    """
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["overlay_metadata"]))
        if meta.get("schema") != SCHEMA or meta.get("base_sha256") != expected_sha256:
            raise ValueError("Overlay schema/base hash mismatch")
        if meta.get("base_transitions") != len(base["obs"]):
            raise ValueError("Overlay transition count mismatch")
        fields = set(z.files) - {"overlay_metadata"}
        if fields != set(meta["fields"]) or not fields <= ALLOWED_FIELDS:
            raise ValueError("Unexpected overlay fields")
        merged = dict(base)
        for key in sorted(fields):
            arr = z[key]
            if key not in base or arr.shape != base[key].shape or arr.dtype != base[key].dtype:
                raise ValueError(f"Overlay shape/dtype mismatch: {key}")
            if not np.isfinite(arr).all():
                raise ValueError(f"Non-finite overlay: {key}")
            if key == "quality_weight" and ((arr <= 0).any() or abs(float(arr.mean()) - 1) > 1e-5):
                raise ValueError("Invalid quality weights")
            merged[key] = arr
    return merged, meta


def stats(a):
    return {"min": float(a.min()), "max": float(a.max()), "mean": float(a.astype(np.float64).mean()), "std": float(a.astype(np.float64).std())}


def cluster_counts(a):
    ids, counts = np.unique(a.reshape(-1), return_counts=True)
    return {str(int(i)): int(n) for i, n in zip(ids, counts)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--output-dir", type=Path, default=Path("data/rebuilt"), help="Separate destination for regenerated overlays")
    args = p.parse_args()
    root = args.root.resolve()
    if args.threads < 1:
        p.error("--threads must be positive")
    target = (root / args.output_dir).resolve()
    if target == (root / "data/processed").resolve():
        p.error("Use a separate output directory; frozen study assets must not be overwritten")
    target.mkdir(parents=True, exist_ok=True)
    expected = json.loads((root / "DATA_SHA256.json").read_text(encoding="utf-8"))
    audit = {
        "schema": SCHEMA,
        "builder": "scripts/prepare_data_variants.py",
        "builder_sha256": sha256(Path(__file__)),
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "normalization": "All moments are over transitions, preserving episode multiplicity; clip raw sigmoid-times-balance to [0.05,5], divide by whole-buffer mean+1e-6, no second clip. Thus normalized bounds need not equal [0.05,5].",
        "no_source_prior": {
            "features": FEATURE_NAMES, "clustering": "StandardScaler; KMeans(k=min(5,n_episodes),random_state=0,n_init='auto')",
            "trajectory_quality": "0.55*minmax(episode_return) + 0.15*success_proxy; alpha_source term removed without changing other coefficients",
            "success_proxy": "Existing transition-weighted 75th-percentile proxy, with unchanged constant/sparse-return exceptions",
            "weight": "sigmoid(zscore(trajectory_quality)+zscore(original_transition_quality)) * transition_count(cluster)^(-1/2), with cluster balance divided by mean+1e-6; clip and normalize as above",
            "preserved": ["obs", "next_obs", "actions", "rewards", "returns_to_go", "support", "uncertainty", "transition_quality", "source_name", "source_dataset_id", "episode_order_and_multiplicity"],
            "source_prior": "Effective overlay is zero; the immutable base retains original values and source provenance.",
        },
        "return_only": {
            "weight": "sigmoid((episode_return - transition_mean(episode_return))/(transition_std(episode_return)+1e-6)), then clip and normalize as above",
            "dependencies": ["episode_return"],
            "constant_return_case": "All weights equal (approximately one under the same epsilon convention).",
            "diagnostic_fields": "trajectory_quality=minmax(episode_return), transition_quality=0, cluster_id=-1. Original support/source fields remain as provenance and must be disabled in the quality-only actor configuration.",
        },
        "constant_bc": {"formula": "mean_over_all_transitions(clip(0.2*(1+uncertainty-0.5*support),0,2))", "purpose": "Match the fixed whole-buffer mean coefficient of the adaptive BC term; not its minibatch realization or weighting by policy loss.", "tuned_on_test": False},
        "datasets": {},
    }
    for name, rel in BASES.items():
        path = root / rel
        before = sha256(path)
        if before != expected.get(rel):
            raise ValueError(f"Base data hash differs from the frozen protocol: {rel}")
        with np.load(path, allow_pickle=False) as z:
            data = {key: z[key] for key in z.files}
        meta = json.loads(str(data["metadata"]))
        if meta.get("schema") != "kitchen_common81_persistent_v1" or meta.get("quality_status") != "recomputed":
            raise ValueError("Expected common-task, quality-recomputed base data")
        if not np.isfinite(data["quality_weight"]).all():
            raise ValueError("Non-finite input quality")
        coefficient = np.clip(np.float32(0.2) * (np.float32(1.0) + data["uncertainty"] - np.float32(0.5) * data["support"]), 0, 2)
        record = {"base_path": rel, "base_sha256": before, "transitions": len(data["obs"]), "episodes": len(np.unique(data["episode_id"])), "constant_bc_lambda_mean": float(coefficient.astype(np.float64).mean()), "adaptive_lambda": stats(coefficient), "original_weight": stats(data["quality_weight"]), "variants": {}}
        with threadpool_limits(limits=args.threads):
            variants = {"no_source_prior": make_no_source_prior(data), "return_only": make_return_only(data)} if name == "low" else {}
        for variant, arrays in variants.items():
            overlay_meta = {"schema": SCHEMA, "variant": variant, "base_path": rel, "base_sha256": before, "base_transitions": len(data["obs"]), "fields": sorted(arrays), "seed": 0, "clustering_n_clusters": 5 if variant == "no_source_prior" else 0, "audit_path": "runs/audit/data_variants_generated.json"}
            out = target / f"mixture_{variant}.npz"
            np.savez_compressed(out, **arrays, overlay_metadata=np.asarray(json.dumps(overlay_meta, sort_keys=True)))
            merged, _ = load_overlay(data, out, expected_sha256=before)
            unchanged = [key for key in data if key not in arrays]
            if any(not np.array_equal(data[key], merged[key]) for key in unchanged):
                raise AssertionError("Overlay changed an unlisted field")
            entry = {"path": str(out.relative_to(root)) if out.is_relative_to(root) else str(out), "sha256": sha256(out), "bytes": out.stat().st_size, "fields": sorted(arrays), "quality_weight": stats(arrays["quality_weight"]), "original_transition_cluster_counts": cluster_counts(data["cluster_id"]), "variant_transition_cluster_counts": cluster_counts(arrays["cluster_id"]), "effective_weight_sample_size": float(arrays["quality_weight"].sum(dtype=np.float64)**2 / np.square(arrays["quality_weight"].astype(np.float64)).sum())}
            if variant == "no_source_prior":
                episode_first = np.unique(data["episode_id"].reshape(-1), return_index=True)[1]
                entry["cluster_adjusted_rand_index_vs_original_episodes"] = float(adjusted_rand_score(data["cluster_id"].reshape(-1)[episode_first], arrays["cluster_id"].reshape(-1)[episode_first]))
                entry["changed_weight_transitions_atol_1e_6"] = int(np.sum(~np.isclose(arrays["quality_weight"], data["quality_weight"], rtol=0, atol=1e-6)))
            record["variants"][variant] = entry
        if sha256(path) != before:
            raise AssertionError("Base file changed")
        record["base_unchanged_after_build"] = True
        audit["datasets"][name] = record
    (root / "runs" / "audit").mkdir(parents=True, exist_ok=True)
    (root / "runs" / "audit" / "data_variants_generated.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {"constant_bc_lambda_mean": rec["constant_bc_lambda_mean"], "overlay_paths": [v["path"] for v in rec["variants"].values()]} for name, rec in audit["datasets"].items()}, indent=2))


if __name__ == "__main__":
    main()

"""Scientific invariance and overlay integrity checks; no training required."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_data_variants", ROOT / "scripts" / "prepare_data_variants.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture():
    rng = np.random.default_rng(73)
    n, width = 48, 3
    episodes = np.repeat(np.arange(8), 6).astype(np.int32).reshape(-1, 1)
    rewards = rng.integers(0, 5, (n, 1)).astype(np.float32)
    returns = np.concatenate([np.full((6, 1), rewards[i:i+6].sum(), dtype=np.float32) for i in range(0, n, 6)])
    return {"episode_id": episodes, "rewards": rewards,
            "episode_return": returns, "obs": rng.normal(size=(n, width)).astype(np.float32),
            "next_obs": rng.normal(size=(n, width)).astype(np.float32),
            "actions": rng.normal(size=(n, 2)).astype(np.float32),
            "source_prior": rng.uniform(size=(n, 1)).astype(np.float32),
            "transition_quality": rng.normal(size=(n, 1)).astype(np.float32),
            "quality_weight": np.ones((n, 1), dtype=np.float32),
            "trajectory_quality": np.zeros((n, 1), dtype=np.float32),
            "cluster_id": np.zeros((n, 1), dtype=np.int32),
            "metadata": np.asarray("original provenance")}


class QualityVariantTests(unittest.TestCase):
    def test_no_source_invariant_to_changed_prior_and_names(self):
        data = fixture()
        original = {k: v.copy() for k, v in data.items()}
        alternative = dict(data)
        alternative["source_prior"] = np.full_like(data["source_prior"], np.nan)
        alternative["source_name"] = np.asarray(["counterfactual"] * len(data["obs"]))
        with threadpool_limits(limits=1):
            before = m.make_no_source_prior(data)
            after = m.make_no_source_prior(alternative)
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])
        for key in data:
            np.testing.assert_array_equal(original[key], data[key])
        self.assertEqual(m.no_source_features(data)[1].shape[1], 7)
        self.assertEqual(float(before["source_prior"].max()), 0)

    def test_return_only_needs_only_returns_and_is_monotone(self):
        data = fixture()
        direct = m.make_return_only({"episode_return": data["episode_return"]})
        full = m.make_return_only(data)
        order = np.argsort(data["episode_return"].reshape(-1))
        for key in direct:
            np.testing.assert_array_equal(direct[key], full[key])
        self.assertTrue((np.diff(direct["quality_weight"].reshape(-1)[order]) >= 0).all())
        self.assertTrue((direct["cluster_id"] == -1).all())
        self.assertTrue((direct["transition_quality"] == 0).all())

    def test_constant_returns_receive_equal_positive_weights(self):
        w = m.make_return_only({"episode_return": np.full((20, 1), 7, dtype=np.float32)})["quality_weight"]
        np.testing.assert_allclose(w, np.ones_like(w), rtol=0, atol=3e-6)

    def test_overlay_wrong_hash_and_unlisted_fields_are_rejected(self):
        data = fixture()
        overlay = m.make_return_only(data)
        metadata = {"schema": m.SCHEMA, "base_sha256": "correct", "base_transitions": len(data["obs"]), "fields": sorted(overlay)}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "overlay.npz"
            np.savez_compressed(path, **overlay, overlay_metadata=np.asarray(json.dumps(metadata)))
            merged, _ = m.load_overlay(data, path, expected_sha256="correct")
            np.testing.assert_array_equal(merged["metadata"], data["metadata"])
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                m.load_overlay(data, path, expected_sha256="wrong")
            np.savez_compressed(path, **overlay, obs=data["obs"], overlay_metadata=np.asarray(json.dumps(metadata)))
            with self.assertRaisesRegex(ValueError, "Unexpected overlay"):
                m.load_overlay(data, path, expected_sha256="correct")

    def test_delivered_overlays_and_immutable_bases(self):
        audit_path = ROOT / "runs" / "audit" / "data_variants.json"
        if not audit_path.is_file():
            self.skipTest("Optional local-data integration check: import or prepare datasets first")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        for name, rec in audit["datasets"].items():
            path = ROOT / rec["base_path"]
            before = m.sha256(path)
            self.assertEqual(before, rec["base_sha256"])
            with np.load(path, allow_pickle=False) as z:
                data = {key: z[key] for key in z.files}
            for variant, info in rec["variants"].items():
                overlay = ROOT / info["path"]
                self.assertEqual(m.sha256(overlay), info["sha256"])
                merged, meta = m.load_overlay(data, overlay, expected_sha256=before)
                self.assertEqual(meta["variant"], variant)
                w = merged["quality_weight"]
                self.assertTrue(np.isfinite(w).all())
                self.assertTrue((w > 0).all())
                self.assertLess(abs(float(w.mean()) - 1), 1e-5)
                for key in data:
                    if key not in info["fields"]:
                        np.testing.assert_array_equal(merged[key], data[key])
            coefficient = np.clip(np.float32(.2) * (np.float32(1) + data["uncertainty"] - np.float32(.5) * data["support"]), 0, 2)
            self.assertAlmostEqual(float(coefficient.astype(np.float64).mean()), rec["constant_bc_lambda_mean"], places=12)
            self.assertEqual(m.sha256(path), before)


if __name__ == "__main__":
    unittest.main()

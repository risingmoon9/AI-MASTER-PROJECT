"""Data integrity checks use synthetic arrays; no dataset or simulator required."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import h5py
import numpy as np
import pytest

from scripts.build_kitchen_data import TASKS, GOALS, canonical_observations, physical_identity, record_arrays
from scripts.import_data import atomic_verified_copy, member_name, remap_variant_manifest
from scripts.validate_data import validate


def test_verified_copy_is_idempotent_and_refuses_conflicting_data(tmp_path):
    payload = b"verified scientific data"
    expected = hashlib.sha256(payload).hexdigest()
    target = tmp_path / "assets" / "sample.npz"
    assert atomic_verified_copy(io.BytesIO(payload), target, expected) == "imported"
    assert atomic_verified_copy(io.BytesIO(payload), target, expected) == "already_verified"
    target.write_bytes(b"different")
    with pytest.raises(FileExistsError):
        atomic_verified_copy(io.BytesIO(payload), target, expected)
    assert target.read_bytes() == b"different"


def test_bad_checksum_leaves_no_partial_asset(tmp_path):
    target = tmp_path / "sample.npz"
    with pytest.raises(ValueError, match="Checksum mismatch"):
        atomic_verified_copy(io.BytesIO(b"corrupt"), target, "0" * 64)
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_archive_ambiguous_members_are_rejected(tmp_path):
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("one/data/x.npz", b"1")
        archive.writestr("two/data/x.npz", b"2")
    with zipfile.ZipFile(path) as archive:
        with pytest.raises(ValueError, match="found 2"):
            member_name(archive, "data/x.npz")


def test_import_remaps_paths_without_mutating_original_manifest():
    record = {"datasets": {"low": {"variants": {"no_source_prior": {}, "return_only": {}}}, "complete": {"variants": {}}}}
    clean = remap_variant_manifest(record)
    assert clean["datasets"]["low"]["base_path"] == "data/processed/kitchen_mixture_quality.npz"
    assert clean["datasets"]["low"]["variants"]["return_only"]["path"] == "data/processed/mixture_return_only.npz"
    assert "base_path" not in record["datasets"]["low"]


def test_missing_goal_is_recovered_from_matching_peer_and_conflicts_fail(tmp_path):
    with h5py.File(tmp_path / "source.h5", "w") as h:
        a, b = h.create_group("mixed"), h.create_group("partial")
        for ep in [a, b]:
            ep.create_dataset("observations/observation", data=np.zeros((4, 59), np.float32))
        for task in TASKS:
            values = np.ones((4, len(GOALS[task])), np.float32)
            b.create_dataset("observations/achieved_goal/" + task, data=values)
            if task != "slide cabinet":
                a.create_dataset("observations/achieved_goal/" + task, data=values)
        obs = canonical_observations(a, [a, b])
        assert obs.shape == (4, 81)
        assert np.all(obs[:, 10] == 1)
        with pytest.raises(ValueError, match="slide cabinet"):
            canonical_observations(a, [a])
        b["observations/achieved_goal/kettle"][0, 0] = 2
        with pytest.raises(AssertionError):
            canonical_observations(a, [a, b])


def test_identity_is_sensitive_to_action_and_successor_state():
    obs = np.zeros((4, 59), np.float32)
    act = np.zeros((3, 9), np.float32)
    original = physical_identity(obs, act)
    obs[-1, 0] = 1
    assert physical_identity(obs, act) != original
    obs[-1, 0] = 0
    act[0, 0] = 1
    assert physical_identity(obs, act) != original


def test_validator_checks_reward_and_continuity(tmp_path):
    obs = np.zeros((4, 81), np.float32)
    data = record_arrays(obs, np.zeros((3, 9), np.float32), np.array([0, 0, 1]), 0, "complete")
    data["metadata"] = np.asarray(json.dumps({"schema": "kitchen_common81_persistent_v1", "tasks": TASKS, "goal_order": TASKS,
        "remove_task_when_completed": False, "terminate_on_tasks_completed": False, "discount": .99}))
    path = tmp_path / "data.npz"
    np.savez_compressed(path, **data)
    assert validate(path)["transitions"] == 3
    data["rewards"][0, 0] = 0
    np.savez_compressed(path, **data)
    with pytest.raises(AssertionError):
        validate(path)
    data["rewards"][0, 0] = 4
    data["obs"] = data["obs"].copy()
    data["obs"][1, 40] = 1
    np.savez_compressed(path, **data)
    with pytest.raises(AssertionError):
        validate(path)

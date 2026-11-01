"""Import the frozen study data from a local prepared directory or release ZIP.

Only the six named, checksum-verified assets below are copied. No download,
archive-wide extraction, pickle loading, or experiment-code execution occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    "data/processed/kitchen_mixture_quality.npz": (
        "data_repaired/kitchen_low_complete_repaired_quality.npz",
        "cf03fa462ed792a7c49269f2bb82611cda40c27d099c1047d74a22f04d12ef91"),
    "data/processed/kitchen_complete_quality.npz": (
        "data_repaired/kitchen_complete_original_repaired_quality.npz",
        "ebd7fb41cc7226b5de78d9d2b6dacd64837e60397f06e5f343144174024fe1bd"),
    "data/processed/mixture_no_source_prior.npz": (
        "data_variants/low_no_source_prior.npz",
        "0f1b83112bf00b7f1546c32b67df187036d78f077a222365aa19288f302ee43d"),
    "data/processed/mixture_return_only.npz": (
        "data_variants/low_return_only.npz",
        "0b4b6a0003106992d7960d51410d13babf5339a5e732f878b18afb4ee60bdfc5"),
    "minari_datasets/D4RL/kitchen/complete-v2/data/main_data.hdf5": (
        "minari_datasets/D4RL/kitchen/complete-v2/data/main_data.hdf5",
        "5a4e44b8df1e3fcb7ccda28ed8fa7303cd2bd1cc636d7471e90ec6ea818446be"),
    "minari_datasets/D4RL/kitchen/complete-v2/data/metadata.json": (
        "minari_datasets/D4RL/kitchen/complete-v2/data/metadata.json",
        "63713c60753cdc366a74724d8e85b1512d0c6d2e8dbb19bdf91fadb8a2ba223c"),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def member_name(archive, relative):
    """Find one exact relative asset underneath an optional archive root."""
    matches = [name for name in archive.namelist()
               if name == relative or name.endswith("/" + relative)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {relative} in archive; found {len(matches)}")
    return matches[0]


def atomic_verified_copy(stream, target, expected):
    target = Path(target)
    if target.exists():
        if sha256(target) != expected:
            raise FileExistsError(f"Refusing to replace existing data with a different hash: {target}")
        return "already_verified"
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + target.name + "-", dir=target.parent)
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "wb") as output:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if digest.hexdigest() != expected:
            raise ValueError(f"Checksum mismatch for {target.name}")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return "imported"


def source_file(source, clean, legacy):
    candidates = [source / clean, source / legacy]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Prepared source lacks {clean} (legacy name: {legacy})")


def remap_variant_manifest(manifest):
    # The embedded overlay metadata remains byte-identical to the frozen asset.
    # Its former base_path is informational; load_overlay verifies base_sha256.
    manifest = json.loads(json.dumps(manifest))
    manifest["builder"] = "frozen study overlays imported by scripts/import_data.py"
    for name, filename in [("low", "kitchen_mixture_quality.npz"),
                           ("complete", "kitchen_complete_quality.npz")]:
        record = manifest["datasets"][name]
        record["base_path"] = "data/processed/" + filename
        if name == "complete":
            record["variants"] = {}
        else:
            for variant in ["no_source_prior", "return_only"]:
                record["variants"][variant]["path"] = f"data/processed/mixture_{variant}.npz"
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path, help="Local prepared study release ZIP")
    source.add_argument("--source-dir", type=Path, help="Root of an extracted release or prepared repository")
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository destination")
    args = parser.parse_args()
    root = args.root.resolve()
    expected_manifest = json.loads((root / "DATA_SHA256.json").read_text(encoding="utf-8"))
    if expected_manifest != {clean: value[1] for clean, value in ASSETS.items()}:
        raise ValueError("DATA_SHA256.json differs from the frozen import specification")
    outcomes = {}
    if args.archive:
        with zipfile.ZipFile(args.archive) as archive:
            members = {clean: member_name(archive, legacy)
                       for clean, (legacy, _) in ASSETS.items()}
            for clean, (_, expected) in ASSETS.items():
                with archive.open(members[clean]) as stream:
                    outcomes[clean] = atomic_verified_copy(stream, root / clean, expected)
            record = json.loads(archive.read(member_name(archive, "audit/data_variants.json")))
    else:
        source_root = args.source_dir.resolve()
        inputs = {clean: source_file(source_root, clean, legacy)
                  for clean, (legacy, _) in ASSETS.items()}
        for clean, (_, expected) in ASSETS.items():
            with inputs[clean].open("rb") as stream:
                outcomes[clean] = atomic_verified_copy(stream, root / clean, expected)
        manifests = [source_root / "runs/audit/data_variants.json", source_root / "audit/data_variants.json"]
        manifest_path = next((path for path in manifests if path.is_file()), None)
        if manifest_path is None:
            raise FileNotFoundError("Prepared source lacks runs/audit/data_variants.json")
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = root / "runs" / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    (audit / "data_variants.json").write_text(json.dumps(remap_variant_manifest(record), indent=2) + "\n", encoding="utf-8")
    (audit / "data_import.json").write_text(json.dumps({"source_kind": "local_zip" if args.archive else "local_directory", "assets": outcomes, "sha256": expected_manifest}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "verified", "assets": outcomes}, indent=2))


if __name__ == "__main__":
    main()

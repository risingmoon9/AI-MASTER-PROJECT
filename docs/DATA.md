# Data and preprocessing

The repository contains code, configurations and trajectory-selection metadata.
It does not contain training buffers, Minari HDF5 files, or model checkpoints.
The frozen input hashes are in [`DATA_SHA256.json`](../DATA_SHA256.json).

## Import the exact study inputs

From the project root, import the locally retained prepared-data archive:

```bash
python scripts/import_data.py --archive /path/to/HQA_IQL_AutoDL_Supplement_20261017.zip
python scripts/validate_data.py
```

That archive name is a compatibility input, not a separate experimental method.
Alternatively, use `--source-dir /path/to/prepared_project`. The directory may
contain the legacy asset names or the clean names below, plus
`runs/audit/data_variants.json` (the legacy `audit/data_variants.json` is also
accepted). Neither command downloads data. The archive importer only streams six
named assets; it does not extract or execute other archive contents.

| Asset | Destination |
| --- | --- |
| Mixture training buffer | `data/processed/kitchen_mixture_quality.npz` |
| Complete-only training buffer | `data/processed/kitchen_complete_quality.npz` |
| No-source-prior quality overlay | `data/processed/mixture_no_source_prior.npz` |
| Return-only quality overlay | `data/processed/mixture_return_only.npz` |
| Evaluation data | `minari_datasets/D4RL/kitchen/complete-v2/data/main_data.hdf5` |
| Evaluation environment specification | `minari_datasets/D4RL/kitchen/complete-v2/data/metadata.json` |

All six assets must match their recorded SHA-256 checksums. Already matching
files are retained; conflicting existing files are refused. Import records are
written under `runs/audit/`. The default local Minari root is `minari_datasets/`;
an explicitly supplied `MINARI_DATASETS_PATH` is respected by the runtime. An
external Minari root must provide the same evaluation dataset and metadata.

The GitHub source package alone does not make the frozen data available to a
new reader. Exact study reproduction requires access to those local assets.
Do not describe the archive as a public download unless it has actually been
published with appropriate data redistribution permission.

## Common task and observation schema

The experiment uses a custom persistent-reward Kitchen task. It is not a claim
to reproduce the standard D4RL normalized-score benchmark.

The 81 observation coordinates concatenate, in this order:

1. Achieved goals (11): kettle (7), light switch (2), microwave (1), slide cabinet (1).
2. Desired goals (11), in the same task order.
3. The physical observation (59).

Actions have nine coordinates. At each transition the reward is the number of
the four next-state achieved-goal distances strictly below `0.3`. Rewards can
therefore repeat after a task is achieved. Evaluation uses a 280-step horizon,
`remove_task_when_completed=False` and `terminate_on_tasks_completed=False`.
True-terminal masks are zero in these buffers; truncation flags remain separate,
and value targets bootstrap through time limits. Offline source trajectories are
retained at their full length, including trajectories longer than 280 steps.

The mixture contains **494 episode occurrences and 124,019 transitions**:

| Source | Episode occurrences | Transitions |
| --- | ---: | ---: |
| Complete | 19 | 4,209 |
| Partial | 225 | 56,772 |
| Mixed | 250 | 63,038 |

There are 385 unique physical/action trajectories in the mixture. Repeated
occurrences were retained, rather than deduplicated. The complete-only buffer
contains 19 episodes and 4,209 transitions. The complete source contributes
approximately 3.39% of mixture transitions.

## Reconstruct preprocessing from local Minari sources

`protocol/kitchen_episode_selection.json` records the exact episode order,
multiplicity, source, candidate original IDs, lengths and SHA-256 physical/action
identities. Ambiguous original IDs remain recorded as candidates.

Given all three local HDF5 sources (`complete-v2`, `partial-v2`, `mixed-v2`):

```bash
python scripts/build_kitchen_data.py --minari-root /path/to/minari_datasets
# Optional: also recompute clustering and kNN quality (substantial CPU work).
python scripts/build_kitchen_data.py --minari-root /path/to/minari_datasets \
    --output-dir data/rebuilt_with_quality --with-quality --threads 1
```

The default reconstruction directory is `data/rebuilt/`. Reconstruction never
overwrites the frozen `data/processed/` buffers. The mixed source omits the
slide-cabinet achieved-goal channel. Reconstruction retrieves this channel from
a matching partial trajectory only after verifying identical float32 physical
observations (including the successor row) and actions. It refuses missing or
conflicting goal channels. It does not replace them with noisy physical
coordinates or pad mismatched observation vectors.

The reconstruction was checked against both frozen buffers: all 14 non-quality
transition/provenance arrays matched exactly, including observations, actions,
rewards, returns-to-go, flags and episode order. Regenerated archive bytes and
quality arrays are not promised to be identical across dependency versions.
Use the frozen import route for the committed experiment configurations; do
not replace their expected hashes merely to bypass verification.

## Quality signals and overlays

Base quality uses five episode clusters, seed zero, and kNN support with `k=10`
and at most 100,000 fit transitions. Source priors are complete `1.0`, partial
`0.55`, mixed `0.30`. All normalization moments preserve transition multiplicity.

The no-source-prior control removes the source coefficient from trajectory
quality and removes source prior from clustering features, then recomputes
clusters and weights. It retains the other coefficients without renormalizing
their sum. The return-only control depends only on the repeated episode return,
using a sigmoid of its transition-weighted z-score. Raw weights are clipped to
`[0.05, 5]`, then divided by the whole-buffer mean plus epsilon, with no second
clip. The final normalized range need not be `[0.05, 5]`.

Overlays contain only named quality fields and a base-file hash. Applying them
must preserve observations, actions, rewards, terminal masks and episode order.
The loader checks field names, shapes, dtypes, finiteness, base identity and mean
weight normalization; the runner checks the overlay-file hash separately.

To study preprocessing independently of the frozen experiment assets:

```bash
python scripts/prepare_data_variants.py --threads 1
```

This writes regenerated overlays to `data/rebuilt/` and records
`runs/audit/data_variants_generated.json`. The frozen import manifest is left
unchanged. KMeans labels can vary across scikit-learn versions; new preprocessing
outputs belong to a new, explicitly recorded experiment configuration.

## Validation

`validate_data.py` checks the 81/9 schema, finite arrays, persistent-task settings,
next-state reward reconstruction, episode continuity and timesteps, episode
returns/lengths, discounted returns-to-go, terminal flags and quality weights.
Import verifies file bytes independently. Unit tests use synthetic data so they
run without private datasets; optional local-data tests check the imported
overlays against their immutable bases.

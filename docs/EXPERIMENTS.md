# Experimental protocol and execution

This repository treats the Kitchen experiments as one study. The protocol has two statistical comparison blocks because their training and reset seeds differ. The machine-readable source of truth is [`protocol/experiments.json`](../protocol/experiments.json); each run has an explicit YAML configuration in [`configs/experiments`](../configs/experiments).

## Study design

| Block / preset | Independent training seeds | Validation resets | Test resets | Runs |
|---|---|---|---|---:|
| `comparisons` | 3, 4, 5 | 11000–11019 | 30000–30049 | 42 |
| `ablations` | 0, 1, 2 | 10000–10019 | 20000–20049 | 18 |
| `all` | Both blocks, reported separately | As above | As above | 60 |

Each run uses 200,000 gradient updates and minibatches of 256 transitions. Every 5,000 updates, deterministic evaluation uses 20 validation episodes. The selected model is the first checkpoint attaining a strictly higher validation mean return than all previous checkpoints. Both that model and the fixed 200,000-update endpoint are then evaluated on the same 50 test resets within their block. All runs are retained when summarizing a method; test outcomes do not select the checkpoint or a training seed.

The statistic of interest is the mean across the three independently trained models and its sample standard deviation. The 50 test episodes characterize each model, not 50 independently trained replications. Do not combine the two blocks into an apparent six-seed result: their reset distributions differ. Engineering checks in the recorded ablation experiment used test resets 20000 and 20001; no adjustment based on those outcomes is documented. This limits a strict held-out interpretation of that block. Current smoke checks use separate resets 12000–12003.

## Methods and datasets

The dataset selector is `mixture` or `complete`. Run names retain `low` as a short stable identifier for the mixture; it is not a third dataset. See [DATA.md](DATA.md) for construction and limitations and [METHOD.md](METHOD.md) for the learning objectives.

| CLI method | Meaning | Comparisons | Ablations |
|---|---|---|---|
| `baseline` | Unweighted project IQL baseline (IQL-M) | Both datasets | Both datasets |
| `full` | Quality weights and adaptive BC penalty (HQA-IQL) | Both datasets | Both datasets |
| `quality_only` | Quality weights, no additional BC penalty | Both datasets | Mixture |
| `adaptive_bc_only` | Adaptive BC penalty, no quality weights | — | Mixture |
| `constant_bc` | Quality weights with a dataset-specific constant BC penalty | Both datasets | — |
| `bc` | Behavior cloning with the shared policy network | Both datasets | — |
| `no_source_prior` | Quality-only control with source-prior information removed | Mixture | — |
| `return_only` | Quality-only control with return-derived weights | Mixture | — |
| `reference_iql` | Official-source IQL equation port to PyTorch (IQL-Ref) | Both datasets | — |

This is a custom persistent-reward Kitchen task, not a D4RL normalized-score benchmark reproduction. Do not compare the raw returns in this repository directly with published D4RL scores. The registered evaluation environment has 81 observation coordinates, 9 action coordinates, and a 280-step evaluation horizon. It counts currently completed goals on each transition and does not remove completed goals or terminate on success.

The `recorded_run_name` field is only a provenance link to the retained experiment logs. Public commands and outputs use the unified run names. Learning hyperparameters, budgets, and validation settings were checked against all 60 recorded configurations during consolidation. The declared save interval remains 5,000 for method comparisons and 25,000 for ablations; the trainer additionally saves resumable state at initialization and after validation.

## Run from the repository root

Install dependencies and import/check data first as described in the main README. The help and dry-run commands need only Python and do not touch the datasets or initialize CUDA:

```bash
python scripts/run_experiments.py --help
python scripts/run_experiments.py --preset all --dry-run
```

Start with a small engineering check. `--smoke` selects the lowest training seed for each chosen method/dataset/block and runs 100 updates by default (20 is also accepted):

```bash
python scripts/run_experiments.py --preset comparisons --methods baseline full bc reference_iql --datasets mixture --smoke --jobs 2 --threads 1
python scripts/status.py --smoke
```

Smoke validation resets are 12000–12001 and its final evaluation resets are 12002–12003. Its checkpoints, manifests and metrics are isolated under `runs/smoke`; smoke results must not enter the formal result tables.

Run the full study or a filtered selection:

```bash
python scripts/run_experiments.py --preset all --jobs 6 --threads 1
python scripts/run_experiments.py --preset comparisons --datasets mixture --methods full quality_only --seeds 3 4 5 --jobs 3 --threads 1
python scripts/run_one.py --run comparisons_low_full_s3 --threads 1
python scripts/status.py --preset all
```

`--jobs` bounds independent Python workers; `--threads` bounds each worker's CPU numerical threads. All workers use the visible CUDA device. Select an allocated GPU with `CUDA_VISIBLE_DEVICES` before launching. Six workers with one CPU thread each is a practical starting point for the previously used 32 GB GPU; tune for actual GPU use, CPU allocation, host RAM, and evaluation throughput. Data and replay buffers are loaded by each worker, so GPU memory alone does not determine a safe concurrency level.

Each run has its own OS lock. Different run names can proceed concurrently in the same project, while duplicate launches of the same run are rejected. A failed worker is recorded and does not cancel other queued runs. The scheduler exits nonzero if any run failed; inspect its log and rerun the command after correcting the cause.

## Resume and evidence

Reissuing the same command resumes interrupted runs from the last atomic checkpoint and skips completed runs only after artifact hashes pass. Model, optimizer, random-generator and best-checkpoint state are restored; metrics beyond the resumed step are rolled back. `--no-resume` instead rejects an existing incomplete run. Code, configuration, dataset checksums, dependency versions, GPU identity and thread count are checked before resuming. Keep these fixed for a recorded run; put a deliberately changed experiment in a separate project copy with its own protocol.

A scheduler interrupt stops new launches and terminates active workers. The last completed atomic checkpoint remains usable. `status.py` displays the last recorded state; a hard kill may leave a `training` label until the next resume. Check the process and log before interpreting that label. Do not delete an active lock file. OS locks release when the holding process exits; the remaining file is harmless.

| Location | Contents |
|---|---|
| `runs/checkpoints` | Validation-selected and resumable endpoint models |
| `runs/metrics` | Per-run JSONL training/validation records |
| `runs/configs` | Exact runtime configurations |
| `runs/audit/runs` | Run identity, checksums, final test episodes and status |
| `runs/audit/sessions` | Scheduler outcomes |
| `runs/logs` | Worker standard output and errors |
| `runs/smoke` | Isolated engineering check outputs |
| `runs/exports` | Collected evidence archives |

After workers finish, export evidence:

```bash
python scripts/collect_results.py
```

Collection acquires the planned run locks and fails if training is still active, preventing a torn snapshot. It verifies the code/data identity and completed artifacts, exports both selected and endpoint checkpoints, and includes a per-run summary CSV and SHA-256 inventory. Incomplete runs are listed explicitly. The archive includes model evidence and can be large; it excludes NPZ/HDF5 datasets and is not a GitHub source release. Use `--include-smoke` only for installation diagnosis. All generated output is ignored by Git.

## Checks without training

```bash
python -m pytest -q
```

Runner tests exercise real process concurrency, per-run lock contention and crash release, completion integrity, export boundaries, and the recorded protocol/configuration agreement. They also verify that help and dry-run work without installed scientific packages. These checks do not re-run the 60 experiments or establish equivalence of new GPU trajectories to stored results.

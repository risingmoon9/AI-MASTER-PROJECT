# HQA-IQL

Quality-aware offline reinforcement learning on a shared four-task Kitchen formulation. This repository contains the training implementation, data preparation and validation tools, parallel experiment runner, and numerical records for one study comprising **60 trained models**.

[中文说明](README_CN.md) · [Methods](docs/METHOD.md) · [Data](docs/DATA.md) · [Experiment protocol](docs/EXPERIMENTS.md) · [Recorded results](results/README.md) · [Validation](docs/VALIDATION.md)

The study compares quality weighting and adaptive behavior-cloning regularization with an internal IQL baseline (IQL-M), behavior cloning, constant regularization, quality-signal controls, and an official-source IQL PyTorch port (IQL-Ref). The environment uses a custom shared observation/reward definition. Returns are **raw returns on this task**, not normalized D4RL benchmark scores.

## Installation

Use a source checkout with Python 3.10–3.12. Linux with an NVIDIA GPU is the training target. PyTorch 2.7.1 and the numerical/simulator versions in `requirements.txt` preserve the recorded environment. The scripts use files in this checkout; install in editable mode, rather than copying the Python package elsewhere.

On AutoDL or another Linux GPU host, from the repository root:

```bash
bash autodl/setup.sh
source autodl/env.sh
```

The default PyTorch build is CUDA 11.8. If required by the host, select an available build explicitly, for example `TORCH_CUDA=cu128 bash autodl/setup.sh`. `TORCH_CUDA=cpu` installs a CPU diagnostic environment; the formal training launcher requires CUDA. The driver version reported by `nvidia-smi` is not the PyTorch build selector.

For a manual CPU development installation:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pytest
```

On Windows, activate with `.venv\Scripts\Activate.ps1`; the shell wrappers are for Linux. Unit tests use synthetic fixtures and do not require the study datasets. CI checks the CPU tests and experiment planning; it does not retrain models.

## Data

Datasets, checkpoints, and raw result archives are intentionally stored separately from the repository. The exact prepared study data are identified by `DATA_SHA256.json`. For an existing local prepared release:

```bash
python scripts/import_data.py --archive /path/to/HQA_IQL_AutoDL_Supplement_20261017.zip
python scripts/preflight.py
```

Alternatively, use `--source-dir /path/to/prepared_data`. Only the named data assets are imported, with checksum verification. The archive name above is a compatibility input; the experiment interface uses the unified study protocol. See [Data](docs/DATA.md) for the directory layout, reconstruction procedure, provenance and access limitations. The repository does not provide a public dataset download URL.

Preflight checks both datasets, quality overlays, all 60 configurations, device arithmetic, and the actual 280-step simulator reward/schema. It uses engineering reset seed 12000. `--allow-cpu` enables simulator diagnostics without a GPU. `MINARI_DATASETS_PATH` is respected when explicitly set; it must point to the matching local Minari cache.

## Run experiments

Inspect the plan without loading datasets or PyTorch:

```bash
python scripts/run_experiments.py --preset all --dry-run
```

Run a short engineering check before a full reproduction:

```bash
python scripts/run_experiments.py --preset comparisons --methods baseline full --datasets mixture --smoke --jobs 2 --threads 1
```

Reproduce the complete study, or select only the required runs:

```bash
python scripts/run_experiments.py --preset all --jobs 6 --threads 1
python scripts/run_experiments.py --preset comparisons --methods full reference_iql --datasets mixture --seeds 3 4 5 --jobs 3
python scripts/status.py --preset all
python scripts/collect_results.py
```

Each worker trains one method/seed and holds its own operating-system lock. Different runs can execute concurrently in the same checkout. `--jobs` controls simultaneous workers and `--threads` controls each worker's CPU threads. Adjust concurrency to the host's GPU memory, RAM and CPU capacity. Shell wrappers are available as `bash autodl/run.sh ...`, `status.sh` and `collect.sh`.

Repeating the same command resumes incomplete runs from atomic checkpoints and verifies completed outputs before skipping them. A changed code/config/data identity is rejected to prevent mixing experiments. Preserve the original checkout for resumption; start a separate checkout for changed experiments. Smoke runs use separate output directories and engineering seeds and are excluded from formal summaries. See [Experiment protocol](docs/EXPERIMENTS.md) for output files, interruption behavior and collection options.

## Experimental design

| Block | Trained models | Training seeds | Validation reset seeds | Final test reset seeds |
|---|---:|---|---|---|
| Method comparisons | 42 | 3, 4, 5 | 11000–11019 | 30000–30049 |
| Component ablations | 18 | 0, 1, 2 | 10000–10019 | 20000–20049 |

Every model trains for 200,000 updates with batch size 256. Validation occurs every 5,000 updates over 20 episodes. The first strict maximum validation return selects the reported best checkpoint. Both the selected checkpoint and the fixed training endpoint are tested over 50 episodes. Summaries use the mean and **sample standard deviation across three training seeds**, separately for each block. The two blocks must not be pooled into a six-seed estimate because their reset sets differ.

Two ablation test resets (20000 and 20001) were also used in engineering checks in the recorded study; no adjustment based on their outcomes is documented. This limitation remains part of the protocol. New smoke checks use 12000–12003.

## Recorded findings

Selected-checkpoint raw return, mean ± sample SD across three training seeds, from the method comparisons:

| Dataset | IQL-M | HQA-IQL | Quality only | BC | IQL-Ref |
|---|---:|---:|---:|---:|---:|
| Mixture | 333.94 ± 41.26 | 390.01 ± 50.32 | 396.87 ± 60.65 | 347.98 ± 71.41 | 393.65 ± 24.62 |
| Complete | 317.36 ± 57.25 | 299.02 ± 135.08 | 309.31 ± 46.33 | 302.95 ± 112.91 | 542.89 ± 56.68 |

The results support a dataset-dependent effect, not general superiority of HQA-IQL. Endpoint deterioration is substantial: 40 of the 42 comparison models have lower endpoint return than selected-checkpoint return, and all 42 endpoints have zero all-four-task success. Endpoint values and all controls/ablations are included in [the complete result tables](results/README.md).

Recompute the committed group summaries from the per-model CSV without training or installing dependencies:

```bash
python scripts/summarize_recorded_results.py
```

## Repository layout

| Path | Contents |
|---|---|
| `hqa_orl/` | Actors, critics, IQL-M/HQA-IQL, BC, IQL-Ref, replay, training and evaluation |
| `scripts/` | Data import/build/validation, preflight, scheduling, status and collection |
| `configs/experiments/` | 60 frozen model configurations |
| `protocol/` | Experiment definitions and recorded episode selection |
| `reference/` | Pinned upstream IQL source and its MIT license |
| `results/` | Small, recorded numerical tables and provenance |
| `tests/` | Data, algorithm and execution-integrity tests |
| `runs/` | Generated logs, models, audits and exports; ignored by Git |

Only the documented Kitchen study is a validated reproduction target. Historical experiments with incompatible observations/rewards and unused implementations are outside this release. Repository consolidation changes packaging, validation and execution interfaces; the committed results are from the completed experiments, not a fresh training run of this checkout.

## Attribution and licensing

The IQL reference follows [Kostrikov et al., Implicit Q-Learning](https://arxiv.org/abs/2110.06169) and pinned source from [ikostrikov/implicit_q_learning](https://github.com/ikostrikov/implicit_q_learning). See [third-party notices](THIRD_PARTY_NOTICES.md) for the exact commit and upstream license. The upstream MIT license applies to the upstream files; no project-wide license has been selected for the original project code. Dataset terms remain those of their respective providers.

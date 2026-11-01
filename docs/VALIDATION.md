# Repository validation

Validation date: 2026-11-01. The source release consolidates the completed study;
it does not contain a new full-budget training experiment.

## Checks performed

| Check | Result |
|---|---|
| All 60 configurations against the recorded releases | Learning hyperparameters, budgets and evaluation settings preserved |
| Frozen data import | All six SHA-256 checks passed |
| Real training-buffer validation | 124,019 mixture and 4,209 complete transitions passed |
| Reconstruction from local Minari sources | All 14 non-quality arrays matched both frozen buffers exactly |
| Experiment configuration and overlay validation | All 60 configurations and both quality overlays passed |
| Real simulator preflight | Observation/action schema, persistent reward, flags and 280-step horizon passed |
| Test suite with imported data | 31 tests passed |
| Editable installation | Package metadata and editable build succeeded |
| Python compatibility check | All 30 project/test Python files parsed with Python 3.10 grammar |
| Shell entry points | All five shell scripts passed `bash -n` |
| Upstream reference integrity | All 13 vendored files matched the upstream hash manifest |
| Git staging boundary | Datasets, generated models, logs and caches excluded |
| Completed-run re-entry | HQA-IQL and IQL-Ref skipped after artifact verification |
| Real result export round trip | 12 smoke checkpoints and every inventory hash verified; datasets excluded |

GPU integration used Windows 11, Python 3.12.7, PyTorch 2.7.1+cu118 and an NVIDIA
GeForce RTX 3050 Ti Laptop GPU, with the pinned numerical/simulator versions.
Six mixture-data engineering runs completed with two concurrent workers:
IQL-M, HQA-IQL, BC, IQL-Ref, no-source-prior and return-only controls. Each used
20 updates, two validation episodes and two test episodes for each checkpoint,
with engineering seeds 12000–12003. These checks verify execution and data flow;
they are not performance results. CPU tests also exercise exact interrupted
recovery, lock contention, crash release and export integrity.

The validation environment used an isolated virtual environment with existing
system numerical libraries available; it was not a fresh Linux installation.
The GitHub Actions workflow is provided for Linux CPU tests but was not remotely
executed during packaging. Full AutoDL GPU training was not repeated. Recomputed
quality/clustering arrays were not claimed to be byte-identical to frozen data.

## Recorded numerical evidence

The committed tables were independently checked against JSON/JSONL records in
both original experiment result archives. This verification read numerical
records without loading model checkpoints or executing archived code:

- 120 selected/endpoint evaluation rows matched recomputation from 6,000 test episodes.
- 2,400 validation rows matched 48,000 validation episode records.
- All 40 group means and sample standard deviations matched.
- The first strict validation maximum selected the recorded checkpoint in all 60 runs.
- Endpoint steps, 280-step episode lengths and reset-seed sequences matched the protocol.

Private absolute paths and student/author declarations are not included in the
public source release. Source archives, raw logs, datasets and models remain
separate local assets. `SOURCE_MANIFEST.sha256` records the released file bytes.

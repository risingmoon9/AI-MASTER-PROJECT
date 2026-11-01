# Third-party notices

## Official Implicit Q-Learning implementation

`reference/upstream/` contains selected unmodified files from
[ikostrikov/implicit_q_learning](https://github.com/ikostrikov/implicit_q_learning),
commit `09d700248117881a75cb21f0adb95c6c8a694cb2`.

Copyright (c) 2021 Ilya Kostrikov, Ashvin Nair, Sergey Levine.
These files are distributed under the MIT License, reproduced in full at
[`reference/upstream/LICENSE`](reference/upstream/LICENSE).
Exact source URLs, byte counts, and SHA-256 hashes are recorded in
[`reference/UPSTREAM_MANIFEST.json`](reference/UPSTREAM_MANIFEST.json).

`hqa_orl/reference_iql.py` is derived from the equations and implementation
conventions of that source, reimplemented in PyTorch, and retains attribution
in its module header. See [`reference/IQL_REFERENCE.md`](reference/IQL_REFERENCE.md)
for its scope and differences. The vendored JAX files are reference material;
the training commands do not execute them.

## Dependencies and datasets

Python dependencies are installed separately and retain their respective
licenses. Dataset archives, Minari environment data, checkpoints, and raw
evaluation logs are not included in this source repository. Their original
distribution and use terms continue to apply to separately obtained files.

This notice does not assign a project-wide license to the author's remaining
code. No such license has been selected for this prepared repository.

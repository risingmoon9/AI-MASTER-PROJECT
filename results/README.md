# Recorded results

These are numerical records from the completed Kitchen study, extracted from its audited experiment outputs. No model was retrained while assembling this repository. The original private archives and thesis documents are not distributed here.

| File | Contents |
|---|---|
| `recorded_runs.csv` | 120 final evaluations: selected and endpoint checkpoints for 60 trained models |
| `validation_curves.csv` | 2,400 evaluations: 40 validation points for each model |
| `summary.csv` | 40 dataset/method/checkpoint groups, separated by comparison block |
| `provenance.json` | Hashes of the audited statistical extracts used for publication |

Run `python scripts/summarize_recorded_results.py` to recompute `summary.csv` values from `recorded_runs.csv`; the default output is `runs/exports/recorded_summary.csv`. New training outputs are collected separately with `scripts/collect_results.py` and never overwrite these recorded tables.

The statistical unit is a trained model. Means and sample standard deviations (ddof=1) are computed across three training seeds. The 50 episodes per checkpoint share a trained policy and are not 50 independent training replicates. The two experiment blocks have distinct reset sets; their seed groups are not pooled.

`validation_selected` denotes the first strict maximum of mean validation raw return. `endpoint` denotes update 200,000, whether or not it performs well. `return_mean` is raw cumulative persistent-task reward over 280 steps. `task_completion_percent` is the episode-average fraction of the four tasks ever completed, multiplied by 100. `all_tasks_success_percent` is the fraction of episodes completing all four tasks, multiplied by 100; positive return alone does not count as success.

Interpretation must include both checkpoint types. In the 42-model method comparisons, 40 endpoints have lower return than their selected checkpoints, two are equal, and all endpoints have zero all-four-task success. HQA-IQL does not consistently outperform the other controls across both datasets. IQL-Ref is an official-source equation port evaluated on this custom task, not a reproduction of a published D4RL benchmark score.

The [protocol](../protocol/experiments.json) retains the mapping from clean run names to identifiers in the recorded archives. That mapping is provenance only. Detailed experimental settings and the ablation reset-exposure limitation are described in [the experiment documentation](../docs/EXPERIMENTS.md).

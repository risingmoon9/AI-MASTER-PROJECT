# Methods and implemented objectives

This project studies offline learning on a common Kitchen task. Supported
agents are IQL-M, HQA-IQL and its component controls, behavior cloning (BC), and
IQL-Ref. The files in `configs/experiments/`, rather than dataclass defaults,
specify the completed study.

## Common task

The observation is 81-dimensional: achieved goals (11), desired goals (11), and
physical state (59), with dictionary keys flattened in sorted order. The four
tasks are kettle, light switch, microwave, and slide cabinet; actions have nine
dimensions. Evaluation lasts at most 280 steps. Reward counts tasks whose goal
distance is below 0.3 after each step, so a task can contribute repeatedly.
Task removal and termination upon completion are disabled. This return is not
a D4RL normalized benchmark score.

The replay buffer preserves full source trajectories, including trajectories
longer than the evaluation horizon and repeated episode occurrences. Timeouts
are distinct from true terminations; all true terminal masks in these datasets
are zero, so the value target bootstraps across timeouts. See [DATA.md](DATA.md)
for preprocessing, source selection, and data verification.

## IQL-M and HQA-IQL

Let `w` be a positive offline quality weight, `Qbar` the minimum of the two target
critics, and `A = Qbar(s,a) - V(s)`. For IQL-M, set `w = 1`. The objectives use
minibatch sums and epsilon `1e-6`:

```text
V loss = sum(w * expectile(Qbar(s,a) - V(s))) / (sum(w) + epsilon)
Q target = r + 0.99 * (1 - done) * V(next_s)
Q loss = sum(w * ((Q1 - target)^2 + (Q2 - target)^2)) / (sum(w) + epsilon)
omega = min(exp(3 * A), 10)
actor loss = sum(w * omega * negative_log_likelihood) / (sum(w * omega) + epsilon)
BC term = mean(lambda(s,a) * negative_log_likelihood)
```

The full method adds the BC term with
`lambda = clip(0.2 * (1 + uncertainty - 0.5 * support), 0, 2)`.
Since support equals `1 - uncertainty` in these buffers, this is also
`clip(0.2 * (0.5 + 1.5 * uncertainty), 0, 2)`. The BC term is a batch mean;
normalizing by the coefficient sum would cancel its intended scale.

Each update executes V, Q, actor, then target Q. Both critic targets and actor
advantages use the updated V; the actor uses the previous target Q. Target Q
uses a Polyak coefficient of 0.005. The actor has two 256-unit ReLU hidden
layers, a state-dependent standard deviation, and a tanh-transformed Gaussian
density with its Jacobian correction. Q and V have the same hidden dimensions.
Observations are standardized using the entire training buffer. Actor learning
rate is `1e-4`; Q and V rates are `3e-4`; gradient norms are clipped at 10.

| Method key | Quality weighting | Added BC term |
|---|---|---|
| `baseline` | None | None |
| `full` | Full quality score | Adaptive coefficient |
| `quality_only` | Full quality score | None |
| `adaptive_bc_only` | None | Adaptive coefficient |
| `constant_bc` | Full quality score | Constant coefficient |
| `no_source_prior` | Quality score without source prior | None |
| `return_only` | Episode-return-only weight | None |

The constant coefficients are the complete-buffer mean adaptive coefficient:
0.11832241827748832 for the mixture and 0.14510062652894284 for complete data.
They are fixed before training, rather than fitted from test returns.

## Offline quality signals

The full trajectory score combines normalized episode return (0.55), source
prior (0.30), and a return-threshold proxy (0.15). This proxy is not environment
task success. Transition quality combines normalized return-to-go (0.45),
reward (0.20), support (0.25), and negative uncertainty (0.10). Support comes
from state-action k-nearest-neighbor distance; uncertainty is its complement.

Episode descriptors contain return, length, mean action norm, action variance,
observation displacement, mean reward, maximum reward, and source prior.
Standardized descriptors are clustered into at most five groups. The final
weight is a sigmoid of standardized trajectory plus transition quality,
multiplied by an inverse-square-root transition-count cluster factor. Raw
weights are clipped to `[0.05, 5]` and divided by their complete-buffer mean
plus epsilon. There is no second clip, so the normalized bounds differ from
the raw bounds. All transition-level moments preserve episode length and
sampling multiplicity.

`no_source_prior` removes the source term without rescaling the other score
coefficients and reclusters seven non-source descriptors. `return_only` uses
only the standardized repeated episode return passed through a sigmoid and
the same clipping/normalization convention. Both controls use the quality-only
actor objective; data overlays may change only declared quality diagnostics,
never observations, actions, rewards, or termination masks.

## Behavior cloning

BC minimizes the ordinary batch mean negative log-likelihood of data actions.
It shares the IQL-M actor architecture, observation normalization, actor
learning rate, and gradient clipping. It has no value or critic network and
does not read quality weights, support, or uncertainty in its objective.

## IQL-Ref

IQL-Ref is a PyTorch port of the pinned official IQL equations. It uses raw
observations, a global learned log standard deviation, a tanh-bounded mean
inside an otherwise untransformed diagonal Normal, and actor dropout 0.1.
Its advantage temperature is 0.5, weight cap is 100, and its actor objective
is a batch mean, without normalization by the sum of advantage weights.
All three initial learning rates are `3e-4`; actor learning rate follows cosine
decay over 200,000 updates. There is no gradient clipping or added quality/BC
term. The update order is V, actor, Q, then target Q.

These differences mean IQL-Ref versus IQL-M is not a single-factor ablation.
The implementation has equation, analytic update, and recovery tests; it has
not established full JAX numerical equivalence or original D4RL benchmark
reproduction. See [the reference specification](../reference/IQL_REFERENCE.md).

## Recovery and numerical checks

Training saves model parameters, optimizer states, observation statistics,
update count, Python/NumPy/PyTorch RNG state, and the exact selected-checkpoint
bytes. A committed metrics offset and hash allow recovery to remove an
uncommitted log tail. Configurations, dataset hashes, device type, and PyTorch
version must match on resume. Concurrent execution is coordinated per run by
the experiment runner.

Tests cover IQL-Ref's independent analytic Adam update, objective scaling,
BC independence from quality/value fields, and exact interrupted-versus-
uninterrupted CPU training for all agent families. These are implementation
checks, not new performance experiments.

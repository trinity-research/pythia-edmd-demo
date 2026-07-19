# Analysis Protocol (fixed before model execution)

Protocol date: 2026-07-18  
Protocol version: 1.2

## Question

Can a fixed EDMD procedure recover genuine temporal structure from hidden-state
trajectories of a small public language model, rather than merely fitting an
attractive eigenspectrum in sample?

## Public subject

- Repository: `EleutherAI/pythia-14m`
- License reported by upstream: Apache-2.0
- Revisions and exact commits:
  - `step1000`: `5b020995bfc7aee2931b0f35bd70cf7ee8b1db62`
  - `step10000`: `b9935f34c34c4bddaa99bed4c2ed3fc8e67c7504`
  - `step143000`: `f1545025bb394553a7f4e547db0874886f05ef9c`

No private weights, private training data, model-specific analysis code, or
undisclosed activation transforms are used.

## Fixed analysis choices

- Random seed: `20260718`
- Prompt split: the explicit `train` and `test` labels in `prompts.json`
- Maximum tokens per prompt: `128`
- Burn-in: first `4` token positions excluded from snapshot pairs
- Layer rule: zero-based transformer block `floor((L - 1) / 2)`
- State reduction: PCA rank `16`, fit only on training snapshots
- EDMD observables: coordinate-wise `[z, z^2, tanh(z)]`
- Dictionary centering/scaling: fit only on training snapshots
- EDMD ridge multiplier: `0.001`, applied to the mean diagonal of the training
  Gram matrix so its scale is dimensionless
- Shuffled-target repetitions: `50`
- Evaluation horizons: `1`, `2`, `4`, and `8` tokens

All prompt splits are by whole prompt. Adjacent token pairs from a held-out
prompt never enter PCA, dictionary normalization, or EDMD fitting.

## Primary hypothesis and decision rule

At each checkpoint, the real-pair EDMD model passes only if both conditions hold:

1. held-out one-step normalized RMSE is below `1.0`, where `1.0` is the error of
   predicting the training-target mean; and
2. its error is lower than the shuffled-target controls at a one-sided empirical
   randomization-test level of `p <= 0.05`, using the add-one correction.

The number of passing checkpoints is reported. A partial or failed result is
retained and reported; no checkpoint or prompt category is discarded after the
result is observed.

## Secondary, exploratory measurements

- held-out R-squared;
- multi-step normalized RMSE at 2, 4, and 8 tokens;
- spectral radius;
- fraction of fitted eigenvalues inside the unit circle;
- count of stable persistent modes with `0.95 <= |lambda| < 1`;
- median and maximum half-life among stable modes; and
- PCA variance retained at the fixed rank.

No monotonic checkpoint trend is predicted in advance. Spectral differences are
descriptive unless independently replicated.

For a stable discrete-time eigenvalue with `0 < |lambda| < 1`, half-life is
defined as `log(0.5) / log(|lambda|)` token transitions.

## Interpretation boundary

Passing the primary gate supports the narrow statement that this EDMD procedure
captures out-of-sample temporal structure in the chosen public model's hidden
activations. It does not establish that the model is governed by a global linear
system or that individual modes correspond to semantic memory.

Checkpoint-dependent persistent-mode or half-life measurements may motivate a
memory hypothesis. Calling any feature a *memory bottleneck* requires an
independent behavioral or causal intervention showing that the feature limits
retention. This experiment does not make that stronger claim.

## Amendment history

### Version 1.1 — post-run robustness amendment, 2026-07-18

After the version 1.0 primary result was viewed, two additional reports were
specified before inspecting their values:

1. an identity/persistence comparator, `z_(t+1) = z_t`; and
2. held-out one-step metrics broken out by the pre-existing prompt categories.

No checkpoint, prompt, split, layer rule, PCA rank, dictionary, ridge value,
shuffle count, hypothesis, or original primary decision rule was changed. The
persistence comparison is a post-run robustness result, and the category
breakdown is descriptive. They are not represented as part of the original
pre-specification.

### Version 1.2 — reporting and sensitivity amendment, 2026-07-18

After an independent review of the version 1.1 outputs, the rendered
interpretation was extended with:

1. per-category persistence values alongside the per-category EDMD values;
2. a leave-one-prompt-out sensitivity check of the aggregate
   EDMD-versus-persistence comparison;
3. explicit listing of any held-out category whose EDMD error is at or above
   the mean-prediction baseline;
4. a cross-checkpoint spectral comparability caveat: each checkpoint's PCA
   basis is fit independently and retained variance differs across
   checkpoints, so spectral differences remain descriptive; and
5. documentation that the tokenizer is loaded from the `step143000` commit;
   its tokenizer files are byte-identical across the three pinned commits.

Provenance: items 1 and 3 re-render values already computed by the version 1.1
code, and item 2's outcome was known from the external review before this
amendment was written. None of these results are represented as pre-specified.
No checkpoint, prompt, split, layer rule, PCA rank, dictionary, ridge value,
shuffle count, hypothesis, or original primary decision rule was changed.

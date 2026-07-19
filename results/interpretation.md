# Result and bounded interpretation

The originally pre-specified primary gate passed at **3 of 3** analyzed public checkpoints.

| Checkpoint | EDMD nRMSE | Persistence nRMSE | Shuffled nRMSE (mean +/- SD) | p | Original gate |
| --- | ---: | ---: | ---: | ---: | --- |
| step1000 | 0.8578 | 1.0229 | 1.0390 +/- 0.0163 | 0.0196 | pass |
| step10000 | 0.7538 | 0.9338 | 1.0383 +/- 0.0147 | 0.0196 | pass |
| step143000 | 0.8711 | 0.9227 | 1.0374 +/- 0.0132 | 0.0196 | pass |

Normalized RMSE below 1.0 beats the training-target mean predictor. The one-sided empirical p-value asks how often a shuffled-target fit achieved an error at least as low as the real-pair fit.

At every checkpoint, 0 of 50 shuffled-target fits matched the real fit; the add-one correction gives p = 1/51 ~= 0.0196, the smallest value this test can report.

## Post-run persistence robustness (amendments v1.1 and v1.2)

Across all held-out prompts, EDMD outperformed the identity/persistence predictor at **3 of 3** checkpoints. This comparison was specified only after the original primary result was viewed. The advantage is category-dependent, as the breakdown below shows.

### Held-out category breakdown (descriptive)

| Checkpoint | Periodic EDMD | Periodic persist. | Progressive EDMD | Progressive persist. | Prose EDMD | Prose persist. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| step1000 | 0.7618 | 0.7436 | 0.7854 | 0.6386 | 0.9520 | 1.3439 |
| step10000 | 0.7686 | 0.6922 | 0.6659 | 0.5854 | 0.8223 | 1.2745 |
| step143000 | 1.0332 | 0.7638 | 0.6701 | 0.6426 | 0.8237 | 1.2561 |

Checkpoints where EDMD beat persistence within each category -- periodic: 0 of 3; progressive: 0 of 3; prose: 3 of 3.

Held-out categories with EDMD error at or above the mean-prediction baseline: step143000 periodic (1.0332).

### Leave-one-prompt-out sensitivity (v1.2)

- step1000: the aggregate comparison is unchanged under all 6 single-prompt removals.
- step10000: the aggregate comparison is unchanged under all 6 single-prompt removals.
- step143000: the aggregate EDMD-versus-persistence comparison reverses when any one of 2 of 6 held-out prompts is removed (test-prose-observatory, test-prose-bicycle).

## Spectral comparability caveat

Each checkpoint's rank-16 PCA basis is fit independently, and the retained training-state variance differs across checkpoints (66.0%, 76.4%, 89.9%). Cross-checkpoint spectral comparisons therefore conflate changes in the underlying dynamics with changes in the captured subspace, and they remain descriptive.

Under the fixed protocol, real temporal pairing carried out-of-sample predictive structure at every analyzed checkpoint.

## Defensible conclusion

A fixed, disclosed EDMD pipeline recovered held-out one-step temporal structure from the hidden activations of 3 of 3 public Pythia-14m training checkpoints, outperforming a train-mean predictor and 50 shuffled-pairing controls at each passing checkpoint. On held-out prose it also outperformed state persistence at 3 of 3 checkpoints; the category tallies above show where persistence was better. Cross-checkpoint spectral differences remain descriptive.

The predictive-control result above is the evidentiary gate, not the visual appearance of the spectrum.

## What this does not prove

This result does not show that the public transformer implements a linear recurrence, that an eigenmode has a particular semantic meaning, or that a spectral statistic alone identifies a causal memory bottleneck. It does not validate claims about another model.

A bottleneck claim would need a separate behavioral or causal test: predict a specific retention failure from the spectrum, intervene on the implicated mechanism, and show that the predicted failure changes while suitable controls do not.

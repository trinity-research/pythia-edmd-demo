# Public Checkpoint EDMD Demonstration

This is a standalone, reproducible test of whether Extended Dynamic Mode
Decomposition (EDMD) can recover out-of-sample structure from the hidden-state
trajectories of a small, unrelated, publicly downloadable language model.

The subject is
[`EleutherAI/pythia-14m`](https://huggingface.co/EleutherAI/pythia-14m), an
Apache-2.0 model published with many intermediate training checkpoints. Three
exact public commits are analyzed:

| Published revision | Exact commit |
| --- | --- |
| `step1000` | `5b020995bfc7aee2931b0f35bd70cf7ee8b1db62` |
| `step10000` | `b9935f34c34c4bddaa99bed4c2ed3fc8e67c7504` |
| `step143000` | `f1545025bb394553a7f4e547db0874886f05ef9c` |

The exact commits, prompt corpus, analysis settings, controls, and claim limits
are fixed in [PROTOCOL.md](PROTOCOL.md). The script records file hashes and
software versions in the result manifest.

## What this tests

For every checkpoint, the script:

1. runs the same disclosed prompt corpus through a predeclared middle block;
2. forms within-prompt snapshot pairs, `h_t -> h_(t+1)`;
3. learns a 16-dimensional PCA state using training prompts only;
4. fits ridge EDMD with the fixed dictionary `[z, z^2, tanh(z)]`;
5. evaluates one- and multi-step prediction on entirely held-out prompts; and
6. compares the real temporal pairing with a mean predictor, a persistence
   predictor (`z_(t+1) = z_t`), and 50 shuffled-target EDMD controls.

The primary gate is passed at a checkpoint only when the held-out EDMD error is
below the mean-prediction baseline and better than the shuffled-control
distribution at a one-sided randomization-test level of `p <= 0.05`.

The persistence comparator, held-out category breakdown, and a
leave-one-prompt-out sensitivity check were added as dated post-run
amendments (versions 1.1 and 1.2) after the original primary result was
viewed. This provenance is recorded in `PROTOCOL.md`; none of them is
presented as part of the original preregistration.

All checkpoints are tokenized with the tokenizer from the `step143000`
commit; the tokenizer files are byte-identical across the three pinned
commits.

## What this cannot establish

This experiment can show that a fixed EDMD procedure measures reproducible,
out-of-sample local dynamics in public hidden activations, and that a fitted
spectrum can be reported at each public training checkpoint. Because each
checkpoint's PCA basis is fit independently and retains a different share of
variance, cross-checkpoint spectral differences are descriptive only.

It does **not** show that Pythia implements a linear recurrence, that every
eigenmode has a semantic interpretation, or that a spectral statistic by itself
identifies a causal memory bottleneck. It also does not validate claims about
any other model. Those would require separate evidence.

## Possible extensions

None of these are results. They are directions anyone can take this fixed
pipeline, using only public models:

- **More checkpoints and scales.** The Pythia suite publishes many more
  checkpoints and model sizes. Running the identical protocol across them
  would show how measured local dynamics change with training and scale.
- **Layer sweeps.** The protocol fixes one middle block. Repeating the
  analysis per block would map where in the network temporal structure is
  most linearly predictable.
- **Shared-subspace spectral comparison.** Fitting one common PCA basis on
  pooled activations from all checkpoints would remove the subspace confound
  noted in the results and make cross-checkpoint spectral differences
  directly comparable.
- **Richer dictionaries and ranks.** The dictionary and PCA rank are fixed
  here for preregistration. A held-out sweep would show how much additional
  predictive structure a larger observable basis recovers.
- **Behavioral linkage.** The protocol's interpretation boundary names the
  missing step: predict a specific retention behavior from a spectral
  feature, then test that prediction against the model's outputs. That
  experiment is what would turn a descriptive spectrum into evidence about
  memory.
- **Other architectures.** Nothing in the pipeline is transformer-specific.
  It applies to any model that exposes per-token hidden states, including
  state-space models, RNNs, and hybrids.

If you extend this, preregister your settings the way `PROTOCOL.md` does, and
keep the shuffled-pairing and persistence controls: the predictive gate, not
the appearance of the spectrum, is the evidence.

## Reproduce

Python 3.12 and a CPU are sufficient. The first run downloads exactly three
small model checkpoints from Hugging Face.

```powershell
python -m pip install -r requirements.txt
python analyze.py
```

To rerun after the exact commits are cached:

```powershell
python analyze.py --offline
```

Outputs were verified byte-identical on the Windows x64 environment recorded
in `results/run_manifest.json`. Other operating systems or BLAS builds may
differ in the trailing digits of reported metrics.

The script writes:

- `results/interpretation.md` — plain-language result with explicit claim limits;
- `results/checkpoint_summary.csv` — the principal numerical table;
- `results/summary.json` — full metrics and fitted eigenvalues;
- `results/run_manifest.json` — model commits, hashes, environment, and settings;
- `results/edmd_validation.png` — held-out controls and checkpoint spectra; and
- `results/pca_variance.png` — retained-state variance diagnostic.

## Relation to prior work

None of the mathematics in this package is new, and close relatives of this
analysis exist in several fields. The contribution is a specific combination
and how it is gated, not the method. Each strand below starts with a plain
statement of what that field did, then the citation.

- **Koopman theory and dynamic mode decomposition.** Fluid dynamics and
  control developed the core recipe used here: photograph a complicated
  system over time, view the photos through a dictionary of extra lenses,
  and extract simple modes with measurable decay rates. This package is a
  direct application of EDMD: M. O. Williams, I. G. Kevrekidis, and
  C. W. Rowley,
  [A Data-Driven Approximation of the Koopman Operator: Extending Dynamic
  Mode Decomposition](https://arxiv.org/abs/1408.4408).
- **Recurrent networks as dynamical systems.** Machine learning has long
  reverse-engineered trained recurrent networks through their fixed points
  and low-dimensional dynamics: D. Sussillo and O. Barak, *Opening the
  Black Box: Low-Dimensional Dynamics in High-Dimensional Recurrent Neural
  Networks*, Neural Computation, 2013.
- **Neural population dynamics.** Neuroscience routinely treats recordings
  of many neurons as one low-dimensional dynamical system with rotating and
  decaying modes; this package does to a language model roughly what that
  line does to motor cortex: M. M. Churchland et al., *Neural population
  dynamics during reaching*, Nature, 2012.
- **State-space language models.** The HiPPO/S4/Mamba line builds sequence
  models whose memory timescales are set by eigenvalues *by design*. The
  same eigenvalue-to-timescale dictionary is used here in reverse, as an
  outside measurement of a model that was not built that way: A. Gu et al.,
  [HiPPO](https://arxiv.org/abs/2008.07669),
  [S4](https://arxiv.org/abs/2111.00396),
  [Mamba](https://arxiv.org/abs/2312.00752).
- **Koopman methods applied to deep learning.** Koopman and DMD tools have
  been applied to networks before, notably to training dynamics and
  pruning: A. S. Dogra and W. T. Redman, *Optimizing Neural Networks via
  Koopman Operator Theory*, NeurIPS 2020.
- **Mechanistic interpretability.** The dominant route into transformer
  internals studies discrete features and circuits: C. Olah et al.,
  [Zoom In: An Introduction to Circuits](https://distill.pub/2020/circuits/zoom-in/);
  N. Elhage et al.,
  [A Mathematical Framework for Transformer
  Circuits](https://transformer-circuits.pub/2021/framework/index.html).
  That lens is complementary: circuits ask *what* is computed, while this
  package asks *how long* internal activity persists.

What we have not found published elsewhere is this exact combination:
token-time EDMD on the hidden-state trajectories of a language model,
compared across pinned public training checkpoints, and gated by a
preregistered held-out prediction test with shuffled-pairing and persistence
controls. The combination, not the mathematics, is the claim, and it is
small enough to check. If you know earlier work that already does exactly
this, please open an issue and we will cite it.

The analyzed model suite is Pythia: S. Biderman et al.,
[Pythia: A Suite for Analyzing Large Language Models Across Training and
Scaling](https://arxiv.org/abs/2304.01373).

## Licensing

The analysis code is MIT licensed. The included prompt corpus is released under
CC0-1.0. Pythia weights are not redistributed here; they remain under the
upstream Apache-2.0 license and are downloaded from the cited public repository.

# Prime-gating numerical protocol

Status: specified before the first experiment execution on 2026-11-21.
This is a local protocol, not an externally registered study.

## Question and scope

Does a finite prime-counting waveform improve reconstruction of noisy,
compressed graph signals compared with a heat filter and matched non-prime
controls? A second control tests whether changing only a common diffusion clock
can change an exact linear endpoint at equal integrated time.

The experiment uses an analytical path graph, not a corpus-derived ArrowSpace
graph, trained image model, semantic benchmark, or test of the Riemann Hypothesis
(RH). The graph provides a transparent numerical analogue of the frozen spectral
chart. The original image-quality hypothesis remains untested.

## Fixed design

- Seed: 3407; independent NumPy SeedSequence child streams for surrogate phases,
  synthetic data and bootstrap resampling.
- Graph: combinatorial unweighted path with 128 feature nodes; retain the
  64 smallest eigenmodes, including the constant mode.
- Prime grid: 64 equally spaced log coordinates from log(16) to log(1,000,000).
  Evaluate the right-continuous Chebyshev function using all prime powers at
  or below each real grid point, with natural logarithms and float64 arithmetic.
- Prime waveform: relative error `(psi(x) - x) / x`.
- Envelope waveform: `log(x)^2 / sqrt(x)`. This is the RH asymptotic *shape*,
  not an explicit finite-range upper bound with a certified constant.
- Centre and standardise each waveform to unit root-mean-square amplitude.
  Generate 32 real Fourier phase surrogates of the standardised prime waveform,
  preserving DC and Nyquist components and all Fourier magnitudes.
- Choose one shared scale, `beta = 0.25 / max(abs(h))`, over the envelope,
  prime and all surrogate waveforms. Every multiplier `1 + beta*h` is then
  in [0.75, 1.25]. No clipping, outcome-based scale selection or phase selection.
- Map the 64 log-grid positions to ascending retained mode indices. This is an
  imposed design choice; RH does not identify graph and zeta frequencies.

## Modal reconstruction

### Encoding-quality companion

Before the decoding study, evaluate clean-signal compression at ranks
8, 16, 32, 64 and 128 on the same analytical path. Compare the ascending graph
eigenbasis, a random orthonormal basis, and PCA fitted to 1,024 independent
training signals. Evaluate on 2,048 separate signals per regime. The synthetic
population has known zero mean; all three encoders are linear through the origin
and PCA uses the training second-moment matrix without estimating a mean.
The smooth and broadband generators are as specified below. Use independent
streams under seed namespace `[3407, 1]` for training, test and random basis.
Report MSE, retained signal energy, paired MSE differences from the graph basis
with descriptive bootstrap intervals, and the analytical expected truncation
error for the graph basis. Count retained coefficients only; the shared
dictionary is amortised and this is not a file-size or image-rate measurement.

### Decoding-quality comparison

The reference gain is `exp(-nu)` on retained modes and zero elsewhere.
Multiply its retained gains by each positive waveform multiplier, then rescale
to the reference squared gain norm. This matches the expected transmitted energy
of isotropic noise, not clean-signal energy or measured wall time.
Waveform generation uses seed namespace `[3407, 0]`; decoding data and
bootstrap resampling use independent streams under `[3407, 2]`.

Use 2,048 independent signals per regime, with Gaussian spectral coefficients:
standard deviations `1/(1+4*nu)` (smooth) and 1 (broadband).
Add independent Gaussian noise with standard deviation 0.5 in all 128 modes.
Reuse each clean/noisy pair for every arm; no fitting or hyperparameter search.
Report the identity/noisy observation and unfiltered truncated projection too.

Primary outcome: mean squared reconstruction error per feature. Secondary
outcome: graph-weighted error per feature. Save per-signal outcomes.
Report paired changes from the heat baseline with 2,000 paired bootstrap
resamples. These intervals are descriptive, conditional on this fixed graph and
signal generator; they are not corrected for multiple comparisons.
Report all 32 phase controls, their range and the prime waveform's rank.
No surrogate-rank significance claim: the deterministic prime sequence is not
assumed to be a sample from a random-phase null model.

## Common-clock control

Interpret the same multipliers as 64 equal-duration positive time cells on a
unit horizon. They have mean one. Compare sequential exact heat evolution with
the analytic unit-time endpoint, and explicit Euler at 64, 256 and 1,024 steps.
Refinement subdivides each original time cell without resampling the waveform.
Report maximum transfer-function error. The theorem predicts endpoint identity
for the exact solver; Euler differences are numerical discretisation effects.

## Reporting and disposition

Save configuration, source and lockfile hashes, input-array hashes, Python and
NumPy versions, measured outcomes, numerical checks and generated manuscript
tables. Do not capture environment variables. Include every prescribed arm and
regime, including unfavourable outcomes.

An apparent prime advantage on this one constructed benchmark is only a reason
to propose independent graph/data replication. An absent advantage is retained
as a negative result. No claim about real image coherence, retrieval, RH's
truth, or prime-constrained physical vibrations follows from either outcome.

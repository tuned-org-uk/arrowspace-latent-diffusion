# Independent v2 verification

Date: 2026-11-21. Scope: v2 modal gate, numerical controls and registered
neural decoder. Source changes belong to the parent worker.

## Verified fixes

1. **High, fixed: recipe rejection previously mutated the model.**
   The original `set_extra_state` check ran after buffers and earlier
   modules' parameters were copied, leaving a prime gate labelled as heat.
   `src/ald_sc/modal_gate.py:12-20` now preflights nested recipe metadata.
   Load pre-hooks are installed at `ModalGate`, `GraphDecoder`,
   `ClockGatedGraphDecoder` and `SpectralVAE` roots. At all four roots,
   the reproduction verifies every state tensor and metadata value
   remains unchanged after incompatible-recipe rejection.
   Additional tests verify missing metadata is rejected without mutation
   with both `strict=True` and `strict=False`. Persisted same-recipe
   checkpoints reload successfully and give exactly identical finite
   deterministic decoder outputs at every root.
   Original reproduction:
   `test_rejected_recipe_checkpoint_preserves_the_destination_model`.

2. **High, fixed: endpoint 16 was evaluated below its discontinuity.**
   The original log/exp round-trip gave `15.999999999999998`, excluding
   the prime power 16 and its `log(2)` contribution.
   `src/ald_sc/prime_gating.py:67-70` now explicitly restores the declared
   endpoints before evaluating psi. The first relative-error sample
   correctly equals -0.1569996255421674.
   Regeneration of numerical results, figures and tables is assigned to
   the experiment worker and is not verified by this targeted code check.
   Reproduction:
   `test_waveform_includes_the_prime_power_at_its_declared_lower_endpoint`.

3. **Medium, fixed: gain matching underflowed to NaNs.**
   The original code squared unshifted heat gains, underflowing their
   norms to zero. `src/ald_sc/prime_gating.py:91-97` now factors out the
   minimum eigenvalue before norm matching.
   Eigenvalues 400 through 407 produce the expected representable gains;
   eigenvalues 1000 through 1007 produce finite zeros rather than NaNs.
   Reproduction:
   `test_gain_matching_remains_finite_when_squared_heat_norm_underflows`
   at offsets 400 and 1000.

## Scientific interpretation

- The sieve counts prime powers, so it computes Chebyshev psi, not theta.
  Finite summation is unconditional; floating-point evaluation is not
  exact real arithmetic or evidence for the Riemann Hypothesis.
- The protocol correctly states that the envelope is an asymptotic shape,
  not a certified finite-range bound.
- Equal squared filter norms match transmitted isotropic-noise energy
  in the linear experiment. They do not match clean-signal energy,
  wall time or the effective neural reconstruction operator:
   `src/ald_sc/graph_decoder.py:108-122` combines fixed gains with learned
  gates, projections, normalisation and residual convolutions.
- Index-based prime weights depend on the chosen eigenbasis when
  eigenvalues repeat. Ascending eigenvalues do not resolve rotations
  within a degenerate eigenspace. This is harmless for the study's
  simple-spectrum path, but corpus-graph claims need a frozen basis
  identity or an explicit degeneracy policy. It is not generally a
  basis-independent scalar spectral function of the Laplacian.
- Paired clean/noisy samples and paired bootstrap resampling are used;
  PCA is fitted to independent training data. The protocol correctly
  limits intervals to the fixed generator and avoids interpreting
  surrogate ranks as calibrated significance tests.
- Full-rank orthonormal reconstruction is covered, and coefficient ratio
  is explicitly distinguished from image bitrate.

## Observed verification

Command:

```sh
uv run pytest tests/test_prime_gating.py tests/test_v2_decoder.py -q --tb=short
```

Initial reproduction run: **13 passed, 4 failed** in 1.26 seconds.
The four failures correspond to the three findings above.
After parent source fixes and expanded checkpoint checks:
**32 passed** in 2.65 seconds. No full suite was run by this worker.
The parent owns final full-suite verification and the separate zero-mode
common-clock reproduction in `tests/test_spectral_schedule.py`.
No source was modified by this worker. No commits or pushes were made.

Existing passing checks establish prime-power counting, an analytical
path basis, even-length phase-power preservation, positive equal-budget
waveforms, exact common-clock invariance and Euler convergence,
full-rank recovery, paired outcomes, numerical/neural gain agreement,
changed decoder output under identical learned weights, and registered
decoder parameters with finite upstream gradients.

Additional direct numerical checks passed for ranks 4, 5, 7, 8 and 64:
all 32 phase controls preserve Fourier magnitudes, including odd-length
parity handling; multiplier bounds and unit means hold. Ranks below four
are rejected. The modal gate has zero parameters and its output has no
gradient requirement. A pre-fix rotation experiment within eigenvalues
`[0, 1, 1, 2]` preserved the Laplacian but changed the unnormalised prime
operator, confirming the degenerate-basis caveat. Endpoint correction
changes numerical weights but does not remove this structural dependence.

`uv run ruff check tests/test_prime_gating.py tests/test_v2_decoder.py`
passed after formatting the two assigned test files.

# Changelog

## 2.0.1 — 2026-11-21

### Qualified autoresearch result

- Publish the first bounded representation-search outcome: 96 evaluations,
  with training/development separation and a sealed selection before the audit.
- Record the selected graph basis with training-fitted diagonal gains as the
  synthetic research incumbent, together with all candidate and metric artefacts.
- Fresh audit balanced MSE improved by 2.8343% against the v2.0.0 heat reference.
  Smooth-regime MSE rose by 1.4889%, within the declared 2% guard; broadband MSE
  fell by 7.1576%. The encoded subspace did not change.
- Preserve the v2.0.0 experiment-source commit and immutable result hashes.
  Future promotions require a fresh audit against this incumbent.
- Make demonstration notebooks compatible with 2.0.x patch versions.

The default neural model remains the heat reference. This promotion records
a synthetic linear-reconstruction result, not a trained image-model improvement.

## 2.0.0 — 2026-11-21

### Changed

- `SpectralVAE` requires its frozen prior at construction and uses the
  graph-structured decoder, replacing the lazy single-gate decoder.
- All decoder parameters are registered before optimiser construction.
- Existing sample and training-notebook constructors pass the bound prior.
- Version 1 decoder checkpoints require retraining or explicit conversion.

### Added

- Frozen heat, envelope, prime-error and 32 phase-control modal gate recipes.
- Squared-filter-gain budget matching and recipe-aware checkpoint preflight.
- Reproducible synthetic encoding and decoding experiments with paired controls.
- Two executable v2 notebooks, a new manuscript and source-hashed result receipts.
- Calibration-only frozen feature metrics with exact incidence-factor checks.
- A bounded representation-search runner with independent development and audit
  splits, fixed representation budgets and complete keep/discard logging.

### Fixed

- Prime-power evaluation includes the exact declared grid endpoints.
- Gain normalisation remains finite when squared heat norms underflow.
- Rejected modal recipes are checked before checkpoint tensors are copied.
- The common cosine schedule handles graph null modes without division by zero.

### Evidence scope

The numerical quality measurements concern synthetic graph signals. Neural
execution and gradient checks do not establish trained image quality. The
Riemann Hypothesis motivates an envelope shape, not a demonstrated property of
the graph spectrum.

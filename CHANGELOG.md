# Changelog

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

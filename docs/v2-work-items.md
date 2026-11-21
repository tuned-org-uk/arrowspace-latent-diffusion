# v2.0.0 work items

The user requested parallel development of the revised paper, implementation
and two demonstration notebooks on 2026-11-21.

| Item | Deliverable | Ownership | Integration gate |
|---|---|---|---|
| W1 | Modal-gating implementation, eager graph VAE decoder, version and migration | Main session: `src/`, package metadata, sample CLI, README | Behavioural tests, full suite and source review |
| W2 | Self-contained revised manuscript and PDF | Paper worker: `docs/paper/ald-sc-v2.*` and citation notes | Measured tables available; LaTeX builds |
| W3 | Encoding-quality and prime-decoding notebooks; raw numerical evidence | Notebook worker: two `*-v2-*` notebooks, study/runner scripts, `docs/paper/v2-results/` | Clean-kernel execution and source-hashed receipts |
| W4 | Independent mathematical and implementation verification | Verification worker: `tests/test_prime_gating.py`, `tests/test_v2_decoder.py`, verification report | Reproduction tests for findings; no unresolved material findings |

## Shared contract

The study is fixed by `experiments/prime-gating-protocol.md`. It uses seed 3407,
synthetic path-graph signals, equal squared filter-gain norms and all prescribed
phase controls. RH motivates an envelope shape; no finite computation proves RH.
The neural decoder consumes the same modal gains as the numerical study.

W2 consumes W3's measured tables. W3's final provenance receipt must be refreshed
after W1 source changes. W4 reports source findings to W1 rather than editing
files owned by another worker. All Python commands use the repository `.venv`
through `uv`. No publication or remote push is part of these work items.

## Integration status

- W1 complete: v2.0.0 core and migrations; 140 tests pass, global Ruff passes,
  and the prime-gate sample CLI completed a bounded toy training/generation run.
- W3 complete: both notebooks executed, with eight successful code cells;
  raw outcomes, figures, tables and execution receipt are saved.
- W4 complete: all three independent findings fixed and verified; 32 targeted
  numerical/checkpoint tests pass. The main session also fixed the common-clock
  null-mode division by zero with a reproduction test.
- W2 complete: revised manuscript and an inspected 11-page PDF, with the final
  corrected measurements and explicit evidence limits.

The main session independently checked all 23 recorded source, notebook and
artefact hashes after execution. The measured prime effect is regime-dependent:
MSE increased on the smooth generator and decreased on the broadband generator.

## Follow-on autoresearch

The user subsequently requested an adaptive representation search and inclusion
of calibration-fitted frozen feature metrics, described without product naming.
The separate protocol is at repository-root `autoresearch.md`.

The baseline includes the calibrated-metric implementation and bounded search
runner. Successful search outcomes are published separately as patch versions,
with the sealed selection and fresh audit attached to each promotion.

The first promotion is v2.0.1: 96 evaluations, 95 distinct exact basis/gain hash
pairs, and a fresh audit balanced MSE ratio of 0.971657. The original graph basis
with training-fitted diagonal gains won the declared objective. The encoded
subspace did not change. The smooth-regime regression and broadband improvement
are both preserved in `autoresearch/representation-v2-001/`.
The promotion records the published v2.0.0 source commit explicitly.

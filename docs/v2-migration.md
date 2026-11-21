# Version 2.0.0

## Breaking changes

`SpectralVAE` now requires a frozen `prior` at construction. Its decoder is
`GraphDecoder`, with three constructive spectral blocks. The old lazy
single-gate decoder has been replaced, so v1 checkpoint parameter layouts are
incompatible. All trainable decoder parameters exist before optimiser creation.
The existing forward and decode methods still take the same bound prior; passing
a different prior object to decode is rejected.

```python
vae = SpectralVAE(
    prior=prior,
    feature_dim=prior.F,
    base_channels=32,
    gate_kind="heat",
)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)
z, features, chart, reconstruction = vae(images, prior)
```

Do not load a v1 checkpoint with `strict=False` and treat the resulting randomly
initialised decoder as a trained v2 model. Retrain or document a reviewed
conversion and requalification.

## Modal filter contract

The new `ModalGate` owns frozen eigenvalues and multipliers. It accepts ascending,
finite, nonnegative eigenvalues and at least four retained modes. Recipes are
`heat`, `envelope`, `prime` and `phase-00` through `phase-31`. Each uses the same
finite log-prime grid and seed 3407. The graph basis itself is unchanged.

For the unit-time heat reference and a positive multiplier, the fixed gains are

$$
b_k=e^{-\nu_k},\qquad d_k=
\left[b_k(1+\beta h_k)\right]
\frac{\|b\|_2}{\|b\odot(1+\beta h)\|_2}.
$$

The learned gate in each decoder block is multiplied by these gains.
Squared-gain matching is exact for the standalone spectral filter; it is not a
claim of equal total energy, training cost or noise transmission through the
nonlinear neural decoder.

Recipes are stored with checkpoint buffers. A different declared gate recipe is
rejected on restore. The index-based modulation assumes a fixed stored basis;
rotating a repeated-eigenvalue subspace can change a nonuniform gate. Cross-graph
or cross-basis transfer requires a new evaluation.

## Common clock

`SpectralSchedule` retains the existing cosine tempo. Its normalised modal times
reduce to a common external-time ratio; v2 evaluates that ratio directly to avoid
zero-over-zero at a graph null mode. Prime weighting belongs to `ModalGate`,
not to a claim that the common cosine tempo becomes mode-dependent.

## Evidence and reproduction

Run `uv sync --locked --group notebooks` once at the project root. The two v2
notebooks and `scripts/run_prime_study.py` use the same mathematical implementation
as the decoder's fixed gate. Source hashes and raw paired outcomes accompany the
study. Numerical quality measurements concern synthetic spectral signals.
Neural gradient and checkpoint checks establish software behaviour only.

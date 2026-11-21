# Representation autoresearch

## Objective

Find the best **tested linear representation** for noisy reconstruction under
the v2 synthetic graph-signal benchmark. This is a bounded local CPU experiment,
not a real-image or general semantic-quality claim.

The primary score is the equally weighted mean of each regime's development MSE
divided by its development heat-reference MSE. Lower is better; the reference is
one. A candidate is eligible only if neither regime exceeds 102% of its reference
MSE. This limits improvements obtained by sacrificing the other regime.

## Fixed budgets

- Seed 3407 with new namespaces, independent of the manuscript experiment.
- 128 features, exactly 64 encoded coefficients.
- Orthonormal encoder columns and a diagonal nonnegative decoder gain.
- Same squared gain norm as the v2 unit-time heat filter for every candidate.
- 2,048 training and 2,048 development signals per regime.
- One final audit on 4,096 newly generated signals per regime, after selection.
- At most 96 trials; no training or development decision uses audit outcomes.
- Smooth and broadband generators and noise standard deviation 0.5 match the
  v2 protocol. The new samples are independent; the generator itself is known.

## Search space

Compare graph, pooled PCA, regime-balanced PCA, random orthonormal and coordinate
bases, plus low/high spectral charts from calibration-fitted feature metrics.
PCA and feature-graph fitting see training clean signals only. Evaluate
heat-shaped gains at times 0, 0.5, 1 and 2, and training-fitted diagonal regression gains.
Use empirical training heat errors as regime weights when fitting the gains.
The fitted gains solve the diagonal quadratic risk under the shared gain-norm
constraint by a one-dimensional Lagrange-multiplier search.

Compare prime and envelope modulation at half, unit and double the v2 strength,
plus all 32 fixed phase controls, on the graph heat reference. Then refine the
best eligible development basis with heat times 0.25, 0.75, 1.25, 1.5 and 4,
and eight convex blends between its reference heat filter and training-fitted
filter, renormalising every candidate to the fixed gain budget.

## Calibrated metric methodology

Fit positive-cosine feature neighbours from a training-only Gram matrix, using
4 or 8 neighbours and a canonical undirected union. Freeze the symmetric
normalised feature Laplacian and add an unconditional positive diagonal lift
of 0.01. Isolated features have zero graph term and retain the lift.

An incidence factor provides the exact full metric:

$$
S=\begin{bmatrix}BD^{-1/2}\\\sqrt{\epsilon}I\end{bmatrix},
\qquad M=S^\top S=L_{\mathrm{sym}}+\epsilon I.
$$

Its normalised transformed vectors implement the frozen similarity, with an
explicit near-zero branch. Verify factor/metric score parity and preserve
calibration, coordinate and recipe identity across all splits.

The full factor expands to edge-count plus feature-count dimensions; it does
**not** enter the 64-coefficient leaderboard as if it were compressed.
Instead, separately identified low/high 64-mode charts of the fitted graph are
candidate encoders. Truncating these charts is an additional approximation,
not an exact export of the full metric. The eigensystem is frozen with each
candidate. No graph is fitted over items or evaluation samples.

Keep a candidate only when it is eligible and improves the primary score.
Record every trial, including rejected or inferior candidates. Stop at the
finite candidate budget; do not keep searching after inspecting the final audit.

## Outputs

`autoresearch.jsonl` records the sequential keep/discard decisions.
`autoresearch/representation-v2-001/` stores source and data hashes, the selected
encoder/gains, the selection sealed before audit, all raw paired audit outcomes,
and final evidence. `autoresearch.sh` runs this experiment using root `.venv`.

The original paper experiment remains an independently specified comparison.
Any follow-on manuscript result must identify this adaptive search as a separate
experiment with its own development exposure.

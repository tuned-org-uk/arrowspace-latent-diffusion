# Representation search results

Trials: 96. Selected basis: `graph`.
Selected recipe: `{'family': 'training-fitted-quadratic'}`.

Development balanced MSE ratio: 0.972076.
Fresh audit balanced MSE ratio: 0.971657.
Audit regime guard passed: True.

| Regime | Heat MSE | Selected MSE | Paired change | 95% interval |
|---|---:|---:|---:|---:|
| smooth | 0.060793 | 0.061698 | +0.000905 | [+0.000789, +0.001031] |
| broadband | 0.682176 | 0.633349 | -0.048827 | [-0.049325, -0.048326] |

The audit was generated only after selection was written and sealed.
It did not change the selected recipe. Intervals are descriptive paired
bootstrap intervals conditional on the fixed synthetic generator.

Projector distance from reference: 0.
A zero distance means the improvement changes decoder gains, not the encoded subspace.

This is the best eligible development configuration among the tested
candidates under a fixed rank and filter-noise budget. It does not establish
a globally optimal representation, real-image quality, or prime-specific benefit.

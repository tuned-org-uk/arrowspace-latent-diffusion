# V2 source verification

Checked on 21 November 2026. The new manuscript is `ald-sc-v2.tex`; the original remains a historical source.

## Bibliography

| Source | Verification | Use |
| --- | --- | --- |
| Rombach et al., arXiv:2112.10752 | arXiv record, authors, title, CVPR 2022 comment | Latent diffusion architecture |
| Kingma and Welling, arXiv:1312.6114 | arXiv record, authors, title | Variational reparameterisation |
| Peebles and Xie, arXiv:2212.09748 | arXiv record, authors, title | Diffusion transformer background |
| Shuman et al., arXiv:1211.0053 | arXiv record and linked DOI 10.1109/MSP.2012.2235192 | Graph spectral filtering background |
| tuned-org-uk/pyarrowspace | Public repository and README inspected | ArrowSpace software source; no corpus experiment inferred |
| Schoenfeld (1976), DOI 10.1090/S0025-5718-1976-0457374-X | DOI metadata confirms title, journal, volume, pages | Classical RH/Chebyshev error reference |

The Schoenfeld journal full-text HTML timed out, and a direct PDF download returned HTTP 403. Its DOI metadata was accessible. NIST DLMF §27.12 independently identifies Schoenfeld (1976) and von Koch (1901) as classical RH-bound sources; that section's displayed explicit bound concerns prime counting, not the exact Chebyshev formula used here. The manuscript uses the standard asymptotic Chebyshev statement and does not claim to have checked a finite-range constant in the primary PDF.

## Local evidence

- Read `AGENTS.md`, `docs/00.md`, the historical manuscript, and the fixed numerical protocol.
- Inspected `prime_gating.py`, `modal_gate.py`, `graph_decoder.py`, `vae.py`, `spectral_schedule.py`, and the numerical exporter.
- Decoder equations follow global pooling and a spatially broadcast correction, not per-pixel graph propagation.
- Reparameterisation and deterministic graph reconstruction have separate roles.
- The fixed modal gain multiplies the learned chart gate; the dispersion descriptor remains part of that learned gate.
- Common cosine attenuation follows cancellation of the eigenvalue in normalised clock time. It is not mode-dependent resolution.
- RH motivation, unconditional finite arithmetic, amplitude normalisation and exact common-clock invariance are separate claims.
- The only quantitative performance evidence admitted is the measured v2 JSON and its generated tables; historical unit-test counts and toy-image claims are excluded.

## Final measured artefacts

Read the corrected final `v2-results/results.json` and all three generated tables after the worker's completion notification. The input stream includes the corrected inclusive prime-grid endpoints. The report records Python 3.13.12, NumPy 2.5.1 and `prime_gating.py` SHA-256 `26e3842e80e0d9a52df4e9fd522357e4408033313a6692ab96a6b89e34b763d0`.

- Prime smooth MSE: 0.061753721846582524 versus heat 0.061150755598928214; paired change +0.0006029662476544036.
- Prime broadband MSE: 0.6805813529776746 versus heat 0.6849175771143323; paired change −0.004336224136660462.
- Prime ranks: 29/33 smooth, 2/33 broadband. Phase-13 outperforms prime on broadband; no rank-based significance claim.
- Maximum exact common-clock error: 5.46229728115577e−14.
- Results prose includes both regimes, phase ranges, encoding PCA/random controls and graph-weighted secondary errors. The analytical bias–noise decomposition is a derivation, not a new numerical experiment.

## Manuscript build

Built `ald-sc-v2.pdf` with `pdflatex` from `docs/paper` and resolved cross-references. The manuscript is 11 pages, with the three generated numerical tables, two generated figures, and a code-to-claim table. Inspected the parsed PDF text and rendered results pages; no missing references, LaTeX warnings or overfull/underfull boxes remained in the checked build.

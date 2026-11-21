# Current design

Version 2.0.0 implements the frozen-graph ALD-SC architecture described in
`paper/ald-sc-v2.tex`. The earlier research notes in `00.md` and `01.md` remain
background; the executable modal-gating contract is in `v2-migration.md`.

The image encoder returns a spatial VAE latent and a feature vector.
The frozen prior derives the chart descriptor from that feature vector.
Latent diffusion still operates only on the spatial latent.

`SpectralVAE` binds the prior at construction and eagerly builds `GraphDecoder`.
Each of its three spectral blocks projects pooled activations to the stored
chart, multiplies a learned chart gate by the fixed `ModalGate` gains, lifts
back through the same basis, and applies a residual convolutional update.
The graph eigenvectors and finite prime waveform are buffers, not parameters.

The independent `SpectralSchedule` is a common cosine tempo. Its eigenvalue
cancels in normalised external time; `ClockGatedGraphDecoder` can apply this
tempo to residual updates in addition to the modal weights.

The numerical study in `experiments/prime-gating-protocol.md` isolates linear
encoding and filtering on a synthetic analytical graph. It establishes neither
the quality of a trained VAE nor the benefit of a corpus-derived ArrowSpace
graph. Those remain explicitly separate research questions.

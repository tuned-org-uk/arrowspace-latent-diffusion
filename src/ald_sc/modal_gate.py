"""Frozen v2 spectral gains shared by neural decoding and numerical studies."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ald_sc.arrow_prior import ArrowSpacePrior
from ald_sc.prime_gating import waveforms


def _check_modal_recipes(
    module,
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
) -> None:
    """Preflight every nested modal recipe before the root copies any tensors."""
    for name, child in module.named_modules():
        if isinstance(child, ModalGate):
            key = prefix + (f"{name}." if name else "") + "_extra_state"
            if state_dict.get(key) != child.get_extra_state():
                raise ValueError(
                    "checkpoint modal gate recipe does not match this model"
                )


class ModalGate(nn.Module):
    """Positive mode weights with heat-matched squared gain norm.

    Args:
        prior: Frozen, ascending spectral chart, with at least four modes.
        kind: ``heat``, ``envelope``, ``prime`` or ``phase-00`` to ``phase-31``.

    The recipe uses the fixed seed 3407 and a finite log-prime grid. Eigenvectors
    remain those of the supplied graph; no zeta/graph frequency identity is used.
    """

    def __init__(self, prior: ArrowSpacePrior, kind: str = "heat") -> None:
        super().__init__()
        nu = prior.eigvals_q.detach().clone()
        if not torch.isfinite(nu).all() or (nu < 0).any() or (nu.diff() < 0).any():
            raise ValueError("eigenvalues must be finite, nonnegative and ascending")
        _, _, names, multipliers, _ = waveforms(prior.q)
        if kind not in names:
            raise ValueError(f"unknown modal gate kind: {kind}")
        self.kind = kind
        self.register_load_state_dict_pre_hook(_check_modal_recipes)
        self.register_buffer("nu", nu)
        self.register_buffer(
            "multiplier",
            torch.as_tensor(multipliers[names.index(kind)], dtype=nu.dtype),
        )

    def forward(self) -> Tensor:
        """Return fixed unit-time gains; the operation is differentiable upstream."""
        # Subtracting the minimum avoids norm underflow for large eigenvalues.
        relative = torch.exp(-(self.nu - self.nu.min()))
        weighted = relative * self.multiplier
        scale = torch.linalg.vector_norm(relative) / torch.linalg.vector_norm(weighted)
        return weighted * scale * torch.exp(-self.nu.min())

    def get_extra_state(self) -> dict:
        """Persist recipe identity alongside the frozen numeric buffers."""
        return {"version": "2.0.0", "kind": self.kind, "seed": 3407, "surrogates": 32}

    def set_extra_state(self, state: dict) -> None:
        """Reject loading a checkpoint under a differently labelled recipe."""
        if state != self.get_extra_state():
            raise ValueError("checkpoint modal gate recipe does not match this model")

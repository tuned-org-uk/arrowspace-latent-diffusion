"""Behavioural checks for the v2 modal-gating training path."""

import copy

import numpy as np
import pytest
import torch

from ald_sc.arrow_prior import ArrowSpacePrior
from ald_sc.graph_decoder import ClockGatedGraphDecoder, GraphDecoder
from ald_sc.modal_gate import ModalGate
from ald_sc.prime_gating import matched_gains, path_basis, waveforms
from ald_sc.spectral_schedule import SpectralSchedule
from ald_sc.vae import SpectralVAE


def prior_fixture():
    nu, basis = path_basis(16)
    return ArrowSpacePrior(
        torch.tensor((basis * nu) @ basis.T),
        torch.tensor(basis[:, :8]),
        torch.tensor(nu[:8]),
        torch.linspace(0, 1, 16),
    )


def test_v2_gate_matches_the_numerical_experiment_and_has_no_parameters():
    prior = prior_fixture()
    decoder = GraphDecoder(4, 3, 16, 8, prior, gate_kind="prime")
    _, _, names, multipliers, _ = waveforms(8)
    expected = matched_gains(prior.eigvals_q.numpy().astype(float), multipliers)[
        names.index("prime")
    ]
    np.testing.assert_allclose(
        decoder.spectral_gate().detach().numpy(), expected, rtol=2e-6
    )
    assert list(decoder.spectral_gate.parameters()) == []


def test_prime_gate_changes_decoder_output_with_identical_learned_weights():
    prior = prior_fixture()
    torch.manual_seed(3407)
    heat = GraphDecoder(4, 3, 16, 8, prior, gate_kind="heat")
    torch.manual_seed(3407)
    prime = GraphDecoder(4, 3, 16, 8, prior, gate_kind="prime")
    for a, b in zip(heat.parameters(), prime.parameters(), strict=True):
        torch.testing.assert_close(a, b)
    z = torch.randn(2, 4, 4, 4, requires_grad=True)
    chart = torch.randn(2, 24)
    assert not torch.allclose(heat(z, chart), prime(z, chart), atol=1e-7)
    prime(z, chart).square().mean().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_vae_registers_modal_decoder_before_optimizer_creation():
    prior = prior_fixture()
    vae = SpectralVAE(prior=prior, feature_dim=16, base_channels=8, gate_kind="prime")
    assert isinstance(vae.decoder, GraphDecoder)
    parameter_ids = {id(p) for p in vae.parameters()}
    optimizer = torch.optim.SGD(vae.parameters(), lr=0.01)
    assert parameter_ids == {
        id(p) for group in optimizer.param_groups for p in group["params"]
    }
    x = torch.randn(2, 3, 16, 16)
    output = vae(x, prior)[-1]
    assert output.shape == x.shape
    output.square().mean().backward()
    assert parameter_ids == {id(p) for p in vae.parameters()}
    for block in (
        vae.decoder.wave_block_1,
        vae.decoder.wave_block_2,
        vae.decoder.wave_block_3,
    ):
        assert block.gate.weight.grad is not None
        assert block.gate.weight.grad.abs().sum() > 0


def test_checkpoint_rejects_a_different_gate_recipe():
    prior = prior_fixture()
    prime = GraphDecoder(4, 3, 16, 8, prior, gate_kind="prime")
    heat = GraphDecoder(4, 3, 16, 8, prior, gate_kind="heat")
    with pytest.raises(ValueError, match="recipe"):
        heat.load_state_dict(prime.state_dict())


def checkpoint_model(root, kind):
    prior = prior_fixture()
    if root is ModalGate:
        return ModalGate(prior, kind)
    if root is SpectralVAE:
        return SpectralVAE(prior=prior, feature_dim=16, base_channels=8, gate_kind=kind)
    if root is ClockGatedGraphDecoder:
        return root(4, 3, 16, 8, prior, SpectralSchedule(prior), gate_kind=kind)
    return root(4, 3, 16, 8, prior, gate_kind=kind)


@pytest.mark.parametrize(
    "root", [ModalGate, GraphDecoder, ClockGatedGraphDecoder, SpectralVAE]
)
def test_rejected_recipe_checkpoint_preserves_the_destination_model(root):
    """A failed identity check must not leave prime buffers labelled as heat."""
    torch.manual_seed(3407)
    prime = checkpoint_model(root, "prime")
    torch.manual_seed(3408)
    heat = checkpoint_model(root, "heat")
    before = copy.deepcopy(heat.state_dict())
    with pytest.raises(ValueError, match="recipe"):
        heat.load_state_dict(prime.state_dict())
    after = heat.state_dict()
    for name, value in before.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(after[name], value, rtol=0, atol=0, msg=name)
        else:
            assert after[name] == value


@pytest.mark.parametrize(
    "root", [ModalGate, GraphDecoder, ClockGatedGraphDecoder, SpectralVAE]
)
@pytest.mark.parametrize("strict", [True, False])
def test_missing_recipe_metadata_is_rejected_without_mutation(root, strict):
    """Missing identity must fail even when ordinary missing keys are allowed."""
    torch.manual_seed(3407)
    source = checkpoint_model(root, "prime")
    torch.manual_seed(3408)
    target = checkpoint_model(root, "prime")
    before = copy.deepcopy(target.state_dict())
    checkpoint = source.state_dict()
    metadata_keys = [key for key in checkpoint if key.endswith("_extra_state")]
    assert len(metadata_keys) == 1
    del checkpoint[metadata_keys[0]]
    # Ensure a direct ModalGate load would also visibly mutate a numeric buffer.
    for name, value in checkpoint.items():
        if name.endswith("multiplier"):
            checkpoint[name] = value + 0.1
    with pytest.raises(ValueError, match="recipe"):
        target.load_state_dict(checkpoint, strict=strict)
    for name, value in before.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(
                target.state_dict()[name], value, rtol=0, atol=0, msg=name
            )
        else:
            assert target.state_dict()[name] == value


@pytest.mark.parametrize(
    "root", [ModalGate, GraphDecoder, ClockGatedGraphDecoder, SpectralVAE]
)
def test_same_recipe_checkpoint_roundtrip_preserves_output(root, tmp_path):
    """Recipe preflight must permit valid persisted checkpoints at every root."""
    torch.manual_seed(3407)
    source = checkpoint_model(root, "prime").eval()
    torch.manual_seed(3408)
    target = checkpoint_model(root, "prime").eval()
    path = tmp_path / "prime.pt"
    torch.save(source.state_dict(), path)
    result = target.load_state_dict(torch.load(path, weights_only=True))
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    for name, value in source.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(
                target.state_dict()[name], value, rtol=0, atol=0, msg=name
            )
        else:
            assert target.state_dict()[name] == value
    z, chart = torch.randn(2, 4, 4, 4), torch.randn(2, 24)
    with torch.no_grad():
        if root is ModalGate:
            expected, actual = source(), target()
        elif root is SpectralVAE:
            expected = source.decode(z, chart, source.prior)
            actual = target.decode(z, chart, target.prior)
        else:
            expected, actual = source(z, chart), target(z, chart)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

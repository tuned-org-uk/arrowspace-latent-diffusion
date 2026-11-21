"""Tests for samplers."""

from __future__ import annotations

import torch

from ald_sc.dit import MinimalDiT
from ald_sc.schedule import CosineSchedule
from ald_sc.sampling import sample_euler, sample_ddim, sample_ddim_steps


def _make_dit(
    latent_channels: int = 4, latent_size: int = 8, spec_dim: int = 12
) -> MinimalDiT:
    return MinimalDiT(
        latent_channels=latent_channels,
        latent_size=latent_size,
        patch_size=2,
        dim=32,
        depth=2,
        num_heads=4,
        text_dim=0,
        spec_dim=spec_dim,
        cfg_dropout=0.0,
    )


class TestSampleEuler:
    def test_output_shape(self) -> None:
        torch.manual_seed(3407)
        dit = _make_dit()
        sched = CosineSchedule(num_steps=100)
        c_spec = torch.randn(2, 12)
        steps = 10

        z = sample_euler(dit, sched, c_spec=c_spec, batch_size=2, steps=steps)
        assert z.shape == (2, 4, 8, 8)

    def test_determinism_with_seed(self) -> None:
        torch.manual_seed(3407)
        dit = _make_dit()
        sched = CosineSchedule(num_steps=100)
        c_spec = torch.randn(2, 12)

        z1 = sample_euler(dit, sched, c_spec=c_spec, batch_size=2, steps=10, seed=3407)
        z2 = sample_euler(dit, sched, c_spec=c_spec, batch_size=2, steps=10, seed=3407)
        assert torch.allclose(z1, z2, atol=1e-6)


class TestSampleDDIM:
    def test_oracle_velocity_follows_one_fixed_noise_trajectory(self) -> None:
        schedule = CosineSchedule(num_steps=12)

        class ZeroCleanOracle(torch.nn.Module):
            latent_channels = latent_size = 1

            def forward(self, x, t, c_spec=None):
                alpha = schedule[t].view(-1, 1, 1, 1)
                return (alpha / (1 - alpha)).sqrt() * x

        times = schedule.sample_sigmas(4)
        model = ZeroCleanOracle()
        frames = list(sample_ddim_steps(model, schedule, steps=4))
        # For a clean latent of zero, every frame is the same epsilon times
        # sqrt(1-alpha_t), including the nonzero endpoint of this coarse grid.
        noise = frames[0] / (1 - schedule[times[0]]).sqrt()
        for latent, time in zip(frames, times, strict=True):
            torch.testing.assert_close(latent, noise * (1 - schedule[time]).sqrt())
        torch.testing.assert_close(sample_ddim(model, schedule, steps=4), frames[-1])

    def test_output_shape(self) -> None:
        torch.manual_seed(3407)
        dit = _make_dit()
        sched = CosineSchedule(num_steps=100)
        c_spec = torch.randn(2, 12)

        z = sample_ddim(dit, sched, c_spec=c_spec, batch_size=2, steps=10)
        assert z.shape == (2, 4, 8, 8)

    def test_determinism_with_seed(self) -> None:
        torch.manual_seed(3407)
        dit = _make_dit()
        sched = CosineSchedule(num_steps=100)
        c_spec = torch.randn(2, 12)

        z1 = sample_ddim(dit, sched, c_spec=c_spec, batch_size=2, steps=10, seed=3407)
        z2 = sample_ddim(dit, sched, c_spec=c_spec, batch_size=2, steps=10, seed=3407)
        assert torch.allclose(z1, z2, atol=1e-6)


class TestEulerDDIMParity:
    def test_single_step_parity(self) -> None:
        """At a single step, Euler and DDIM should produce similar results."""
        torch.manual_seed(3407)
        dit = _make_dit()
        sched = CosineSchedule(num_steps=100)
        c_spec = torch.randn(1, 12)

        z_euler = sample_euler(
            dit, sched, c_spec=c_spec, batch_size=1, steps=1, seed=3407
        )
        z_ddim = sample_ddim(
            dit, sched, c_spec=c_spec, batch_size=1, steps=1, seed=3407
        )
        assert torch.allclose(z_euler, z_ddim, atol=1e-4)

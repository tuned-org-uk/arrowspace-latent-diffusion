"""Analytical checks for the prime-gating numerical study."""

import math

import numpy as np
import pytest

from ald_sc.prime_gating import (
    chebyshev_psi,
    clock_errors,
    decoding_experiment,
    encoding_experiment,
    matched_gains,
    path_basis,
    phase_surrogate,
    waveforms,
)


def test_chebyshev_counts_prime_powers_once_and_is_right_continuous():
    x = np.array([1, 2, 3, 4, 7.999, 8, 9, 10], dtype=float)
    expected = np.log([1, 2, 6, 12, 420, 840, 2520, 2520])
    np.testing.assert_allclose(chebyshev_psi(x), expected, atol=1e-14)


@pytest.mark.parametrize("x", [np.array([0.5]), np.array([np.nan])])
def test_chebyshev_rejects_invalid_domain(x):
    with pytest.raises(ValueError):
        chebyshev_psi(x)


def test_path_basis_matches_three_node_laplacian():
    nu, basis = path_basis(3)
    laplacian = np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]])
    np.testing.assert_allclose(nu, [0, 1, 3], atol=1e-14)
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-14)
    np.testing.assert_allclose(laplacian @ basis, basis * nu, atol=1e-14)


def test_phase_surrogate_preserves_power_mean_and_variance():
    signal = np.array([1, -2, 3, 0, -1, 4, -3, 2], dtype=float)
    surrogate = phase_surrogate(signal, np.random.default_rng(3407))
    np.testing.assert_allclose(
        np.abs(np.fft.rfft(surrogate)), np.abs(np.fft.rfft(signal)), atol=1e-13
    )
    assert surrogate.mean() == pytest.approx(signal.mean())
    assert surrogate.var() == pytest.approx(signal.var())
    assert not np.allclose(signal, surrogate)


def test_positive_waveforms_share_budget_without_clipping():
    _, _, names, multipliers, _ = waveforms(16, 4)
    assert names[:3] == ["heat", "envelope", "prime"]
    assert multipliers.min() >= 0.75 - 1e-14
    assert multipliers.max() <= 1.25 + 1e-14
    np.testing.assert_allclose(multipliers.mean(axis=1), 1.0, atol=1e-14)
    np.testing.assert_allclose(
        np.std(multipliers[1:], axis=1), np.std(multipliers[1]), atol=1e-14
    )
    nu, _ = path_basis(32)
    gains = matched_gains(nu[:16], multipliers)
    np.testing.assert_allclose(
        np.sum(gains**2, axis=1), np.sum(np.exp(-nu[:16]) ** 2), atol=1e-13
    )


def test_exact_clock_is_invariant_and_euler_converges():
    nu = np.array([0.0, 0.5, 2.0])
    gates = np.array([[1, 1, 1, 1], [0.75, 1.25, 1.25, 0.75]])
    rows = clock_errors(nu, ["flat", "variable"], gates, (1, 4, 16))
    for name in ("flat", "variable"):
        result = [row for row in rows if row["arm"] == name]
        assert max(row["exact_error"] for row in result) < 1e-14
        errors = [row["euler_error"] for row in result]
        assert errors[0] > errors[1] > errors[2]


def test_encoding_full_rank_recovers_signals_and_smaller_rank_loses_energy():
    result = encoding_experiment(n_features=16, n_train=64, n_test=32, ranks=(4, 16))
    for row in result["rows"]:
        if row["rank"] == 16:
            assert row["mse"] < 1e-25
            assert row["retained_energy"] == pytest.approx(1.0)
        else:
            assert row["mse"] > 0
            assert 0 < row["retained_energy"] < 1
    assert result["config"]["seed"] == 3407


def test_decoding_uses_all_arms_and_paired_raw_outcomes():
    result = decoding_experiment(
        n_features=16, rank=8, n_test=32, n_surrogates=4, n_boot=20
    )
    assert len(result["rows"]) == 2 * (3 + 4 + 2)
    for regime, values in result["per_signal"].items():
        assert values.shape == (32, 9)
        heat = next(
            row
            for row in result["rows"]
            if row["regime"] == regime and row["arm"] == "heat"
        )
        assert heat["delta_mse"] == 0
        assert heat["ci_low"] == 0
        assert heat["ci_high"] == 0
        assert math.isfinite(heat["mse"])


def test_waveform_includes_the_prime_power_at_its_declared_lower_endpoint():
    """The log-grid starts at 16, whose jump includes log(2)."""
    _, error, _, _, _ = waveforms(8, 4)
    expected = (chebyshev_psi(np.array([16.0]))[0] - 16.0) / 16.0
    assert error[0] == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize("offset", [400.0, 1000.0])
def test_gain_matching_remains_finite_when_squared_heat_norm_underflows(offset):
    """Finite eigenvalues must not turn a valid heat filter into NaNs."""
    nu = offset + np.arange(8, dtype=float)
    gains = matched_gains(nu, np.ones((2, 8)))
    assert np.isfinite(gains).all()
    np.testing.assert_allclose(
        gains, np.broadcast_to(np.exp(-nu), (2, 8)), rtol=1e-14, atol=0
    )

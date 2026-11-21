"""Finite prime-waveform filters and synthetic spectral experiments.

RH motivates an envelope shape only. The finite prime sums and all numerical
experiments are unconditional computations; none establish image quality.
"""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
SEED = 3407


def chebyshev_psi(x: Array) -> Array:
    """Evaluate the right-continuous prime-power sum at finite real inputs."""
    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)) or np.any(x < 1) or np.any(x > 1_000_000):
        raise ValueError("x must be finite and in [1, 1000000]")
    limit = int(np.floor(x.max()))
    primes = np.ones(limit + 1, dtype=bool)
    primes[:2] = False
    for p in range(2, int(np.sqrt(limit)) + 1):
        if primes[p]:
            primes[p * p :: p] = False
    weights = np.zeros(limit + 1)
    for p in np.flatnonzero(primes):
        power = int(p)
        while power <= limit:
            weights[power] = np.log(p)
            power *= int(p)
    return np.cumsum(weights)[np.floor(x).astype(int)]


def path_basis(n: int) -> tuple[Array, Array]:
    """Return the exact DCT eigenbasis of an unweighted n-node path."""
    if n < 2:
        raise ValueError("n must be at least two")
    k = np.arange(n)
    nu = 2 - 2 * np.cos(np.pi * k / n)
    basis = np.sqrt(2 / n) * np.cos(np.pi * (np.arange(n)[:, None] + 0.5) * k / n)
    basis[:, 0] = 1 / np.sqrt(n)
    return nu, basis


def phase_surrogate(signal: Array, rng: np.random.Generator) -> Array:
    """Randomise interior real-FFT phases, preserving DC and Nyquist."""
    spectrum = np.fft.rfft(signal)
    stop = len(spectrum) - 1 if len(signal) % 2 == 0 else len(spectrum)
    spectrum[1:stop] *= np.exp(1j * rng.uniform(-np.pi, np.pi, stop - 1))
    return np.fft.irfft(spectrum, n=len(signal))


def waveforms(
    rank: int = 64, n_surrogates: int = 32
) -> tuple[Array, Array, list[str], Array, float]:
    """Build bounded, equal-mean, equal-RMS finite waveform multipliers.

    Returns log-grid, relative prime error, arm names, multipliers and beta.
    All nonconstant arms share one scale selected without observing outcomes.
    """
    if rank < 4 or n_surrogates < 1:
        raise ValueError("rank >= 4 and n_surrogates >= 1 are required")
    s = np.linspace(np.log(16), np.log(1_000_000), rank)
    x = np.exp(s).clip(max=1_000_000)
    x[0], x[-1] = 16.0, 1_000_000.0
    error = (chebyshev_psi(x) - x) / x
    envelope = s**2 * np.exp(-s / 2)

    def standardise(values: Array) -> Array:
        centred = values - values.mean()
        return centred / np.sqrt(np.mean(centred**2))

    prime = standardise(error)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 0]))
    signals = np.stack(
        [standardise(envelope), prime]
        + [phase_surrogate(prime, rng) for _ in range(n_surrogates)]
    )
    beta = 0.25 / np.max(np.abs(signals))
    multipliers = np.vstack([np.ones(rank), 1 + beta * signals])
    names = ["heat", "envelope", "prime"] + [
        f"phase-{i:02d}" for i in range(n_surrogates)
    ]
    return s, error, names, multipliers, float(beta)


def matched_gains(nu: Array, multipliers: Array) -> Array:
    """Apply waveform gains at unit heat time, matching squared gain norms."""
    base = np.exp(-(nu - np.min(nu)))
    gains = base * multipliers
    return (
        gains
        * np.sqrt(np.sum(base**2) / np.sum(gains**2, axis=-1, keepdims=True))
        * np.exp(-np.min(nu))
    )


def clock_errors(
    nu: Array,
    names: list[str],
    multipliers: Array,
    refinements: tuple[int, ...] = (1, 4, 16),
) -> list[dict]:
    """Compare exact and Euler endpoints at equal integrated diffusion time."""
    rows = []
    target = np.exp(-nu)
    for name, gate in zip(names, multipliers, strict=True):
        for refinement in refinements:
            rates = np.repeat(gate, refinement)
            dt = 1 / len(rates)
            exact = np.prod(np.exp(-dt * rates[:, None] * nu), axis=0)
            euler = np.prod(1 - dt * rates[:, None] * nu, axis=0)
            rows.append(
                {
                    "arm": name,
                    "steps": len(rates),
                    "exact_error": float(np.max(np.abs(exact - target))),
                    "euler_error": float(np.max(np.abs(euler - target))),
                }
            )
    return rows


def _digest(array: Array) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array, dtype="<f8").tobytes()
    ).hexdigest()


def _intervals(deltas: Array, rng: np.random.Generator, n_boot: int) -> Array:
    boot = np.empty((n_boot, deltas.shape[1]))
    for i in range(n_boot):
        boot[i] = deltas[rng.integers(0, len(deltas), len(deltas))].mean(axis=0)
    return np.quantile(boot, [0.025, 0.975], axis=0)


def encoding_experiment(
    n_features: int = 128,
    n_train: int = 1024,
    n_test: int = 2048,
    ranks: tuple[int, ...] = (8, 16, 32, 64, 128),
    n_boot: int = 2000,
) -> dict:
    """Measure held-out linear encoding quality against PCA and random bases."""
    if any(q < 1 or q > n_features for q in ranks):
        raise ValueError("ranks must be between one and n_features")
    nu, graph = path_basis(n_features)
    streams = np.random.SeedSequence([SEED, 1]).spawn(7)
    random_basis, _ = np.linalg.qr(
        np.random.default_rng(streams[0]).normal(size=graph.shape)
    )
    rows, raw, hashes = [], {}, {}
    for i, regime in enumerate(("smooth", "broadband")):
        sd = 1 / (1 + 4 * nu) if regime == "smooth" else np.ones(n_features)
        train = (
            np.random.default_rng(streams[1 + 3 * i]).normal(size=(n_train, n_features))
            * sd
        ) @ graph.T
        test = (
            np.random.default_rng(streams[2 + 3 * i]).normal(size=(n_test, n_features))
            * sd
        ) @ graph.T
        _, pca = np.linalg.eigh(train.T @ train / n_train)
        bases = {"graph": graph, "pca": pca[:, ::-1], "random": random_basis}
        hashes[regime] = {"train": _digest(train), "test": _digest(test)}
        bootstrap_rng = np.random.default_rng(streams[3 + 3 * i])
        for rank in ranks:
            errors, energies = [], []
            for basis in bases.values():
                code = test @ basis[:, :rank]
                reconstructed = code @ basis[:, :rank].T
                errors.append(np.mean((reconstructed - test) ** 2, axis=1))
                energies.append(np.sum(code**2, axis=1) / np.sum(test**2, axis=1))
            errors = np.stack(errors, axis=1)
            raw[f"{regime}_q{rank}"] = errors
            deltas = errors - errors[:, :1]
            ci = _intervals(deltas, bootstrap_rng, n_boot)
            for j, name in enumerate(bases):
                rows.append(
                    {
                        "regime": regime,
                        "basis": name,
                        "rank": rank,
                        "coefficient_ratio": rank / n_features,
                        "mse": float(errors[:, j].mean()),
                        "retained_energy": float(np.mean(energies[j])),
                        "delta_mse": float(deltas[:, j].mean()),
                        "ci_low": float(ci[0, j]),
                        "ci_high": float(ci[1, j]),
                        "expected_graph_mse": float(
                            np.sum(sd[rank:] ** 2) / n_features
                        ),
                    }
                )
    return {
        "config": {
            "seed": SEED,
            "features": n_features,
            "n_train": n_train,
            "n_test": n_test,
            "ranks": list(ranks),
            "n_boot": n_boot,
            "version": "2.0.0",
        },
        "rows": rows,
        "per_signal": raw,
        "input_hashes": hashes,
    }


def decoding_experiment(
    n_features: int = 128,
    rank: int = 64,
    n_test: int = 2048,
    n_surrogates: int = 32,
    n_boot: int = 2000,
) -> dict:
    """Evaluate all fixed filters on paired noisy synthetic graph signals."""
    if rank > n_features:
        raise ValueError("rank cannot exceed n_features")
    nu, _ = path_basis(n_features)
    s, error, names, multipliers, beta = waveforms(rank, n_surrogates)
    gains = matched_gains(nu[:rank], multipliers)
    filters = np.pad(gains, ((0, 0), (0, n_features - rank)))
    filters = np.vstack(
        [
            filters,
            np.r_[np.ones(rank), np.zeros(n_features - rank)],
            np.ones(n_features),
        ]
    )
    all_names = names + ["projection", "noisy"]
    streams = np.random.SeedSequence([SEED, 2]).spawn(4)
    rows, raw, weighted_raw, hashes, ranks = [], {}, {}, {}, {}
    for i, regime in enumerate(("smooth", "broadband")):
        rng = np.random.default_rng(streams[2 * i])
        sd = 1 / (1 + 4 * nu) if regime == "smooth" else np.ones(n_features)
        clean = rng.normal(size=(n_test, n_features)) * sd
        noisy = clean + 0.5 * rng.normal(size=clean.shape)
        hashes[regime] = {"clean": _digest(clean), "noisy": _digest(noisy)}
        mse, graph_error = [], []
        for gain in filters:
            squared = (noisy * gain - clean) ** 2
            mse.append(squared.mean(axis=1))
            graph_error.append((squared * nu).mean(axis=1))
        mse, graph_error = np.stack(mse, axis=1), np.stack(graph_error, axis=1)
        raw[regime], weighted_raw[regime] = mse, graph_error
        deltas = mse - mse[:, :1]
        ci = _intervals(deltas, np.random.default_rng(streams[2 * i + 1]), n_boot)
        means = mse.mean(axis=0)
        ranks[regime] = {
            "prime_rank_among_phase_controls": int(
                1 + np.sum(means[3 : 3 + n_surrogates] < means[2])
            ),
            "out_of": n_surrogates + 1,
            "phase_mse_min": float(means[3 : 3 + n_surrogates].min()),
            "phase_mse_max": float(means[3 : 3 + n_surrogates].max()),
        }
        for j, name in enumerate(all_names):
            rows.append(
                {
                    "regime": regime,
                    "arm": name,
                    "mse": float(means[j]),
                    "graph_error": float(graph_error[:, j].mean()),
                    "delta_mse": float(deltas[:, j].mean()),
                    "ci_low": float(ci[0, j]),
                    "ci_high": float(ci[1, j]),
                    "squared_gain_norm": float(np.sum(filters[j] ** 2)),
                }
            )
    return {
        "config": {
            "version": "2.0.0",
            "seed": SEED,
            "features": n_features,
            "rank": rank,
            "n_test": n_test,
            "n_surrogates": n_surrogates,
            "n_boot": n_boot,
            "noise_sd": 0.5,
            "beta": beta,
            "log_grid": [float(s[0]), float(s[-1])],
        },
        "rows": rows,
        "per_signal": raw,
        "graph_error_per_signal": weighted_raw,
        "input_hashes": hashes,
        "phase_ranks": ranks,
        "clock": clock_errors(nu, names, multipliers),
        "waveforms": {
            "s": s,
            "relative_error": error,
            "multipliers": multipliers,
            "names": names,
        },
        "gains": gains,
    }

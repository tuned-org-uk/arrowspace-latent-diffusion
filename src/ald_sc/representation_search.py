"""Bounded development-only search for calibrated spectral representations."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from ald_sc.prime_gating import path_basis, waveforms

REGIMES = ("smooth", "broadband")
SEED = 3407


def _hash(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array, dtype="<f8").tobytes()
    ).hexdigest()


def generate_split(name: str, n_features: int, count: int) -> dict:
    """Generate an independent split under a new, fixed seed namespace."""
    split = {"train": 0, "development": 1, "audit": 2}[name]
    nu, basis = path_basis(n_features)
    streams = np.random.SeedSequence([SEED, 20, split]).spawn(2)
    result = {}
    for regime, stream in zip(REGIMES, streams, strict=True):
        rng = np.random.default_rng(stream)
        sd = 1 / (1 + 4 * nu) if regime == "smooth" else np.ones(n_features)
        clean = (rng.normal(size=(count, n_features)) * sd) @ basis.T
        observed = clean + 0.5 * rng.normal(size=clean.shape)
        result[regime] = {"clean": clean, "observed": observed}
    return result


def constrained_gains(a: np.ndarray, cross: np.ndarray, budget: float) -> np.ndarray:
    """Minimise a diagonal quadratic risk on a fixed nonnegative gain sphere.

    Args:
        a: Strictly positive observed second moments.
        cross: Strictly positive observed/clean cross moments.
        budget: Required sum of squared gains.

    The unique positive stationary solution has gain cross/(a + multiplier).
    A monotone bisection determines its common Lagrange multiplier.
    """
    a, cross = np.asarray(a, dtype=float), np.asarray(cross, dtype=float)
    if (
        a.shape != cross.shape
        or a.ndim != 1
        or not np.isfinite(a).all()
        or not np.isfinite(cross).all()
        or np.any(a <= 0)
        or np.any(cross <= 0)
        or not np.isfinite(budget)
        or budget <= 0
    ):
        raise ValueError("positive finite moments and budget are required")
    lower = np.nextafter(-a.min(), np.inf)
    upper = float(np.max(a) + np.linalg.norm(cross) / np.sqrt(budget))
    for _ in range(100):
        middle = (lower + upper) / 2
        norm = np.sum((cross / (a + middle)) ** 2)
        if norm > budget:
            lower = middle
        else:
            upper = middle
    gains = cross / (a + (lower + upper) / 2)
    return gains * np.sqrt(budget / np.sum(gains**2))


def development_score(errors: dict, reference: dict) -> tuple[float, bool]:
    """Balance both regime ratios and reject any more than 2% worse than heat."""
    ratios = np.array([errors[regime] / reference[regime] for regime in REGIMES])
    return float(ratios.mean()), bool(
        np.isfinite(ratios).all() and np.all(ratios <= 1.02)
    )


def _normalise(gains: np.ndarray, budget: float) -> np.ndarray:
    return gains * np.sqrt(budget / np.sum(gains**2))


@dataclass(frozen=True)
class _Basis:
    name: str
    vectors: np.ndarray
    rates: np.ndarray
    recipe: dict


def _project(data: dict, basis: np.ndarray) -> dict:
    return {
        regime: {
            "clean": rows["clean"] @ basis,
            "observed": rows["observed"] @ basis,
            "clean_squared": np.sum(rows["clean"] ** 2, axis=1),
        }
        for regime, rows in data.items()
    }


def _errors(projected: dict, gains: np.ndarray, n_features: int) -> dict:
    return {
        regime: (
            values["clean_squared"]
            + np.sum(
                (values["observed"] * gains) ** 2
                - 2 * values["observed"] * gains * values["clean"],
                axis=1,
            )
        )
        / n_features
        for regime, values in projected.items()
    }


def _fit_gains(projected: dict, weights: dict, budget: float) -> np.ndarray:
    a = sum(
        weights[r] * np.mean(projected[r]["observed"] ** 2, axis=0) for r in REGIMES
    )
    cross = sum(
        weights[r] * np.mean(projected[r]["observed"] * projected[r]["clean"], axis=0)
        for r in REGIMES
    )
    return constrained_gains(a, cross, budget)


def _bases(
    training: dict, graph: np.ndarray, nu: np.ndarray, rank: int
) -> tuple[list, dict, dict]:
    from ald_sc.calibrated_metric import fit_feature_metric

    n_features = len(nu)
    reference_laplacian = (graph * nu) @ graph.T

    def make(name, vectors, recipe):
        rates = np.sum(vectors * (reference_laplacian @ vectors), axis=0).clip(min=0)
        return _Basis(name, vectors, rates, recipe)

    bases = [make("graph", graph[:, :rank], {"construction": "analytical-path"})]
    covariance = [
        training[r]["clean"].T @ training[r]["clean"] / len(training[r]["clean"])
        for r in REGIMES
    ]
    for name, matrix in (
        ("pca", sum(covariance)),
        ("balanced-pca", sum(c / np.trace(c) for c in covariance)),
    ):
        _, vectors = np.linalg.eigh(matrix)
        bases.append(make(name, vectors[:, ::-1][:, :rank], {"fit_split": "train"}))
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 21]))
    random, _ = np.linalg.qr(rng.normal(size=(n_features, n_features)))
    bases.append(make("random", random[:, :rank], {"seed_namespace": [SEED, 21]}))
    bases.append(
        make("coordinates", np.eye(n_features)[:, :rank], {"columns": "first-q"})
    )
    calibration = np.concatenate([training[r]["clean"] for r in REGIMES])
    metric_receipts, metric_arrays = {}, {}
    for neighbours in (4, 8):
        metric = fit_feature_metric(
            calibration, neighbours=min(neighbours, n_features - 1)
        )
        values, vectors = np.linalg.eigh(metric.laplacian)
        name = f"calibrated-k{neighbours}"
        factor_error = float(
            np.max(np.abs(metric.factor.T @ metric.factor - metric.matrix))
        )
        encoded, near_zero = metric.encode_normalised(calibration[:16])
        gram = calibration[:16] @ metric.matrix @ calibration[:16].T
        norm = np.sqrt(np.diag(gram))
        score_error = float(
            np.max(np.abs(encoded @ encoded.T - gram / np.outer(norm, norm)))
        )
        metric_receipts[name] = {
            "artifact_id": metric.artifact_id,
            "calibration_hash": _hash(calibration),
            "factor_shape": list(metric.factor.shape),
            "factor_error": factor_error,
            "normalised_score_error": score_error,
            "near_zero_count": int(np.sum(near_zero)),
            "compressed_chart_is_exact_full_metric": False,
            "neighbours": min(neighbours, n_features - 1),
            "epsilon": 0.01,
            "normalisation": "symmetric",
            "norm_floor": 1e-12,
            "coordinate_id": "synthetic-path-features-v1",
        }
        metric_arrays.update(
            {
                f"{name}_{field}": getattr(metric, field)
                for field in ("adjacency", "laplacian", "matrix", "factor")
            }
        )
        for band, index in (
            ("low", np.arange(rank)),
            ("high", np.arange(n_features - 1, n_features - rank - 1, -1)),
        ):
            bases.append(
                _Basis(
                    f"{name}-{band}",
                    vectors[:, index],
                    values[index].clip(min=0),
                    {
                        "metric_id": metric.artifact_id,
                        "band": band,
                        "fit_split": "train",
                    },
                )
            )
    return bases, metric_receipts, metric_arrays


def run_search(
    out: Path,
    log_path: Path,
    *,
    n_features: int = 128,
    rank: int = 64,
    n_train: int = 2048,
    n_dev: int = 2048,
    n_audit: int = 4096,
    max_trials: int = 96,
    n_boot: int = 2000,
) -> dict:
    """Run a bounded keep/discard loop, seal selection, then audit once.

    Existing output directories or log files are rejected to prevent accidental
    overwriting of exposure history. The winning encoder is stored separately
    from the fixed-protocol v2 manuscript artefacts.
    """
    if not 4 <= rank <= n_features or not 1 <= max_trials <= 96:
        raise ValueError("require 4 <= rank <= features and 1 <= trials <= 96")
    if out.exists() or log_path.exists():
        raise FileExistsError(
            "search output or log already exists; preserve its exposure history"
        )
    out.mkdir(parents=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    sources = [
        "autoresearch.md",
        "autoresearch.sh",
        "src/ald_sc/representation_search.py",
        "src/ald_sc/calibrated_metric.py",
        "src/ald_sc/prime_gating.py",
        "src/ald_sc/__init__.py",
        "scripts/run_representation_search.py",
        "pyproject.toml",
        "uv.lock",
    ]
    source_hashes = {
        p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sources
    }
    started = perf_counter()
    training = generate_split("train", n_features, n_train)
    development = generate_split("development", n_features, n_dev)
    nu, graph = path_basis(n_features)
    baseline_gains = np.exp(-nu[:rank])
    budget = float(np.sum(baseline_gains**2))
    baseline_training = _errors(
        _project(training, graph[:, :rank]), baseline_gains, n_features
    )
    weights = {r: 1 / baseline_training[r].mean() for r in REGIMES}
    baseline_dev = _errors(
        _project(development, graph[:, :rank]), baseline_gains, n_features
    )
    reference = {r: float(baseline_dev[r].mean()) for r in REGIMES}
    bases, metric_receipts, metric_arrays = _bases(training, graph, nu, rank)
    np.savez_compressed(out / "calibrated-metrics.npz", **metric_arrays)
    projections = {b.name: _project(development, b.vectors) for b in bases}
    fitted = {
        b.name: _fit_gains(_project(training, b.vectors), weights, budget)
        for b in bases
    }
    np.savez_compressed(
        out / "candidate-bases.npz",
        **{
            f"{b.name}_{field}": value
            for b in bases
            for field, value in (
                ("basis", b.vectors),
                ("rates", b.rates),
                ("fitted_gains", fitted[b.name]),
            )
        },
    )
    by_name = {b.name: b for b in bases}
    _, _, waveform_names, multipliers, _ = waveforms(rank)
    trials, champion = [], None
    champion_basis, champion_gains = bases[0], baseline_gains

    with log_path.open("x") as log:

        def evaluate(basis, gains, recipe):
            nonlocal champion, champion_basis, champion_gains
            if len(trials) >= max_trials:
                return
            tick = perf_counter()
            orthogonality = np.max(
                np.abs(basis.vectors.T @ basis.vectors - np.eye(rank))
            )
            if (
                orthogonality > 1e-10
                or np.any(gains < 0)
                or not np.isfinite(gains).all()
            ):
                raise ValueError("invalid representation")
            if not np.isclose(np.sum(gains**2), budget, rtol=1e-12):
                raise ValueError("gain budget changed")
            values = _errors(projections[basis.name], gains, n_features)
            means = {r: float(values[r].mean()) for r in REGIMES}
            score, eligible = development_score(means, reference)
            keep = eligible and (champion is None or score < champion["score"] - 1e-12)
            row = {
                "trial": len(trials),
                "basis": basis.name,
                "recipe": recipe,
                "basis_recipe": basis.recipe,
                "mse": means,
                "score": score,
                "eligible": eligible,
                "decision": "keep" if keep else "discard",
                "decision_reason": "improved eligible score"
                if keep
                else ("regime guard failed" if not eligible else "no improvement"),
                "basis_hash": _hash(basis.vectors),
                "gain_hash": _hash(gains),
                "coefficient_count": rank,
                "squared_gain_norm": float(np.sum(gains**2)),
                "evaluation_seconds": perf_counter() - tick,
            }
            if keep:
                champion, champion_basis, champion_gains = row, basis, gains.copy()
            trials.append(row)
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()

        evaluate(bases[0], baseline_gains, {"family": "heat", "time": 1.0})
        for basis in bases:
            for time in (0.0, 0.5, 1.0, 2.0):
                if basis.name == "graph" and time == 1.0:
                    continue
                evaluate(
                    basis,
                    _normalise(np.exp(-time * basis.rates), budget),
                    {"family": "heat", "time": time},
                )
            evaluate(basis, fitted[basis.name], {"family": "training-fitted-quadratic"})
        for name in ("prime", "envelope"):
            for strength in (0.5, 1.0, 2.0):
                modulation = 1 + strength * (
                    multipliers[waveform_names.index(name)] - 1
                )
                evaluate(
                    bases[0],
                    _normalise(baseline_gains * modulation, budget),
                    {"family": name, "strength": strength},
                )
        for name in waveform_names[3:]:
            evaluate(
                bases[0],
                _normalise(
                    baseline_gains * multipliers[waveform_names.index(name)], budget
                ),
                {"family": name, "strength": 1.0},
            )
        refinement_basis = by_name[champion["basis"]]
        for time in (0.25, 0.75, 1.25, 1.5, 4.0):
            evaluate(
                refinement_basis,
                _normalise(np.exp(-time * refinement_basis.rates), budget),
                {"family": "heat-refinement", "time": time},
            )
        heat = _normalise(np.exp(-refinement_basis.rates), budget)
        for amount in np.arange(1, 9) / 8:
            gain = _normalise(
                (1 - amount) * heat + amount * fitted[refinement_basis.name], budget
            )
            evaluate(
                refinement_basis,
                gain,
                {"family": "fitted-blend-refinement", "fitted_fraction": float(amount)},
            )

    selection = {**champion, "trial_count": len(trials), "reference_dev_mse": reference}
    selection_path = out / "selection.json"
    selection_path.write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n")
    selection_hash = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    np.savez_compressed(
        out / "representation.npz",
        basis=champion_basis.vectors,
        gains=champion_gains,
        rates=champion_basis.rates,
    )

    # Audit data are first generated after the immutable selection is written.
    audit = generate_split("audit", n_features, n_audit)
    audit_reference = _errors(
        _project(audit, graph[:, :rank]), baseline_gains, n_features
    )
    audit_winner = _errors(
        _project(audit, champion_basis.vectors), champion_gains, n_features
    )
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 22]))
    audit_rows, raw = {}, {}
    for regime in REGIMES:
        delta = audit_winner[regime] - audit_reference[regime]
        boot = np.array(
            [
                delta[rng.integers(0, len(delta), len(delta))].mean()
                for _ in range(n_boot)
            ]
        )
        low, high = np.quantile(boot, [0.025, 0.975])
        audit_rows[regime] = {
            "heat_mse": float(audit_reference[regime].mean()),
            "selected_mse": float(audit_winner[regime].mean()),
            "delta_mse": float(delta.mean()),
            "ci_low": float(low),
            "ci_high": float(high),
        }
        raw[f"{regime}_heat_mse"] = audit_reference[regime]
        raw[f"{regime}_selected_mse"] = audit_winner[regime]
    audit_score, audit_guard = development_score(
        {r: audit_rows[r]["selected_mse"] for r in REGIMES},
        {r: audit_rows[r]["heat_mse"] for r in REGIMES},
    )
    np.savez_compressed(out / "audit-outcomes.npz", **raw)
    for path, digest in source_hashes.items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Source changed during search: {path}")
    report = {
        "config": {
            "seed": SEED,
            "features": n_features,
            "rank": rank,
            "n_train": n_train,
            "n_dev": n_dev,
            "n_audit": n_audit,
            "max_trials": max_trials,
            "n_boot": n_boot,
            "squared_gain_budget": budget,
        },
        "trial_count": len(trials),
        "distinct_basis_gain_hash_pairs": len(
            {(row["basis_hash"], row["gain_hash"]) for row in trials}
        ),
        "trial_count_is_not_unique_representation_count": True,
        "selection": selection,
        "audit": {
            "selection_was_sealed": hashlib.sha256(
                selection_path.read_bytes()
            ).hexdigest()
            == selection_hash,
            "selection_sha256": selection_hash,
            "rows": audit_rows,
            "balanced_ratio": audit_score,
            "regime_guard_passed": audit_guard,
            "does_not_reselect": True,
        },
        "representation": {
            "projector_distance_from_reference": float(
                np.linalg.norm(
                    champion_basis.vectors @ champion_basis.vectors.T
                    - graph[:, :rank] @ graph[:, :rank].T
                )
            ),
            "basis_coefficients": int(champion_basis.vectors.size),
            "gain_coefficients": rank,
            "dictionary_is_shared": True,
        },
        "calibrated_metrics": metric_receipts,
        "data_hashes": {
            split: {
                r: {k: _hash(v) for k, v in rows.items()} for r, rows in data.items()
            }
            for split, data in (
                ("train", training),
                ("development", development),
                ("audit", audit),
            )
        },
        "provenance": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "source_sha256": source_hashes,
            "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
            "artifact_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(out.iterdir())
                if p.is_file()
            },
        },
        "elapsed_seconds": perf_counter() - started,
        "scope": "Best tested fixed-budget synthetic linear reconstruction configuration, not real-image quality",
    }
    (out / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report

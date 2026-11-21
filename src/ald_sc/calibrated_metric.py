"""Calibration-only feature graphs with an exact, frozen positive metric lift.

Graph vertices are feature columns. The full factor has E + F rows for E
undirected edges and F features; it is not a compressed spectral embedding.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _real_rows(values: np.ndarray, name: str) -> FloatArray:
    """Convert a finite real matrix to float64 without discarding imaginary parts."""
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real")
    try:
        rows = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite real matrix") from error
    if rows.ndim != 2 or not np.isfinite(rows).all():
        raise ValueError(f"{name} must be a finite two-dimensional matrix")
    return rows


def _positive_scalar(value: float, name: str) -> float:
    """Validate a positive finite real recipe parameter."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be positive and finite")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _immutable(array: FloatArray) -> FloatArray:
    """Copy into immutable byte storage, preventing write-flag re-enabling."""
    return np.frombuffer(
        np.asarray(array, dtype=np.float64).tobytes(), dtype=np.float64
    ).reshape(array.shape)


@dataclass(frozen=True, eq=False)
class FeatureMetric:
    """Frozen feature graph and exact regularised incidence lift.

    Attributes:
        adjacency: Symmetric positive-cosine edge weights, shape (F, F).
        laplacian: Combinatorial or symmetrically normalised Laplacian.
        matrix: Positive metric, equal to factor.T @ factor.
        factor: Incidence rows followed by sqrt(epsilon) times identity, (E+F, F).
        artifact_id: SHA256 identity of calibration, configuration and edge recipe.
        norm_floor: Transformed norms at or below this value are treated as zero.
    """

    adjacency: FloatArray
    laplacian: FloatArray
    matrix: FloatArray
    factor: FloatArray
    artifact_id: str
    norm_floor: float

    def __post_init__(self) -> None:
        """Detach numeric buffers and freeze their underlying storage."""
        for name in ("adjacency", "laplacian", "matrix", "factor"):
            object.__setattr__(self, name, _immutable(getattr(self, name)))

    def encode_normalised(self, rows: np.ndarray) -> tuple[FloatArray, BoolArray]:
        """Apply the same frozen full factor and normalise each nonzero row.

        Args:
            rows: Finite real evaluation matrix of shape (N, F); N may be zero.

        Returns:
            Encodings of shape (N, E+F) and a Boolean near-zero mask of shape (N,).
            Rows with transformed norm <= norm_floor have exactly zero encodings.

        Raises:
            ValueError: Inputs are invalid or their transform is not representable.
        """
        rows = _real_rows(rows, "rows")
        if rows.shape[1] != self.factor.shape[1]:
            raise ValueError("rows must have the calibration feature dimension")
        try:
            with np.errstate(over="raise", invalid="raise"):
                transformed = rows @ self.factor.T
                norms = np.hypot.reduce(transformed, axis=1)
        except FloatingPointError as error:
            raise ValueError(
                "transformed rows must have finite representable norms"
            ) from error
        near_zero = norms <= self.norm_floor
        encoded = np.zeros_like(transformed)
        nonzero = ~near_zero
        encoded[nonzero] = transformed[nonzero] / norms[nonzero, None]
        return encoded, near_zero


def fit_feature_metric(
    calibration: np.ndarray,
    neighbours: int = 4,
    epsilon: float = 0.01,
    normalisation: str = "symmetric",
    norm_floor: float = 1e-12,
) -> FeatureMetric:
    """Fit a frozen feature graph using calibration rows only.

    Uncentred feature cosines come from calibration.T @ calibration. Each
    feature selects up to k positive-cosine neighbours, breaking equal-weight
    ties by ascending feature index. The undirected union uses the original
    cosine weight once per edge. Edges are ordered lexicographically, with
    positive incidence at the smaller endpoint.

    Args:
        calibration: Finite real matrix with at least two rows and two features.
        neighbours: Integer k in [1, F-1]; fewer edges are allowed if cosines <= 0.
        epsilon: Positive finite identity-lift weight.
        normalisation: "combinatorial" or "symmetric". The latter scales incidence
            columns by inverse square-root degree; isolated columns remain zero.
        norm_floor: Positive finite transformed-norm threshold for zero encodings.

    Returns:
        An immutable float64 metric with the full (E+F, F) incidence factor.
        No item graph, item-pair distances or eigendecomposition is constructed.

    Raises:
        ValueError: Calibration or recipe is invalid, or its Gram matrix overflows.
    """
    calibration = _real_rows(calibration, "calibration")
    samples, features = calibration.shape
    if samples < 2 or features < 2:
        raise ValueError("calibration requires at least two samples and two features")
    if (
        isinstance(neighbours, (bool, np.bool_))
        or not isinstance(neighbours, Integral)
        or not 1 <= neighbours < features
    ):
        raise ValueError("neighbours must be an integer in [1, features-1]")
    neighbours = int(neighbours)
    epsilon = _positive_scalar(epsilon, "epsilon")
    norm_floor = _positive_scalar(norm_floor, "norm_floor")
    if normalisation not in ("combinatorial", "symmetric"):
        raise ValueError("normalisation must be 'combinatorial' or 'symmetric'")

    try:
        with np.errstate(over="raise", invalid="raise"):
            gram = calibration.T @ calibration
    except FloatingPointError as error:
        raise ValueError("calibration Gram matrix must be finite") from error
    if not np.isfinite(gram).all():
        raise ValueError("calibration Gram matrix must be finite")
    norms = np.sqrt(np.diag(gram))
    cosine = np.zeros_like(gram)
    active = np.flatnonzero(norms > 0)
    block = gram[np.ix_(active, active)] / norms[active, None] / norms[None, active]
    cosine[np.ix_(active, active)] = np.clip(block, 0.0, 1.0)
    # One canonical weight per pair also removes last-bit division asymmetry.
    cosine = np.triu(cosine, k=1)
    cosine += cosine.T
    selected = np.zeros((features, features), dtype=bool)
    indices = np.arange(features)
    for feature in range(features):
        order = np.lexsort((indices, -cosine[feature]))
        positive = order[cosine[feature, order] > 0]
        selected[feature, positive[:neighbours]] = True
    adjacency = np.where(selected | selected.T, cosine, 0.0)
    left, right = np.nonzero(np.triu(adjacency, k=1))
    incidence = np.zeros((len(left), features), dtype=np.float64)
    edge_weights = np.sqrt(adjacency[left, right])
    incidence[np.arange(len(left)), left] = edge_weights
    incidence[np.arange(len(left)), right] = -edge_weights
    if normalisation == "symmetric":
        degree = adjacency.sum(axis=1)
        inverse_root = np.zeros(features, dtype=np.float64)
        positive_degree = degree > 0
        inverse_root[positive_degree] = 1 / np.sqrt(degree[positive_degree])
        incidence *= inverse_root
    laplacian = incidence.T @ incidence
    factor = np.vstack([incidence, np.sqrt(epsilon) * np.eye(features)])
    matrix = factor.T @ factor

    calibration_hash = hashlib.sha256()
    calibration_hash.update(np.asarray(calibration.shape, dtype="<i8").tobytes())
    calibration_hash.update(np.ascontiguousarray(calibration, dtype="<f8").tobytes())
    identity = {
        "schema": 1,
        "calibration_sha256": calibration_hash.hexdigest(),
        "neighbours": neighbours,
        "epsilon": epsilon,
        "normalisation": normalisation,
        "norm_floor": norm_floor,
        "edge_recipe": {
            "similarity": "uncentred-feature-gram-positive-cosine",
            "selection": "top-k-positive-ties-ascending-feature-index",
            "symmetrisation": "undirected-union-canonical-upper-triangle-weight",
            "incidence": "lexicographic-edges-positive-at-smaller-index",
            "isolates": "zero-inverse-degree",
            "lift": "stack-incidence-sqrt-epsilon-identity",
            "near_zero": "transformed-norm-less-than-or-equal-to-floor",
            "arithmetic": "float64",
        },
    }
    artifact_id = hashlib.sha256(
        json.dumps(
            identity, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    return FeatureMetric(adjacency, laplacian, matrix, factor, artifact_id, norm_floor)

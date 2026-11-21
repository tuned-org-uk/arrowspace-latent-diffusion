"""Analytical contracts for calibration-only frozen feature metrics."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from ald_sc.calibrated_metric import FeatureMetric, fit_feature_metric


@pytest.mark.parametrize("normalisation", ["combinatorial", "symmetric"])
def test_two_feature_metric_has_the_analytical_incidence_factor(normalisation):
    calibration = np.array([[1.0, 1.0], [0.0, 1.0]])
    metric = fit_feature_metric(
        calibration, neighbours=1, epsilon=0.25, normalisation=normalisation
    )
    weight = 1 / np.sqrt(2)
    np.testing.assert_allclose(metric.adjacency, [[0, weight], [weight, 0]])
    incidence = np.array([[1.0, -1.0]])
    if normalisation == "combinatorial":
        incidence *= np.sqrt(weight)
    expected_factor = np.vstack([incidence, 0.5 * np.eye(2)])
    np.testing.assert_allclose(metric.factor, expected_factor, atol=1e-15)
    np.testing.assert_allclose(metric.laplacian, incidence.T @ incidence, atol=1e-15)
    np.testing.assert_allclose(metric.matrix, metric.laplacian + 0.25 * np.eye(2))
    assert isinstance(metric, FeatureMetric)


@pytest.mark.parametrize("normalisation", ["combinatorial", "symmetric"])
def test_three_feature_isolate_retains_only_the_epsilon_lift(normalisation):
    calibration = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    metric = fit_feature_metric(
        calibration, neighbours=1, epsilon=0.04, normalisation=normalisation
    )
    np.testing.assert_array_equal(metric.adjacency[2], 0)
    np.testing.assert_array_equal(metric.laplacian[2], 0)
    np.testing.assert_allclose(metric.matrix[2], [0, 0, 0.04])
    np.testing.assert_allclose(metric.factor @ np.array([0, 0, 1]), [0, 0, 0, 0.2])
    np.testing.assert_allclose(
        metric.factor.T @ metric.factor, metric.matrix, atol=1e-15
    )
    assert np.linalg.eigvalsh(metric.matrix).min() > 0


@pytest.mark.parametrize(
    "calibration",
    [
        np.zeros((2, 3)),
        np.array([[1.0, -1.0], [2.0, -2.0]]),
        np.eye(3),
    ],
)
def test_empty_positive_graph_is_valid_and_gives_scaled_identity(calibration):
    metric = fit_feature_metric(calibration, neighbours=1, epsilon=0.09)
    features = calibration.shape[1]
    np.testing.assert_array_equal(metric.adjacency, np.zeros((features, features)))
    np.testing.assert_array_equal(metric.laplacian, np.zeros((features, features)))
    np.testing.assert_allclose(metric.factor, 0.3 * np.eye(features))
    np.testing.assert_allclose(metric.matrix, 0.09 * np.eye(features))


def test_equal_cosines_use_feature_index_ties_and_undirected_union():
    metric = fit_feature_metric(
        np.ones((3, 4)), neighbours=1, normalisation="combinatorial"
    )
    expected = np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])
    np.testing.assert_allclose(metric.adjacency, expected, atol=1e-15)
    np.testing.assert_allclose(
        metric.factor[:3],
        [
            [1, -1, 0, 0],
            [1, 0, -1, 0],
            [1, 0, 0, -1],
        ],
        atol=1e-15,
    )
    assert metric.factor.shape == (3 + 4, 4)


@pytest.mark.parametrize("normalisation", ["combinatorial", "symmetric"])
def test_normalised_encoding_matches_metric_cosine_and_quadratic_form(normalisation):
    rng = np.random.default_rng(3407)
    metric = fit_feature_metric(
        rng.normal(size=(20, 5)), neighbours=2, normalisation=normalisation
    )
    rows = rng.normal(size=(7, 5))
    encoded, near_zero = metric.encode_normalised(rows)
    assert not near_zero.any()
    quadratic = rows @ metric.matrix @ rows.T
    norms = np.sqrt(np.diag(quadratic))
    expected = quadratic / np.outer(norms, norms)
    np.testing.assert_allclose(encoded @ encoded.T, expected, atol=1e-14)
    raw = rows @ metric.factor.T
    np.testing.assert_allclose(np.sum(raw**2, axis=1), np.diag(quadratic), atol=1e-14)
    split, split_mask = metric.encode_normalised(rows[3:])
    np.testing.assert_allclose(split, encoded[3:], atol=1e-14)
    np.testing.assert_array_equal(split_mask, near_zero[3:])


def test_near_zero_rows_have_explicit_zero_encoding_without_clamping():
    metric = fit_feature_metric(np.eye(2), neighbours=1, epsilon=1.0, norm_floor=0.01)
    rows = np.array([[0, 0], [0.01, 0], [0.005, 0], [0.02, 0], [3, 4]])
    encoded, near_zero = metric.encode_normalised(rows)
    np.testing.assert_array_equal(near_zero, [True, True, True, False, False])
    np.testing.assert_allclose(encoded, [[0, 0], [0, 0], [0, 0], [1, 0], [0.6, 0.8]])
    assert near_zero.dtype == np.bool_
    assert encoded.dtype == np.float64


def test_encoding_norm_does_not_overflow_or_underflow_from_squaring():
    metric = fit_feature_metric(np.eye(2), neighbours=1, epsilon=1.0, norm_floor=1e-300)
    encoded, mask = metric.encode_normalised(
        np.array([[1e200, 1e200], [1e-200, 1e-200]])
    )
    assert not mask.any()
    np.testing.assert_allclose(encoded, np.full((2, 2), 1 / np.sqrt(2)))


def test_frozen_arrays_do_not_alias_calibration_and_cannot_be_made_writeable():
    calibration = np.array([[1.0, 1.0], [0.0, 1.0]])
    metric = fit_feature_metric(calibration, neighbours=1)
    adjacency = metric.adjacency.copy()
    calibration[:] = 0
    np.testing.assert_array_equal(metric.adjacency, adjacency)
    with pytest.raises(FrozenInstanceError):
        metric.artifact_id = "changed"
    for array in (metric.adjacency, metric.laplacian, metric.matrix, metric.factor):
        assert array.dtype == np.float64
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 99
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_artifact_identity_is_deterministic_and_tracks_calibration_and_recipe():
    calibration = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0]])
    first = fit_feature_metric(calibration, neighbours=1)
    equivalent = fit_feature_metric(
        np.asfortranarray(calibration.astype(">f8")), neighbours=1
    )
    assert first.artifact_id == equivalent.artifact_id
    assert len(first.artifact_id) == 64
    int(first.artifact_id, 16)
    alternatives = [
        fit_feature_metric(calibration * 2, neighbours=1),
        fit_feature_metric(calibration, neighbours=2),
        fit_feature_metric(calibration, neighbours=1, epsilon=0.02),
        fit_feature_metric(calibration, neighbours=1, normalisation="combinatorial"),
        fit_feature_metric(calibration, neighbours=1, norm_floor=1e-10),
    ]
    assert all(other.artifact_id != first.artifact_id for other in alternatives)


@pytest.mark.parametrize(
    "calibration",
    [
        np.ones(3),
        np.ones((1, 3)),
        np.ones((2, 0)),
        np.ones((2, 1)),
        np.ones((2, 2, 2)),
        np.array([[1, np.nan], [0, 1]]),
        np.array([[1, np.inf], [0, 1]]),
        np.ones((2, 2), dtype=complex) * (1 + 1j),
    ],
)
def test_invalid_calibration_is_rejected(calibration):
    with pytest.raises(ValueError):
        fit_feature_metric(calibration, neighbours=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"neighbours": 0},
        {"neighbours": -1},
        {"neighbours": 3},
        {"neighbours": 1.5},
        {"neighbours": True},
        {"epsilon": 0},
        {"epsilon": -1},
        {"epsilon": np.inf},
        {"epsilon": np.nan},
        {"norm_floor": 0},
        {"norm_floor": -1},
        {"norm_floor": np.inf},
        {"norm_floor": np.nan},
        {"normalisation": "invalid"},
    ],
)
def test_invalid_recipe_is_rejected(kwargs):
    config = {"neighbours": 1}
    config.update(kwargs)
    with pytest.raises(ValueError):
        fit_feature_metric(np.eye(3), **config)


@pytest.mark.parametrize(
    "rows",
    [
        np.ones(2),
        np.ones((2, 3)),
        np.array([[np.nan, 0]]),
        np.array([[np.inf, 0]]),
        np.ones((1, 2), dtype=complex) * 1j,
    ],
)
def test_encoding_rejects_invalid_rows(rows):
    metric = fit_feature_metric(np.eye(2), neighbours=1)
    with pytest.raises(ValueError):
        metric.encode_normalised(rows)


def test_empty_evaluation_split_preserves_output_shapes():
    metric = fit_feature_metric(np.eye(2), neighbours=1)
    encoded, near_zero = metric.encode_normalised(np.empty((0, 2)))
    assert encoded.shape == (0, metric.factor.shape[0])
    assert near_zero.shape == (0,)

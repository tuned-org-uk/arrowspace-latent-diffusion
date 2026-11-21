"""Independent invariants for the bounded representation search."""

import numpy as np
import pytest

from ald_sc.representation_search import (
    constrained_gains,
    development_score,
    generate_split,
    run_search,
)


def test_constrained_gains_matches_isotropic_closed_form():
    cross = np.array([1.0, 2.0, 3.0])
    gain = constrained_gains(np.full(3, 2.0), cross, budget=4.0)
    np.testing.assert_allclose(gain, 2 * cross / np.linalg.norm(cross), atol=1e-12)
    assert np.sum(gain**2) == pytest.approx(4.0)


def test_balanced_objective_rejects_sacrificing_one_regime():
    reference = {"smooth": 1.0, "broadband": 10.0}
    score, eligible = development_score({"smooth": 1.03, "broadband": 1.0}, reference)
    assert score == pytest.approx(0.565)
    assert not eligible
    assert development_score(reference, reference) == (1.0, True)


def test_split_streams_are_reproducible_but_independent():
    first = generate_split("train", 16, 32)
    again = generate_split("train", 16, 32)
    development = generate_split("development", 16, 32)
    for regime in first:
        np.testing.assert_array_equal(first[regime]["clean"], again[regime]["clean"])
        assert not np.array_equal(first[regime]["clean"], development[regime]["clean"])


def test_bounded_search_exports_frozen_selection_and_does_not_overwrite(tmp_path):
    output = tmp_path / "run"
    result = run_search(
        output,
        tmp_path / "trials.jsonl",
        n_features=16,
        rank=8,
        n_train=64,
        n_dev=32,
        n_audit=32,
        max_trials=4,
        n_boot=20,
    )
    assert result["trial_count"] == 4
    assert result["selection"]["eligible"]
    assert result["selection"]["score"] <= 1.0
    assert result["audit"]["selection_was_sealed"]
    assert (output / "selection.json").exists()
    with np.load(output / "representation.npz") as artifact:
        assert artifact["basis"].shape == (16, 8)
        np.testing.assert_allclose(
            artifact["basis"].T @ artifact["basis"], np.eye(8), atol=1e-12
        )
        assert (artifact["gains"] >= 0).all()
    with pytest.raises(FileExistsError):
        run_search(output, tmp_path / "other.jsonl", max_trials=1)

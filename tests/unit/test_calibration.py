"""Tests for model evaluation metrics (ModelEvaluator)."""

from __future__ import annotations

import numpy as np
import pytest

from moneygone.models.evaluation import ModelEvaluator


class TestBrierScore:
    """Test the Brier score computation."""

    def test_brier_score_perfect(self) -> None:
        """All predictions correct -> Brier score should be 0."""
        probs = [1.0, 0.0, 1.0, 0.0]
        outcomes = [1, 0, 1, 0]
        score = ModelEvaluator.brier_score(probs, outcomes)
        assert score == pytest.approx(0.0, abs=1e-10)

    def test_brier_score_worst(self) -> None:
        """All predictions maximally wrong -> Brier score should be 1."""
        probs = [0.0, 1.0, 0.0, 1.0]
        outcomes = [1, 0, 1, 0]
        score = ModelEvaluator.brier_score(probs, outcomes)
        assert score == pytest.approx(1.0, abs=1e-10)

    def test_brier_score_uncertain(self) -> None:
        """All predictions at 0.5 -> Brier score should be 0.25."""
        probs = [0.5, 0.5, 0.5, 0.5]
        outcomes = [1, 0, 1, 0]
        score = ModelEvaluator.brier_score(probs, outcomes)
        assert score == pytest.approx(0.25, abs=1e-10)


class TestLogLoss:
    """Test log loss computation with clipping."""

    def test_log_loss_clipping(self) -> None:
        """Probabilities of exactly 0 or 1 should not produce infinity."""
        probs = [0.0, 1.0, 0.0, 1.0]
        outcomes = [0, 1, 1, 0]
        loss = ModelEvaluator.log_loss(probs, outcomes)
        assert np.isfinite(loss), "Log loss should be finite with clipping"
        assert loss > 0, "Log loss should be positive"

    def test_log_loss_perfect(self) -> None:
        """Near-perfect predictions should have very low log loss."""
        probs = [0.999, 0.001, 0.999, 0.001]
        outcomes = [1, 0, 1, 0]
        loss = ModelEvaluator.log_loss(probs, outcomes)
        assert loss < 0.01


class TestExpectedCalibrationError:
    """Test ECE computation."""

    def test_ece_perfect_calibration(self) -> None:
        """Perfectly calibrated predictions should have ECE near 0.

        A perfectly calibrated model means that for all predictions at
        probability p, the true outcome fraction is also p.  We construct
        such data by binning.
        """
        np.random.seed(42)
        n = 5000
        # Generate well-calibrated probabilities
        probs = np.random.uniform(0, 1, n)
        # Outcomes drawn with probability = predicted prob
        outcomes = (np.random.uniform(0, 1, n) < probs).astype(float)

        ece = ModelEvaluator.expected_calibration_error(probs, outcomes, n_bins=10)
        # With 5000 samples, ECE should be small (< 0.03 typically)
        assert ece < 0.05, f"ECE {ece} too large for calibrated predictions"


class TestReliabilityDiagram:
    """Test reliability_diagram output shapes."""

    def test_reliability_diagram_shapes(self) -> None:
        """Returned arrays should have length == n_bins."""
        probs = [0.1, 0.3, 0.5, 0.7, 0.9]
        outcomes = [0, 0, 1, 1, 1]
        n_bins = 5

        centers, fractions, counts = ModelEvaluator.reliability_diagram(
            probs, outcomes, n_bins=n_bins
        )

        assert centers.shape == (n_bins,)
        assert fractions.shape == (n_bins,)
        assert counts.shape == (n_bins,)
        # Total counts should equal number of samples
        assert counts.sum() == len(probs)
        # Centers should be in [0, 1]
        assert np.all(centers >= 0) and np.all(centers <= 1)

"""Tests for the conservative sharp sportsbook probability model."""

from __future__ import annotations

import pytest

from moneygone.models.sharp_sportsbook import SharpSportsbookModel


def test_sharp_sportsbook_model_uses_pinnacle_anchor_with_small_adjustments() -> None:
    model = SharpSportsbookModel()

    prediction = model.predict_proba(
        {
            "pinnacle_win_prob": 0.61,
            "moneyline_movement": 0.03,
            "power_rating_edge": 1.5,
            "team_injury_impact": 0.2,
            "home_field_advantage": 1.0,
        }
    )

    assert prediction.raw_probability == pytest.approx(0.61)
    assert prediction.probability > prediction.raw_probability
    assert 0.50 <= prediction.confidence <= 0.90


def test_sharp_sportsbook_model_falls_back_to_consensus() -> None:
    model = SharpSportsbookModel()

    prediction = model.predict_proba({"sportsbook_win_prob": 0.54})

    assert prediction.raw_probability == pytest.approx(0.54)
    assert prediction.probability == pytest.approx(0.54)

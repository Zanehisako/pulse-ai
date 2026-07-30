from __future__ import annotations

from types import SimpleNamespace

from ml.core.forecast_horizon import (
    ForecastHorizon,
    forecast_selection_target_for_query,
    is_forecast_horizon_routing_query,
    rank_model_scores_for_horizon,
)


def test_fallback_metric_uses_its_own_direction_when_primary_metric_missing():
    runtime = SimpleNamespace(
        model_id="hospital_shortage_predictor_champion",
        aliases=[],
        description="Hospital stockout horizon model.",
        defaults={},
    )

    scores = rank_model_scores_for_horizon(
        "Predict stockout risk for hospital H001 over the next 3 days",
        [runtime],
        ForecastHorizon("short", 3),
        features={"hospital_id": "H001"},
        metric_payloads_fn=lambda _runtime: [
            {
                "overall_metrics": {
                    "accuracy": 0.91,
                }
            }
        ],
    )

    assert scores
    assert scores[0].metric_name == "accuracy"
    assert scores[0].direction == "maximize"
    assert scores[0].rank_score > 0


def test_stockout_regression_metric_is_minimized_when_available():
    runtime = SimpleNamespace(
        model_id="hospital_shortage_predictor_champion",
        aliases=[],
        description="Hospital stockout horizon model.",
        defaults={},
    )

    scores = rank_model_scores_for_horizon(
        "Predict stockout risk for hospital H001 over the next 3 days",
        [runtime],
        ForecastHorizon("short", 3),
        features={"hospital_id": "H001"},
        metric_payloads_fn=lambda _runtime: [
            {
                "overall_metrics": {
                    "root_mean_squared_error": 0.42,
                    "accuracy": 0.91,
                }
            }
        ],
    )

    assert scores
    assert scores[0].metric_name == "root_mean_squared_error"
    assert scores[0].direction == "minimize"
    assert scores[0].metric_score == -0.42


def test_configured_donor_terms_enable_donation_horizon_routing():
    assert is_forecast_horizon_routing_query(
        "Predict donor donation probability in the next 3 months",
        {"donor_id": "D001"},
    )
    assert (
        forecast_selection_target_for_query(
            "Predict donor donation probability in the next 3 months",
            {"donor_id": "D001"},
        )
        == "donation"
    )


def test_donation_target_routes_to_donor_horizon_models():
    donor_runtime = SimpleNamespace(
        model_id="donor_short_horizon_probability",
        aliases=[],
        description="Donor short-horizon forecast.",
        defaults={
            "forecast_selection": {"target": "donation", "primary_metric": "brier_score", "direction": "minimize"},
            "horizon_routing": {
                "enabled": True,
                "timeframes": ["short"],
                "min_days": 0,
                "max_days": 90,
                "target": "donation",
            },
        },
    )
    stockout_runtime = SimpleNamespace(
        model_id="short_horizon_probability",
        aliases=[],
        description="Stockout short-horizon forecast.",
        defaults={
            "forecast_selection": {"target": "stockout", "primary_metric": "brier_score", "direction": "minimize"},
            "horizon_routing": {
                "enabled": True,
                "timeframes": ["short"],
                "min_days": 0,
                "max_days": 7,
                "target": "stockout",
            },
        },
    )

    scores = rank_model_scores_for_horizon(
        "Predict donor donation probability in the next 3 months",
        [donor_runtime, stockout_runtime],
        ForecastHorizon("short", 60),
        features={"donor_id": "D001"},
        target="donation",
        metric_payloads_fn=lambda runtime: [
            {"overall_metrics": {"brier_score": 0.04 if "donor" in runtime.model_id else 0.01}}
        ],
    )

    assert [score.model_id for score in scores] == ["donor_short_horizon_probability"]

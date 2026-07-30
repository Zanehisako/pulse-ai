"""
Tests for inventory.forecast_ml_bridge._rolling_demand and
build_forecast_features.

Uses Django TestCase (real in-memory DB) so BloodSupplySnapshot rows can be
created and queried.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

from django.test import TestCase

from inventory.forecast_ml_bridge import _rolling_demand, build_forecast_features
from inventory.models import BloodSupply, BloodSupplySnapshot, Hospital


def _make_hospital(hospital_id="H_TEST"):
    return Hospital.objects.create(hospital_id=hospital_id, name="Test Hospital")


def _make_supply(hospital, blood_type="O+", stock=100.0, usage=10.0):
    return BloodSupply.objects.create(
        supply_id=f"S_{hospital.hospital_id}_{blood_type}",
        hospital=hospital,
        blood_product_type=blood_type,
        current_stock_units=stock,
        usage_today=usage,
        lead_time_days=3,
        days_since_last_restock=5,
        stockout_count_90d=0,
        scheduled_surgeries_next7d=2,
        event_timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )


def _add_snapshot(hospital, blood_type, usage, days_ago):
    ts = datetime(2025, 1, 1, tzinfo=timezone.utc) - timedelta(days=days_ago)
    BloodSupplySnapshot.objects.create(
        hospital=hospital,
        blood_product_type=blood_type,
        current_stock_units=100.0,
        usage_today=usage,
        days_since_last_restock=5,
        recorded_at=ts,
    )


# ── _rolling_demand ──────────────────────────────────────────────────────────

class RollingDemandFallbackTests(TestCase):
    """When fewer than 2 snapshots exist, falls back to supply.usage_today."""

    def test_no_snapshots_returns_usage_today(self):
        hospital = _make_hospital("H001")
        supply = _make_supply(hospital, usage=15.0)
        self.assertEqual(_rolling_demand(supply, 7), 15.0)

    def test_single_snapshot_returns_usage_today(self):
        hospital = _make_hospital("H002")
        supply = _make_supply(hospital, usage=20.0)
        _add_snapshot(hospital, "O+", usage=5.0, days_ago=1)
        # Only 1 snapshot — below the 2-row threshold
        self.assertEqual(_rolling_demand(supply, 7), 20.0)


class RollingDemandAverageTests(TestCase):
    """When sufficient snapshots exist, returns the mean of those rows."""

    def test_7d_average_across_multiple_snapshots(self):
        hospital = _make_hospital("H003")
        supply = _make_supply(hospital, usage=99.0)  # fallback value, should not be used
        # Add 4 snapshots with known usage values
        for days_ago, usage in enumerate([10.0, 20.0, 30.0, 40.0], start=1):
            _add_snapshot(hospital, "O+", usage=usage, days_ago=days_ago)
        result = _rolling_demand(supply, 7)
        expected = round((10.0 + 20.0 + 30.0 + 40.0) / 4, 2)
        self.assertAlmostEqual(result, expected, places=2)

    def test_window_limits_to_requested_days(self):
        hospital = _make_hospital("H004")
        supply = _make_supply(hospital, usage=0.0)
        # 10 snapshots, most recent 3 have usage=5, older 7 have usage=100
        for i in range(1, 4):
            _add_snapshot(hospital, "O+", usage=5.0, days_ago=i)
        for i in range(4, 11):
            _add_snapshot(hospital, "O+", usage=100.0, days_ago=i)
        # 7-day window: should only see the 7 most recent snapshots
        result = _rolling_demand(supply, 7)
        expected = round((5.0 * 3 + 100.0 * 4) / 7, 2)
        self.assertAlmostEqual(result, expected, places=2)

    def test_different_blood_types_are_isolated(self):
        hospital = _make_hospital("H005")
        supply_o_pos = _make_supply(hospital, blood_type="O+", usage=0.0)
        supply_a_pos = _make_supply(hospital, blood_type="A+", stock=50.0, usage=0.0)
        # Snapshots for O+ only
        for days_ago, usage in enumerate([8.0, 12.0], start=1):
            _add_snapshot(hospital, "O+", usage=usage, days_ago=days_ago)
        # A+ has no snapshots — should fall back to usage_today
        self.assertAlmostEqual(_rolling_demand(supply_o_pos, 7), round((8.0 + 12.0) / 2, 2), places=2)
        self.assertEqual(_rolling_demand(supply_a_pos, 7), 0.0)

    def test_different_hospitals_are_isolated(self):
        h1 = _make_hospital("H006")
        h2 = _make_hospital("H007")
        supply_h1 = _make_supply(h1, usage=0.0)
        supply_h2 = _make_supply(h2, usage=0.0)
        # Snapshots only for h2
        for days_ago, usage in enumerate([50.0, 60.0], start=1):
            _add_snapshot(h2, "O+", usage=usage, days_ago=days_ago)
        # h1 has no snapshots → fallback
        self.assertEqual(_rolling_demand(supply_h1, 7), 0.0)
        self.assertAlmostEqual(_rolling_demand(supply_h2, 7), round((50.0 + 60.0) / 2, 2), places=2)


# ── build_forecast_features ──────────────────────────────────────────────────

class BuildForecastFeaturesSchemaTests(TestCase):
    """Output DataFrame must have the 20 prefixed columns the model expects."""

    EXPECTED_COLS = {
        "forecast_features_features__hospital_id",
        "forecast_features_features__blood_product_type",
        "forecast_features_features__current_stock_units",
        "forecast_features_features__usage_today",
        "forecast_features_features__usage_lag_1",
        "forecast_features_features__usage_lag_3",
        "forecast_features_features__lead_time_days",
        "forecast_features_features__days_since_last_restock",
        "forecast_features_features__stockout_count_90d",
        "forecast_features_features__scheduled_surgeries_next7d",
        "forecast_features_features__historical_demand_7d",
        "forecast_features_features__historical_demand_30d",
        "forecast_features_features__incoming_supply_scheduled_7d",
        "forecast_features_features__incoming_supply_scheduled_30d",
        "forecast_features_features__incoming_supply_day1",
        "forecast_features_features__expiry_rate_7d",
        "forecast_features_features__demand_spike_indicator",
        "forecast_features_features__is_holiday",
        "forecast_features_features__season",
        "forecast_features_features__stock_trend_7d",
    }

    def test_output_has_all_20_columns(self):
        hospital = _make_hospital("H_SCHEMA")
        supply = _make_supply(hospital, stock=150.0, usage=25.0)
        df = build_forecast_features(supply)
        self.assertEqual(set(df.columns), self.EXPECTED_COLS)

    def test_output_has_exactly_one_row(self):
        hospital = _make_hospital("H_ROW")
        supply = _make_supply(hospital)
        df = build_forecast_features(supply)
        self.assertEqual(len(df), 1)

    def test_rolling_demand_used_when_snapshots_present(self):
        hospital = _make_hospital("H_ROLL")
        supply = _make_supply(hospital, usage=99.0)
        # Add 3 snapshots with stable usage of 5.0
        for i in range(1, 4):
            _add_snapshot(hospital, "O+", usage=5.0, days_ago=i)
        df = build_forecast_features(supply)
        # historical_demand_7d must be the snapshot average (5.0), NOT the fallback 99.0
        self.assertAlmostEqual(
            df["forecast_features_features__historical_demand_7d"].iloc[0],
            5.0,
            places=1,
        )

    def test_rolling_demand_falls_back_to_usage_today_when_no_snapshots(self):
        hospital = _make_hospital("H_FB")
        supply = _make_supply(hospital, usage=42.0)
        df = build_forecast_features(supply)
        self.assertAlmostEqual(
            df["forecast_features_features__historical_demand_7d"].iloc[0],
            42.0,
            places=1,
        )

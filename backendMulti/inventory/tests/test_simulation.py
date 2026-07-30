"""
Tests for inventory.simulation — data drift engine.

Config validation tests use SimpleTestCase (no DB).
Tick behaviour tests use TestCase (real in-memory DB via transactions).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from django.test import SimpleTestCase, TestCase, override_settings

from inventory import simulation


# ── Helpers ─────────────────────────────────────────────────────────────────

def _write_config(directory: str, overrides: dict | None = None) -> Path:
    """Write a valid simulation config to a temp directory, with optional overrides."""
    base: dict = {
        "enabled": True,
        "tick_interval_minutes": 10,
        "blood_supply": {
            "consumption_variance_pct": 0.10,
            "restock_probability": 0.08,
            "restock_amount_min": 50,
            "restock_amount_max": 300,
            "min_stock_floor": 0,
        },
        "hospital_features": {
            "temperature_drift_max": 2.0,
            "rain_mm_max": 8.0,
            "scheduled_surgeries_variance_pct": 0.15,
            "trauma_cases_variance_pct": 0.20,
        },
        "donors": {
            "recency_increment": 1,
            "availability_churn_rate": 0.02,
            "availability_recovery_rate": 0.03,
        },
    }
    if overrides:
        base.update(overrides)
    path = Path(directory) / "simulation_config.json"
    path.write_text(json.dumps(base), encoding="utf-8")
    return path


# ── Config validation ────────────────────────────────────────────────────────

class SimulationConfigValidationTests(SimpleTestCase):

    def test_valid_config_loads_without_error(self):
        with TemporaryDirectory() as tmp:
            path = _write_config(tmp)
            with override_settings(PIOS_SIMULATION_CONFIG_PATH=path):
                cfg = simulation._load_simulation_config()
        self.assertTrue(cfg["enabled"])
        self.assertEqual(cfg["tick_interval_minutes"], 10)

    def test_missing_file_raises_runtime_error(self):
        with override_settings(PIOS_SIMULATION_CONFIG_PATH=Path("/nonexistent/path.json")):
            with self.assertRaises(RuntimeError, msg="Simulation config not found"):
                simulation._load_simulation_config()

    def test_invalid_json_raises_runtime_error(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{not valid json", encoding="utf-8")
            with override_settings(PIOS_SIMULATION_CONFIG_PATH=path):
                with self.assertRaises(RuntimeError):
                    simulation._load_simulation_config()

    def test_missing_top_level_field_raises(self):
        for missing_key in ("enabled", "tick_interval_minutes", "blood_supply",
                            "hospital_features", "donors"):
            with self.subTest(missing=missing_key):
                with TemporaryDirectory() as tmp:
                    path = _write_config(tmp)
                    cfg = json.loads(path.read_text())
                    del cfg[missing_key]
                    path.write_text(json.dumps(cfg))
                    with override_settings(PIOS_SIMULATION_CONFIG_PATH=path):
                        with self.assertRaisesRegex(RuntimeError, missing_key):
                            simulation._load_simulation_config()

    def test_missing_blood_supply_subkey_raises(self):
        with TemporaryDirectory() as tmp:
            path = _write_config(tmp)
            cfg = json.loads(path.read_text())
            del cfg["blood_supply"]["restock_probability"]
            path.write_text(json.dumps(cfg))
            with override_settings(PIOS_SIMULATION_CONFIG_PATH=path):
                with self.assertRaisesRegex(RuntimeError, "restock_probability"):
                    simulation._load_simulation_config()

    def test_missing_hospital_features_subkey_raises(self):
        with TemporaryDirectory() as tmp:
            path = _write_config(tmp)
            cfg = json.loads(path.read_text())
            del cfg["hospital_features"]["temperature_drift_max"]
            path.write_text(json.dumps(cfg))
            with override_settings(PIOS_SIMULATION_CONFIG_PATH=path):
                with self.assertRaisesRegex(RuntimeError, "temperature_drift_max"):
                    simulation._load_simulation_config()

    def test_missing_donors_subkey_raises(self):
        with TemporaryDirectory() as tmp:
            path = _write_config(tmp)
            cfg = json.loads(path.read_text())
            del cfg["donors"]["availability_churn_rate"]
            path.write_text(json.dumps(cfg))
            with override_settings(PIOS_SIMULATION_CONFIG_PATH=path):
                with self.assertRaisesRegex(RuntimeError, "availability_churn_rate"):
                    simulation._load_simulation_config()

    def test_get_tick_interval_minutes(self):
        with TemporaryDirectory() as tmp:
            path = _write_config(tmp, {"tick_interval_minutes": 5})
            with override_settings(PIOS_SIMULATION_CONFIG_PATH=path):
                self.assertEqual(simulation.get_tick_interval_minutes(), 5)

    def test_is_simulation_enabled_true(self):
        with TemporaryDirectory() as tmp:
            path = _write_config(tmp, {"enabled": True})
            with override_settings(PIOS_SIMULATION_CONFIG_PATH=path):
                self.assertTrue(simulation.is_simulation_enabled())

    def test_is_simulation_enabled_false(self):
        with TemporaryDirectory() as tmp:
            path = _write_config(tmp, {"enabled": False})
            with override_settings(PIOS_SIMULATION_CONFIG_PATH=path):
                self.assertFalse(simulation.is_simulation_enabled())


# ── Disabled tick returns early ──────────────────────────────────────────────

class SimulationDisabledTests(SimpleTestCase):

    def test_advance_tick_returns_disabled_when_config_disabled(self):
        with TemporaryDirectory() as tmp:
            path = _write_config(tmp, {"enabled": False})
            with override_settings(PIOS_SIMULATION_CONFIG_PATH=path):
                result = simulation.advance_simulation_tick()
        self.assertEqual(result["status"], "disabled")


# ── Tick behaviour — DB tests ────────────────────────────────────────────────

class SimulationTickBloodSupplyTests(TestCase):

    def _make_hospital_and_supply(self, hospital_id="H001", supply_id="S001",
                                  stock=200.0, usage=20.0):
        from inventory.models import BloodSupply, Hospital
        hospital = Hospital.objects.create(
            hospital_id=hospital_id,
            name="Test Hospital",
            wilaya="Québec",
        )
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        supply = BloodSupply.objects.create(
            supply_id=supply_id,
            hospital=hospital,
            blood_product_type="O+",
            current_stock_units=stock,
            usage_today=usage,
            lead_time_days=3,
            days_since_last_restock=5,
            stockout_count_90d=0,
            scheduled_surgeries_next7d=3,
            event_timestamp=ts,
        )
        return supply

    def _run_tick(self, tmp_dir: str) -> dict:
        path = _write_config(tmp_dir)
        with override_settings(PIOS_SIMULATION_CONFIG_PATH=path):
            return simulation.advance_simulation_tick()

    def test_tick_decreases_stock(self):
        supply = self._make_hospital_and_supply(stock=200.0, usage=20.0)
        original_stock = supply.current_stock_units

        with TemporaryDirectory() as tmp:
            # Zero variance + zero restock probability → stock always decreases
            path = Path(tmp) / "cfg.json"
            path.write_text(json.dumps({
                "enabled": True,
                "tick_interval_minutes": 10,
                "blood_supply": {
                    "consumption_variance_pct": 0.0,
                    "restock_probability": 0.0,
                    "restock_amount_min": 50,
                    "restock_amount_max": 50,
                    "min_stock_floor": 0,
                },
                "hospital_features": {
                    "temperature_drift_max": 0.0,
                    "rain_mm_max": 0.0,
                    "scheduled_surgeries_variance_pct": 0.0,
                    "trauma_cases_variance_pct": 0.0,
                },
                "donors": {
                    "recency_increment": 1,
                    "availability_churn_rate": 0.0,
                    "availability_recovery_rate": 0.0,
                },
            }))
            with override_settings(PIOS_SIMULATION_CONFIG_PATH=path):
                result = simulation.advance_simulation_tick()

        supply.refresh_from_db()
        self.assertEqual(result["status"], "ok")
        self.assertLess(supply.current_stock_units, original_stock)

    def test_tick_increments_days_since_restock(self):
        supply = self._make_hospital_and_supply()
        original_days = supply.days_since_last_restock

        with TemporaryDirectory() as tmp:
            path = _write_config(tmp, {})
            # Force restock_probability to 0 so counter always increments
            cfg = json.loads(path.read_text())
            cfg["blood_supply"]["restock_probability"] = 0.0
            path.write_text(json.dumps(cfg))
            with override_settings(PIOS_SIMULATION_CONFIG_PATH=path):
                simulation.advance_simulation_tick()

        supply.refresh_from_db()
        self.assertGreater(supply.days_since_last_restock, original_days)

    def test_tick_resets_restock_counter_on_restock(self):
        self._make_hospital_and_supply(stock=100.0, usage=5.0)

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.json"
            # Force restock every time (probability=1.0)
            path.write_text(json.dumps({
                "enabled": True,
                "tick_interval_minutes": 10,
                "blood_supply": {
                    "consumption_variance_pct": 0.0,
                    "restock_probability": 1.0,
                    "restock_amount_min": 50,
                    "restock_amount_max": 50,
                    "min_stock_floor": 0,
                },
                "hospital_features": {
                    "temperature_drift_max": 0.0,
                    "rain_mm_max": 0.0,
                    "scheduled_surgeries_variance_pct": 0.0,
                    "trauma_cases_variance_pct": 0.0,
                },
                "donors": {
                    "recency_increment": 0,
                    "availability_churn_rate": 0.0,
                    "availability_recovery_rate": 0.0,
                },
            }))
            with override_settings(PIOS_SIMULATION_CONFIG_PATH=path):
                result = simulation.advance_simulation_tick()

        self.assertGreater(result["blood_supply"]["restocked"], 0)

    def test_tick_stock_never_goes_below_floor(self):
        # Start at 1 unit with high usage — stock must clamp at floor=0
        supply = self._make_hospital_and_supply(stock=1.0, usage=100.0)

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.json"
            path.write_text(json.dumps({
                "enabled": True,
                "tick_interval_minutes": 10,
                "blood_supply": {
                    "consumption_variance_pct": 0.0,
                    "restock_probability": 0.0,
                    "restock_amount_min": 0,
                    "restock_amount_max": 0,
                    "min_stock_floor": 0,
                },
                "hospital_features": {
                    "temperature_drift_max": 0.0,
                    "rain_mm_max": 0.0,
                    "scheduled_surgeries_variance_pct": 0.0,
                    "trauma_cases_variance_pct": 0.0,
                },
                "donors": {
                    "recency_increment": 0,
                    "availability_churn_rate": 0.0,
                    "availability_recovery_rate": 0.0,
                },
            }))
            with override_settings(PIOS_SIMULATION_CONFIG_PATH=path):
                simulation.advance_simulation_tick()

        supply.refresh_from_db()
        self.assertGreaterEqual(supply.current_stock_units, 0.0)

    def test_tick_returns_ok_status_with_summary_keys(self):
        self._make_hospital_and_supply()

        with TemporaryDirectory() as tmp:
            path = _write_config(tmp)
            with override_settings(PIOS_SIMULATION_CONFIG_PATH=path):
                result = simulation.advance_simulation_tick()

        self.assertEqual(result["status"], "ok")
        self.assertIn("blood_supply", result)
        self.assertIn("hospital_features", result)
        self.assertIn("donors", result)
        self.assertIn("updated", result["blood_supply"])

    def test_tick_with_empty_db_returns_ok(self):
        with TemporaryDirectory() as tmp:
            path = _write_config(tmp)
            with override_settings(PIOS_SIMULATION_CONFIG_PATH=path):
                result = simulation.advance_simulation_tick()

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["blood_supply"]["updated"], 0)
        self.assertEqual(result["hospital_features"]["updated"], 0)
        self.assertEqual(result["donors"]["updated"], 0)


class SimulationTickDonorTests(TestCase):

    def _make_donor(self, donor_id="D001", recency=30, days_eligible=20, availability=1):
        from inventory.models import Donor
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        return Donor.objects.create(
            donor_id=donor_id,
            name="Test Donor",
            wilaya="Québec",
            event_timestamp=ts,
            city_id="C000",
            lat=46.8,
            lon=-71.2,
            availability=availability,
            blood_group="O+",
            recency_days=recency,
            frequency_365=4,
            days_until_eligible=days_eligible,
            cluster_id=0,
        )

    def _zero_variance_config(self, tmp: str, recency_increment: int = 1) -> Path:
        path = Path(tmp) / "cfg.json"
        path.write_text(json.dumps({
            "enabled": True,
            "tick_interval_minutes": 10,
            "blood_supply": {
                "consumption_variance_pct": 0.0,
                "restock_probability": 0.0,
                "restock_amount_min": 0,
                "restock_amount_max": 0,
                "min_stock_floor": 0,
            },
            "hospital_features": {
                "temperature_drift_max": 0.0,
                "rain_mm_max": 0.0,
                "scheduled_surgeries_variance_pct": 0.0,
                "trauma_cases_variance_pct": 0.0,
            },
            "donors": {
                "recency_increment": recency_increment,
                "availability_churn_rate": 0.0,
                "availability_recovery_rate": 0.0,
            },
        }))
        return path

    def test_recency_increments(self):
        donor = self._make_donor(recency=30)
        with TemporaryDirectory() as tmp:
            path = self._zero_variance_config(tmp, recency_increment=1)
            with override_settings(PIOS_SIMULATION_CONFIG_PATH=path):
                simulation.advance_simulation_tick()
        donor.refresh_from_db()
        self.assertEqual(donor.recency_days, 31)

    def test_days_until_eligible_decrements_to_floor(self):
        donor = self._make_donor(days_eligible=1)
        with TemporaryDirectory() as tmp:
            path = self._zero_variance_config(tmp, recency_increment=5)
            with override_settings(PIOS_SIMULATION_CONFIG_PATH=path):
                simulation.advance_simulation_tick()
        donor.refresh_from_db()
        self.assertEqual(donor.days_until_eligible, 0)

    def test_churn_rate_one_makes_available_donor_unavailable(self):
        donor = self._make_donor(availability=1)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.json"
            path.write_text(json.dumps({
                "enabled": True,
                "tick_interval_minutes": 10,
                "blood_supply": {
                    "consumption_variance_pct": 0.0,
                    "restock_probability": 0.0,
                    "restock_amount_min": 0,
                    "restock_amount_max": 0,
                    "min_stock_floor": 0,
                },
                "hospital_features": {
                    "temperature_drift_max": 0.0,
                    "rain_mm_max": 0.0,
                    "scheduled_surgeries_variance_pct": 0.0,
                    "trauma_cases_variance_pct": 0.0,
                },
                "donors": {
                    "recency_increment": 0,
                    "availability_churn_rate": 1.0,
                    "availability_recovery_rate": 0.0,
                },
            }))
            with override_settings(PIOS_SIMULATION_CONFIG_PATH=path):
                result = simulation.advance_simulation_tick()

        donor.refresh_from_db()
        self.assertEqual(donor.availability, 0)
        self.assertGreater(result["donors"]["churned"], 0)

    def test_recovery_rate_one_makes_unavailable_donor_available(self):
        donor = self._make_donor(availability=0)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.json"
            path.write_text(json.dumps({
                "enabled": True,
                "tick_interval_minutes": 10,
                "blood_supply": {
                    "consumption_variance_pct": 0.0,
                    "restock_probability": 0.0,
                    "restock_amount_min": 0,
                    "restock_amount_max": 0,
                    "min_stock_floor": 0,
                },
                "hospital_features": {
                    "temperature_drift_max": 0.0,
                    "rain_mm_max": 0.0,
                    "scheduled_surgeries_variance_pct": 0.0,
                    "trauma_cases_variance_pct": 0.0,
                },
                "donors": {
                    "recency_increment": 0,
                    "availability_churn_rate": 0.0,
                    "availability_recovery_rate": 1.0,
                },
            }))
            with override_settings(PIOS_SIMULATION_CONFIG_PATH=path):
                result = simulation.advance_simulation_tick()

        donor.refresh_from_db()
        self.assertEqual(donor.availability, 1)
        self.assertGreater(result["donors"]["recovered"], 0)

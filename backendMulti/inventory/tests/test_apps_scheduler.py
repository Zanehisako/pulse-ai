"""
Tests for inventory.apps — APScheduler startup logic.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from django.test import SimpleTestCase, override_settings


class SimulationSchedulerStartupTests(SimpleTestCase):

    def _write_config(self, tmp: str, enabled: bool = True, interval: int = 10) -> Path:
        path = Path(tmp) / "simulation_config.json"
        path.write_text(json.dumps({
            "enabled": enabled,
            "tick_interval_minutes": interval,
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
        }), encoding="utf-8")
        return path

    def test_scheduler_does_not_start_when_disabled(self):
        """When simulation is disabled in config, _start_simulation_scheduler must return early."""
        started_jobs = []

        class _FakeScheduler:
            def add_job(self, *a, **kw):
                started_jobs.append(kw.get("id"))

            def start(self):
                started_jobs.append("start")

        with TemporaryDirectory() as tmp:
            path = self._write_config(tmp, enabled=False)
            with override_settings(PIOS_SIMULATION_CONFIG_PATH=path):
                from unittest.mock import patch
                with patch("apscheduler.schedulers.background.BackgroundScheduler",
                           return_value=_FakeScheduler()):
                    from inventory.apps import _start_simulation_scheduler
                    _start_simulation_scheduler()

        self.assertEqual(started_jobs, [], "Scheduler must not start when simulation is disabled")

    def test_scheduler_starts_with_correct_interval(self):
        """When simulation is enabled, scheduler must be started with the configured interval."""
        add_job_kwargs: list[dict] = []
        start_called = []

        class _FakeScheduler:
            def add_job(self, func, trigger, minutes, id, replace_existing, misfire_grace_time):
                add_job_kwargs.append({"minutes": minutes, "id": id, "trigger": trigger})

            def start(self):
                start_called.append(True)

        with TemporaryDirectory() as tmp:
            path = self._write_config(tmp, enabled=True, interval=7)
            with override_settings(PIOS_SIMULATION_CONFIG_PATH=path):
                from unittest.mock import patch
                with patch("apscheduler.schedulers.background.BackgroundScheduler",
                           return_value=_FakeScheduler()):
                    # import after mock to avoid caching
                    import importlib
                    import inventory.apps as apps_module
                    importlib.reload(apps_module)
                    apps_module._start_simulation_scheduler()

        self.assertEqual(len(add_job_kwargs), 1)
        self.assertEqual(add_job_kwargs[0]["minutes"], 7)
        self.assertEqual(add_job_kwargs[0]["trigger"], "interval")
        self.assertEqual(add_job_kwargs[0]["id"], "simulation_tick")
        self.assertTrue(start_called)

    def test_run_prediction_chain_calls_tick_and_predictions(self):
        """_run_prediction_chain must call advance_simulation_tick and run_configured_scheduled_predictions."""
        calls = []

        def _fake_tick():
            calls.append("tick")
            return {"status": "ok", "blood_supply": {"updated": 0, "restocked": 0},
                    "hospital_features": {"updated": 0}, "donors": {"updated": 0, "churned": 0, "recovered": 0}}

        def _fake_predictions(**kw):
            calls.append("predictions")
            return {"jobs": []}

        def _fake_call_command(cmd, *args, **kwargs):
            calls.append(f"management:{cmd}")

        from unittest.mock import patch
        with patch("inventory.simulation.advance_simulation_tick", _fake_tick), \
             patch("inventory.tasks.run_configured_scheduled_predictions", _fake_predictions), \
             patch("django.core.management.call_command", _fake_call_command):
            from inventory.apps import _run_prediction_chain
            _run_prediction_chain()

        self.assertIn("tick", calls)
        self.assertIn("predictions", calls)
        self.assertIn("management:seed_dashboard_from_inventory", calls)

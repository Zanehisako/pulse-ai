from __future__ import annotations

from simulation_studio.app import standalone_snapshot
from simulation_studio.app.operational_world import reset_operational_world, seed_operational_world


def test_capture_standalone_snapshot_shape(monkeypatch):
    reset_operational_world(seed_operational_world())
    monkeypatch.setattr(
        standalone_snapshot,
        "run_standalone_predictions",
        lambda world, config=None: (
            [
                {
                    "entity_id": "S-1",
                    "model_name": "demo_model",
                    "predicted_value": 0.42,
                    "predicted_for_date": "2026-06-15",
                    "created_at": "2026-06-15T12:00:00+00:00",
                    "alert_triggered": False,
                }
            ],
            [],
        ),
    )
    monkeypatch.setattr(
        standalone_snapshot,
        "load_model_configs_from_catalog",
        lambda payload: [{"model_id": "demo_model", "is_active": True, "updated_at": "x"}],
    )

    snapshot = standalone_snapshot.capture_standalone_operational_snapshot(
        run_predictions=True,
        advance_tick=False,
    )

    assert snapshot["payload"]["predictions"]
    assert snapshot["payload"]["model_configs"]
    assert snapshot["snapshot_id"].startswith("standalone-")
from __future__ import annotations

import sqlite3
from pathlib import Path

from pios_ml_backend.config_db import (
    DEFAULT_CONFIG_KEY,
    init_db,
    load_model_config_snapshot,
    upsert_model_config,
)


def test_runtime_config_is_materialized_as_rows(monkeypatch, tmp_path: Path):
    db_path = tmp_path / "config.db"
    monkeypatch.setenv("PIOS_CONFIG_DB_BACKEND", "sqlite")
    monkeypatch.setenv("PIOS_CONFIG_DB_PATH", str(db_path))

    init_db(config_key=DEFAULT_CONFIG_KEY)

    payload = {
        "models": [
            {
                "id": "donor_agent_final",
                "description": "Individual donor propensity model",
                "file_path": "/tmp/donor_agent_final.pkl",
                "type": "online_agent_snapshot",
                "features": ["age", "bmi"],
                "feature_info": {"age": {"type": "number"}},
                "examples": [{"kind": "good", "user_query": "Is donor age 35 likely to donate?"}],
                "defaults": {"age": 35},
                "enabled": True,
            },
            {
                "id": "days_until_stockout_reg",
                "description": "Stockout horizon model",
                "file_path": "/tmp/days_until_stockout_reg.pkl",
                "type": "regression",
                "features": ["hospital", "stock_start"],
                "feature_info": {},
                "examples": [{"kind": "bad", "user_query": "What is US blood type prevalence?"}],
                "defaults": {},
                "enabled": True,
            },
        ]
    }

    upsert_model_config(payload=payload, config_key=DEFAULT_CONFIG_KEY)

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute("SELECT model_id, description, enabled FROM ml_models ORDER BY model_id")
        rows = cur.fetchall()
    finally:
        conn.close()

    assert len(rows) == 2
    assert rows[0][0] == "days_until_stockout_reg"
    assert rows[1][0] == "donor_agent_final"
    assert rows[0][2] == 1
    assert rows[1][2] == 1

    snapshot = load_model_config_snapshot(config_key=DEFAULT_CONFIG_KEY)
    assert snapshot.source.endswith("ml_models_rows")
    models = snapshot.payload.get("models", [])
    assert len(models) == 2
    donor = next(row for row in models if row.get("id") == "donor_agent_final")
    assert donor.get("description") == "Individual donor propensity model"
    assert isinstance(donor.get("examples"), list)

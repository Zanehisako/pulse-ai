from __future__ import annotations

import os
from pathlib import Path

from fastapi.testclient import TestClient

from simulation_studio.app.main import api_app, app
from simulation_studio.app import simulator_service
from simulation_studio.app import snapshot_layer
from simulation_studio.app import studio_routes
from simulation_studio.app.schemas import DataSnapshotRequest


def isolate_custom_store(tmp_path, monkeypatch):
    custom_store = tmp_path / "custom_scenarios.json"
    custom_store.write_text("[]\n", encoding="utf-8")
    monkeypatch.setattr(simulator_service, "CUSTOM_SCENARIOS_PATH", custom_store)


def isolate_dreamerv3_runs(tmp_path, monkeypatch):
    dreamer_root = tmp_path / "dreamerv3_runs"
    checkpoint_dir = dreamer_root / "pink_demo_run" / "ckpt" / "000001"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "done").write_text("", encoding="utf-8")
    (checkpoint_dir.parent / "latest").write_text("000001\n", encoding="utf-8")
    monkeypatch.setattr(simulator_service, "DREAMERV3_RUNS_ROOT", dreamer_root)


def isolate_learned_policy_models(tmp_path, monkeypatch):
    model_root = tmp_path / "learned_models"
    model_root.mkdir(parents=True, exist_ok=True)
    checkpoints = {
        "PIOS_PPO_CONTINUOUS_CHECKPOINT": model_root / "ppo_continuous.zip",
        "PIOS_SAC_CHECKPOINT": model_root / "sac_continuous.zip",
        "PIOS_IQL_CHECKPOINT": model_root / "iql_offline.pt",
        "PIOS_CQL_CHECKPOINT": model_root / "cql_offline.pt",
    }
    for env_key, checkpoint_path in checkpoints.items():
        checkpoint_path.write_bytes(env_key.encode("utf-8"))
        monkeypatch.setenv(env_key, str(checkpoint_path))


def isolate_snapshot_model_registry(tmp_path, monkeypatch):
    ml_models_root = tmp_path / "ml_models"
    notebook_models_root = tmp_path / "notebook_models"
    simulator_models_root = tmp_path / "simulator_models"
    for root in (ml_models_root, notebook_models_root, simulator_models_root):
        root.mkdir(parents=True, exist_ok=True)

    (simulator_models_root / "sac_blood_model.zip").write_bytes(b"sac")
    (simulator_models_root / "iql_blood_model.pt").write_bytes(b"iql")
    monkeypatch.setattr(snapshot_layer, "ML_MODELS_ROOT", ml_models_root)
    monkeypatch.setattr(snapshot_layer, "NOTEBOOK_MODELS_ROOT", notebook_models_root)
    monkeypatch.setattr(snapshot_layer, "SIMULATOR_MODELS_ROOT", simulator_models_root)


def test_meta_lists_scenarios_and_strategies(tmp_path, monkeypatch):
    isolate_custom_store(tmp_path, monkeypatch)
    isolate_dreamerv3_runs(tmp_path, monkeypatch)
    isolate_learned_policy_models(tmp_path, monkeypatch)
    client = TestClient(app)

    response = client.get("/api/meta")

    assert response.status_code == 200
    payload = response.json()
    assert payload["scenarios"]
    assert payload["strategies"]
    assert payload["dreamerv3_runs"]
    assert payload["default_dreamerv3_run_key"] == "pink_demo_run"
    assert any(scenario["key"] == "baseline" for scenario in payload["scenarios"])
    assert any(
        scenario["key"] == "holiday_flu_wave" and scenario["is_holdout"]
        for scenario in payload["scenarios"]
    )
    strategies = {strategy["key"]: strategy for strategy in payload["strategies"]}
    assert "dreamerv4" not in strategies
    assert strategies["baseline"]["available"] is True
    assert strategies["dreamerv3_official"]["available"] is True
    assert strategies["ppo_continuous"]["available"] is True
    assert strategies["sac_continuous"]["available"] is True
    assert strategies["iql_offline"]["available"] is True
    assert strategies["cql_offline"]["available"] is True
    assert payload["defaults"]["comparison_strategies"] == [
        "ppo_continuous",
        "sac_continuous",
        "iql_offline",
        "cql_offline",
        "dreamerv3_official",
    ]
    assert payload["custom_scenario_editor"]["fields"]
    assert any(
        field["key"] == "episode_budget"
        for field in payload["custom_scenario_editor"]["fields"]
    )
    baseline = next(
        scenario for scenario in payload["scenarios"] if scenario["key"] == "baseline"
    )
    assert "demand_forecast_noise" in baseline["params"]
    assert "demand_forecast_interval_h" in baseline["params"]
    assert "congestion_threshold_units" in baseline["params"]


def test_api_only_app_skips_spa_routes():
    client = TestClient(api_app)

    root_response = client.get("/")
    static_response = client.get("/static/app.js")
    health_response = client.get("/api/health")

    assert root_response.status_code == 404
    assert static_response.status_code == 404
    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok"}


def test_web_app_serves_enterprise_setup_theme():
    client = TestClient(app)

    root_response = client.get("/")
    theme_response = client.get("/static/theme-overrides.css")

    assert root_response.status_code == 200
    assert theme_response.status_code == 200
    assert "enterprise-setup-20260618b" in root_response.text
    assert "Enterprise setup flow" in theme_response.text


def test_enterprise_setup_theme_mirror_matches_source():
    repo_root = Path(__file__).resolve().parents[2]
    source_theme = repo_root / "simulation_studio/app/static/theme-overrides.css"
    public_theme = (
        repo_root
        / "web-app/pios-web/public/simulation-studio/theme-overrides.css"
    )

    if public_theme.exists():
        assert public_theme.read_bytes() == source_theme.read_bytes()


def test_sim_setup_exposes_custom_scenario_editor(tmp_path, monkeypatch):
    isolate_custom_store(tmp_path, monkeypatch)
    isolate_dreamerv3_runs(tmp_path, monkeypatch)
    isolate_learned_policy_models(tmp_path, monkeypatch)
    client = TestClient(app)

    response = client.get("/api/sim/setup")

    assert response.status_code == 200
    payload = response.json()
    editor = payload["custom_scenario_editor"]
    strategy_keys = {strategy["key"] for strategy in payload["strategies"]}
    assert "dreamerv4" not in strategy_keys
    assert {"sac_continuous", "iql_offline", "cql_offline"} <= strategy_keys
    assert editor["groups"]
    assert any(field["key"] == "forced_weather" for field in editor["fields"])
    assert any(field["key"] == "episode_budget" for field in editor["fields"])


def test_create_custom_scenario_and_run_it(tmp_path, monkeypatch):
    isolate_custom_store(tmp_path, monkeypatch)
    isolate_dreamerv3_runs(tmp_path, monkeypatch)
    client = TestClient(app)

    create_response = client.post(
        "/api/custom-scenarios",
        json={
            "name": "Cold snap donor drop",
            "description": "Lower donor turnout during a cold snap.",
            "base_scenario_key": "baseline",
            "recommended_strategy_key": "mass_campaign",
            "sim_hours": 72,
            "donor_show_factor": 0.72,
            "transport_penalty": 1.35,
            "forced_weather": "snow",
        },
    )

    assert create_response.status_code == 200
    scenario_key = create_response.json()["scenario"]["key"]

    run_response = client.post(
        "/api/evaluations/run",
        json={
            "scenario_key": scenario_key,
            "strategy_key": "mass_campaign",
            "hours_override": 48,
            "seed": 101,
            "include_timeline": True,
        },
    )

    assert run_response.status_code == 200
    payload = run_response.json()
    assert payload["scenario"]["key"] == scenario_key
    assert payload["strategy"]["key"] == "mass_campaign"
    assert "summary" in payload
    assert "budget_spent" in payload["summary"]
    assert "total_net_requested" in payload["summary"]
    assert "exact_match_rate" in payload["summary"]
    assert "compatible_substitution_rate" in payload["summary"]
    assert isinstance(payload["centers"], list)


def test_custom_scenario_without_sim_hours_keeps_base_duration(tmp_path, monkeypatch):
    isolate_custom_store(tmp_path, monkeypatch)
    isolate_dreamerv3_runs(tmp_path, monkeypatch)
    client = TestClient(app)

    response = client.post(
        "/api/custom-scenarios",
        json={
            "name": "Donor decline variant",
            "base_scenario_key": "donor_decrease",
            "recommended_strategy_key": "mass_campaign",
            "donor_show_factor": 0.65,
            "episode_budget": 9100,
        },
    )

    assert response.status_code == 200
    scenario = response.json()["scenario"]
    assert scenario["params"]["sim_hours"] == 720
    assert scenario["params"]["episode_budget"] == 9100


def test_compare_endpoint_returns_summary_rows(tmp_path, monkeypatch):
    isolate_custom_store(tmp_path, monkeypatch)
    isolate_dreamerv3_runs(tmp_path, monkeypatch)
    client = TestClient(app)

    response = client.post(
        "/api/evaluations/compare",
        json={
            "scenario_keys": ["baseline", "donor_decrease"],
            "strategy_keys": ["baseline", "mass_campaign"],
            "runs": 1,
            "hours_override": 48,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["summary_rows"]) == 4
    assert len(payload["best_by_scenario"]) == 2
    assert "budget_spent_mean" in payload["summary_rows"][0]
    assert "total_net_requested_mean" in payload["summary_rows"][0]
    assert "exact_match_rate_mean" in payload["summary_rows"][0]
    assert "compatible_substitution_rate_mean" in payload["summary_rows"][0]


def test_load_cached_summary_prefers_newest_supported_path(tmp_path, monkeypatch):
    legacy_path = tmp_path / "strategy_eval_all_summary.csv"
    modern_path = tmp_path / "results" / "strategy_eval_summary.csv"
    modern_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text("strategy,scenario,runs\nbaseline,baseline,1\n", encoding="utf-8")
    modern_path.write_text(
        "strategy,strategy_name,scenario,scenario_name,runs\n"
        "baseline,Baseline,baseline,Normal Operations,2\n",
        encoding="utf-8",
    )
    os.utime(legacy_path, (1, 1))
    os.utime(modern_path, (2, 2))
    monkeypatch.setattr(
        simulator_service,
        "CACHED_SUMMARY_CANDIDATES",
        (legacy_path, modern_path),
    )

    payload = simulator_service.load_cached_summary()

    assert payload["available"] is True
    assert payload["summary_rows"][0]["strategy_name"] == "Baseline"
    assert payload["summary_rows"][0]["runs"] == 2.0


def test_twin_setup_exposes_runtime_configuration(tmp_path, monkeypatch):
    isolate_custom_store(tmp_path, monkeypatch)
    isolate_dreamerv3_runs(tmp_path, monkeypatch)
    isolate_learned_policy_models(tmp_path, monkeypatch)
    client = TestClient(app)

    response = client.get("/api/twin/setup")

    assert response.status_code == 200
    payload = response.json()
    twin_config = payload["twin_config"]
    strategy_keys = {strategy["key"] for strategy in payload["strategies"]}
    assert "dreamerv4" not in strategy_keys
    assert {"sac_continuous", "iql_offline", "cql_offline"} <= strategy_keys
    assert twin_config["tick_interval_s"]["default"] == 2.0
    assert twin_config["default_budget_cycle_hours"] == 168.0
    assert any(
        event["key"] == "demand_surge" for event in twin_config["injectable_events"]
    )
    assert any(
        event["key"] == "transport_disruption"
        for event in twin_config["injectable_events"]
    )
    forecast_panel = payload.get("forecast_panel") or {}
    assert forecast_panel.get("enabled") is True
    assert forecast_panel.get("default_job_id")
    assert isinstance(forecast_panel.get("options"), list)
    assert forecast_panel.get("chart_max_points", 0) >= 1


def test_simulated_state_includes_prediction_count_from_snapshot():
    from simulation_studio.app.ws_digital_twin import _simulated_state_from_step

    state = _simulated_state_from_step(
        {"centers": [], "donor_count_by_center": [], "hour": 1.0},
        [],
        [],
        latest_snapshot={
            "payload": {
                "predictions": [{"predicted_value": 1}, {"predicted_value": 2}],
                "model_configs": [{"model_id": "demo_model"}],
                "drift_reports": [],
                "alerts": [],
            }
        },
    )
    assert state["latest_prediction_count"] == 2
    assert len(state["real_signals"]["predictions"]) == 2


def test_twin_websocket_supports_manual_event_injection(tmp_path, monkeypatch):
    isolate_custom_store(tmp_path, monkeypatch)
    isolate_dreamerv3_runs(tmp_path, monkeypatch)
    client = TestClient(app)

    with client.websocket_connect("/api/twin/ws") as ws:
        ws.send_json(
            {
                "command": "start",
                "scenario_key": "baseline",
                "strategy_key": "baseline",
                "seed": 123,
                "tick_interval_s": 0.25,
                "step_hours": 6,
                "budget_cycle_hours": 24,
                "enable_live_data": False,
                "enable_dynamic_world": False,
            }
        )
        init_msg = ws.receive_json()
        assert init_msg["type"] == "twin_init"

        ws.send_json(
            {
                "command": "inject_event",
                "event_key": "demand_surge",
                "severity": 0.8,
            }
        )

        injected_event = None
        for _ in range(8):
            message = ws.receive_json()
            if (
                message.get("type") == "twin_event"
                and message.get("event", {}).get("event_key") == "demand_surge"
                and message.get("event", {}).get("source") == "manual"
            ):
                injected_event = message
                break

        assert injected_event is not None
        assert injected_event["event"]["name"] == "Demand Surge"
        assert injected_event["event"]["source"] == "manual"
        assert injected_event["event"]["severity"] == 0.8

        ws.send_json({"command": "stop"})


def test_classic_sim_acknowledges_random_event_rate_command(tmp_path, monkeypatch):
    isolate_custom_store(tmp_path, monkeypatch)
    isolate_dreamerv3_runs(tmp_path, monkeypatch)
    client = TestClient(app)

    with client.websocket_connect("/api/sim/ws") as ws:
        ws.send_json(
            {
                "command": "start",
                "scenario_key": "baseline",
                "strategy_key": "baseline",
                "seed": 123,
                "hours_override": 24,
                "step_hours": 6,
                "speed": 1,
            }
        )
        init_msg = ws.receive_json()
        assert init_msg["type"] == "init"

        ws.send_json({"command": "set_random_event_rate", "rate": 0.8})

        info_msg = None
        for _ in range(8):
            message = ws.receive_json()
            if message.get("type") == "info":
                info_msg = message
                break

        assert info_msg is not None
        assert "Digital Twin mode" in info_msg["message"]

        ws.send_json({"command": "stop"})


def test_studio_setup_exposes_backend_only_lab_capabilities(tmp_path, monkeypatch):
    isolate_custom_store(tmp_path, monkeypatch)
    isolate_dreamerv3_runs(tmp_path, monkeypatch)
    isolate_learned_policy_models(tmp_path, monkeypatch)
    monkeypatch.setattr(
        studio_routes,
        "detect_snapshot_backend",
        lambda: {
            "source": "django",
            "adapter": "sql_table",
            "label": "Raw donor table: donneur",
            "donor_table": "donneur",
            "donor_model": None,
        },
    )
    client = TestClient(app)

    response = client.get("/api/studio/setup")

    assert response.status_code == 200
    payload = response.json()
    assert payload["timeline_slider"]["default_months"] == 0
    assert payload["features"]["data_snapshot_layer"] is True
    assert payload["features"]["dashboard_ui"] is False
    assert payload["snapshot_backend"]["adapter"] == "sql_table"
    assert len(payload["default_universes"]) >= 2
    assert payload["policy_selection"]["default_policies"] == len(
        payload["default_universes"]
    )
    assert payload["policy_selection"]["min_policies"] == 2
    assert payload["policy_selection"]["max_policies"] >= 2
    policy_keys = {policy["key"] for policy in payload["policy_catalog"]}
    assert "mass_campaign" in policy_keys
    assert "sac_continuous" in policy_keys
    assert "dreamerv4" not in policy_keys
    assert [universe["strategy_key"] for universe in payload["default_universes"]] == [
        "ppo_continuous",
        "sac_continuous",
        "iql_offline",
        "cql_offline",
        "dreamerv3_official",
    ]
    assert {universe["label"] for universe in payload["default_universes"]} == {
        "PPO Continuous",
        "SAC Continuous",
        "IQL Offline",
        "CQL Offline",
        "Official DreamerV3",
    }


def test_studio_dashboard_data_uses_operational_rows(monkeypatch):
    class OperationalRows:
        redacted = {
            "hospitals": [
                {"hospital_id": "H-1", "name": "General Hospital", "wilaya": "North"},
                {"hospital_id": "H-2", "name": "Care Hospital", "wilaya": "South"},
            ],
            "blood_supplies": [
                {
                    "hospital_id": "H-1",
                    "hospital_name": "General Hospital",
                    "blood_product_type": "O+",
                    "current_stock_units": 12,
                    "usage_today": 3,
                    "event_timestamp": "2026-06-12T10:00:00+00:00",
                },
                {
                    "hospital_id": "H-2",
                    "hospital_name": "Care Hospital",
                    "blood_product_type": "A+",
                    "current_stock_units": 18,
                    "usage_today": 2,
                    "event_timestamp": "2026-06-12T10:00:00+00:00",
                },
            ],
            "supply_snapshots": [
                {
                    "hospital_id": "H-1",
                    "blood_product_type": "O+",
                    "current_stock_units": 10,
                    "recorded_at": "2026-06-11T10:00:00+00:00",
                },
                {
                    "hospital_id": "H-1",
                    "blood_product_type": "O+",
                    "current_stock_units": 12,
                    "recorded_at": "2026-06-12T10:00:00+00:00",
                },
            ],
            "predictions": [
                {
                    "predicted_for_date": "2026-06-13",
                    "predicted_value": 7,
                    "alert_triggered": True,
                }
            ],
            "hospital_features": [
                {
                    "hospital_id": "H-1",
                    "event_timestamp": "2026-06-12T10:00:00+00:00",
                    "temperature": 24,
                    "rain_mm": 2,
                }
            ],
            "alerts": [
                {
                    "severity": "critical",
                    "status": "open",
                    "title": "O+ shortage",
                    "message": "O+ shortage at General Hospital",
                    "entity_id": "H-1",
                    "blood_type": "O+",
                    "opened_at": "2026-06-12T09:45:00+00:00",
                }
            ],
        }
        normalized = redacted
        freshness = {"status": "fresh", "captured_at": "2026-06-12T10:00:00+00:00"}
        warnings = []
        watermarks = {}

    monkeypatch.setattr(
        studio_routes,
        "capture_dashboard_operational_rows",
        lambda: OperationalRows(),
    )
    client = TestClient(app)

    response = client.get("/api/studio/dashboard-data")

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"]["status"] == "fresh"
    assert payload["kpis"]["totalInventory"] == 30
    assert payload["kpis"]["criticalShortages"] == 1
    assert payload["bloodTypes"] == [
        {"type": "A+", "pct": 60.0, "units": 18.0},
        {"type": "O+", "pct": 40.0, "units": 12.0},
    ]
    assert payload["routes"] == [
        {
            "sourceFacilityId": "H-2",
            "targetFacilityId": "H-1",
            "sourceName": "Care Hospital",
            "targetName": "General Hospital",
            "midpoint": {"x": 40.0, "y": 26.0},
            "rotation": "-173.7deg",
            "pressure": 20.0,
            "color": "red",
            "d": "M 556.8 173.6 C 460.0 121.6 308.0 190.8 211.2 148.8",
            "animated": True,
            "derivedFrom": "operational_facilities",
        }
    ]
    assert payload["vehicles"] == []
    assert payload["deliveries"]["items"] == []
    assert payload["facilities"][0]["name"] == "General Hospital"
    assert payload["facilities"][0]["critical"] is True
    assert payload["alerts"][0]["text"] == "O+ shortage at General Hospital"
    assert "2026-06-13" in payload["forecast"]["labels"]
    assert payload["weather"] == "24°C Light rain"
    assert payload["simulationSeed"]["enabled"] is True
    assert payload["simulationSeed"]["hospitalCount"] == 2


def test_studio_snapshot_endpoint_supports_synthetic_source(tmp_path, monkeypatch):
    isolate_custom_store(tmp_path, monkeypatch)
    isolate_dreamerv3_runs(tmp_path, monkeypatch)
    isolate_snapshot_model_registry(tmp_path, monkeypatch)
    client = TestClient(app)

    response = client.post(
        "/api/studio/snapshots",
        json={
            "source": "synthetic",
            "donor_limit": 32,
            "include_model_registry": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "synthetic"
    assert payload["donor_summary"]["total_donors"] == 32
    assert payload["calibration"]["estimated_donor_show_factor"] > 0
    model_ids = {entry["model_id"] for entry in payload["model_registry"]}
    assert {"sac_blood_model", "iql_blood_model"} <= model_ids


def test_studio_experiment_runs_parallel_universes_with_agent_mode_and_replay(tmp_path, monkeypatch):
    isolate_custom_store(tmp_path, monkeypatch)
    isolate_dreamerv3_runs(tmp_path, monkeypatch)
    client = TestClient(app)

    response = client.post(
        "/api/studio/experiments/run",
        json={
            "scenario_key": "baseline",
            "seed": 321,
            "replications": 1,
            "step_hours": 24,
            "timeline_months_ahead": 0,
            "include_timeline": True,
            "snapshot": {"source": "synthetic", "donor_limit": 48},
            "agent_mode": {"enabled": True},
            "replay": {
                "mode": "manual",
                "events": [
                    {
                        "key": "dropout_wave",
                        "title": "Dropout Wave",
                        "hour": 48,
                        "duration_hours": 48,
                        "severity": 0.7,
                        "modifiers": {
                            "donor_show_factor": 0.82,
                            "donor_inter_arrival_h": 1.18,
                        },
                    }
                ],
            },
            "universes": [
                {
                    "key": "policy-a",
                    "label": "Policy A",
                    "strategy_key": "baseline",
                },
                {
                    "key": "policy-b",
                    "label": "Policy B",
                    "strategy_key": "mass_campaign",
                    "parameter_multipliers": {"donor_show_factor": 1.08},
                },
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["universes"]) == 2
    assert payload["mission_control"]["branches"]
    assert "summary" in payload["mission_control"]["branches"][0]
    assert "shortage_rate_mean" in payload["mission_control"]["branches"][0]["summary"]
    assert payload["universes"][0]["replications"][0]["runtime_events"]
    assert payload["universes"][0]["replications"][0]["agent_timeline"]
    assert payload["universes"][0]["confidence_bands"]


def test_studio_strategy_assistant_ranks_candidate_interventions(tmp_path, monkeypatch):
    isolate_custom_store(tmp_path, monkeypatch)
    isolate_dreamerv3_runs(tmp_path, monkeypatch)
    client = TestClient(app)

    response = client.post(
        "/api/studio/assistant/strategy",
        json={
            "scenario_key": "baseline",
            "seed": 654,
            "replications": 1,
            "timeline_months_ahead": 0,
            "dropout_rise_pct": 12,
            "snapshot": {"source": "synthetic", "donor_limit": 48},
            "candidate_universes": [
                {
                    "key": "baseline",
                    "label": "Baseline",
                    "strategy_key": "baseline",
                },
                {
                    "key": "campaign",
                    "label": "Campaign",
                    "strategy_key": "mass_campaign",
                },
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "12.0%" in payload["question"]
    assert len(payload["rankings"]) == 2
    assert "summary" in payload["rankings"][0]
    assert "shortage_rate_mean" in payload["rankings"][0]["summary"]
    assert payload["recommendation"]["best_universe"] is not None


def test_studio_experiment_normalizes_duplicate_universe_keys(tmp_path, monkeypatch):
    isolate_custom_store(tmp_path, monkeypatch)
    isolate_dreamerv3_runs(tmp_path, monkeypatch)
    client = TestClient(app)

    response = client.post(
        "/api/studio/experiments/run",
        json={
            "scenario_key": "baseline",
            "seed": 321,
            "replications": 1,
            "timeline_months_ahead": 0,
            "snapshot": {"source": "synthetic", "donor_limit": 48},
            "universes": [
                {
                    "key": "policy-1",
                    "label": "PPO",
                    "strategy_key": "ppo_continuous",
                },
                {
                    "key": "policy-1",
                    "label": "SAC",
                    "strategy_key": "sac_continuous",
                },
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    universe_keys = [universe["key"] for universe in payload["universes"]]
    assert len(universe_keys) == 2
    assert len(set(universe_keys)) == 2
    assert len(payload["mission_control"]["confidence_bands"]) == 2
    assert len(payload["mission_control"]["map_layers"]) == 2


def test_studio_experiment_matches_classic_run_when_extended_features_are_disabled(
    tmp_path, monkeypatch
):
    isolate_custom_store(tmp_path, monkeypatch)
    isolate_dreamerv3_runs(tmp_path, monkeypatch)
    client = TestClient(app)

    classic = client.post(
        "/api/evaluations/run",
        json={
            "scenario_key": "baseline",
            "strategy_key": "mass_campaign",
            "seed": 111,
            "hours_override": 168,
            "include_timeline": False,
        },
    )
    assert classic.status_code == 200
    classic_payload = classic.json()

    studio = client.post(
        "/api/studio/experiments/run",
        json={
            "scenario_key": "baseline",
            "seed": 111,
            "replications": 1,
            "hours_override": 168,
            "step_hours": 6,
            "timeline_months_ahead": 0,
            "include_timeline": False,
            "snapshot": {"source": "synthetic", "donor_limit": 48},
            "agent_mode": {"enabled": False},
            "replay": {"mode": "none"},
            "universes": [
                {
                    "key": "policy-a",
                    "label": "Policy A",
                    "strategy_key": "mass_campaign",
                }
            ],
        },
    )
    assert studio.status_code == 200
    studio_payload = studio.json()
    universe_summary = studio_payload["universes"][0]["summary"]
    replication = studio_payload["universes"][0]["replications"][0]

    assert replication["execution_mode"] == "classic_parity"
    assert (
        abs(
            universe_summary["shortage_rate_mean"]
            - classic_payload["summary"]["shortage_rate"]
        )
        < 1e-4
    )
    assert (
        abs(
            universe_summary["episode_score_mean"]
            - classic_payload["summary"]["episode_score"]
        )
        < 1e-3
    )
    assert (
        abs(
            universe_summary["budget_spent_mean"]
            - classic_payload["summary"]["budget_spent"]
        )
        < 1e-4
    )


def test_discover_dreamerv3_runs_prefers_aligned_run_over_budget_variant(
    tmp_path, monkeypatch
):
    dreamer_root = tmp_path / "dreamerv3_runs"

    aligned_dir = dreamer_root / "m1_continuous_run8_kpi_aligned" / "ckpt" / "000001"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    (aligned_dir / "done").write_text("", encoding="utf-8")

    budget_dir = dreamer_root / "budget_frugal" / "ckpt" / "000001"
    budget_dir.mkdir(parents=True, exist_ok=True)
    (budget_dir / "done").write_text("", encoding="utf-8")

    monkeypatch.setattr(simulator_service, "DREAMERV3_RUNS_ROOT", dreamer_root)

    runs = simulator_service.discover_dreamerv3_runs()

    assert runs[0]["key"] == "m1_continuous_run8_kpi_aligned"


def test_snapshot_layer_prefers_raw_donor_table_when_available(monkeypatch):
    monkeypatch.setattr(snapshot_layer, "table_exists", lambda name: name == "donneur")
    monkeypatch.setattr(
        snapshot_layer,
        "load_sql_donors",
        lambda request, table_name: [
            snapshot_layer.DonorRecord(
                donor_id="D-001",
                display_name="Jane Donor",
                blood_type="O+",
                eligible=True,
                regular_donor=True,
                donation_count=8,
                dropout_risk=0.18,
                recency_days=42.0,
                last_donation_at="2026-02-20T00:00:00+00:00",
            )
        ],
    )
    monkeypatch.setattr(snapshot_layer, "load_django_events", lambda limit: [])
    monkeypatch.setattr(snapshot_layer, "load_django_model_registry", lambda: [])

    snapshot = snapshot_layer.capture_data_snapshot(
        DataSnapshotRequest(source="django", donor_limit=10)
    )

    assert snapshot.source == "django"
    assert snapshot.donor_records[0].donor_id == "D-001"
    assert any("raw donor table" in warning for warning in snapshot.warnings)

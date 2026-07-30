from __future__ import annotations

from simulation_studio.app.policy_snapshot import summarize_policy_events


def test_continuous_decision_headline_and_changes():
    log = [
        {
            "time": 12.0,
            "kind": "continuous_decision",
            "controller": "official_dreamerv3",
            "levels": {"campaign": 0.72, "mobile_unit": 0.0},
            "upserted": [{"action_key": "campaign", "level": 0.72}],
            "deactivated": ["mobile_unit"],
            "routing_updates": [],
            "allocation_updates": [],
        }
    ]
    summary = summarize_policy_events(log, since_index=0, runtime_controller="official_dreamerv3")
    assert summary is not None
    assert summary["kind"] == "continuous_decision"
    assert "donor campaign" in summary["headline"].lower()
    assert summary["has_changes"] is True
    assert summary["changes"][0]["key"] == "campaign"
    assert summary["top_levels"][0]["key"] == "campaign"


def test_discrete_decision_pick():
    log = [
        {
            "time": 6.0,
            "kind": "decision",
            "selected_action": "campaign",
            "activated": True,
            "policy": {"campaign": 0.61, "mobile_unit": 0.12},
        }
    ]
    summary = summarize_policy_events(log, runtime_controller="heuristic_ppo")
    assert summary is not None
    assert summary["discrete_pick"]["key"] == "campaign"
    assert "0.61" in summary["headline"]
    assert summary["has_changes"] is True


def test_idle_when_no_new_decisions_but_controller_active():
    log = [{"time": 1.0, "kind": "idle", "reason": "waiting"}]
    summary = summarize_policy_events(log, since_index=0, runtime_controller="official_dreamerv3")
    assert summary is not None
    assert summary["kind"] == "idle"
    assert summary["has_changes"] is False


def test_returns_none_without_controller_and_empty_slice():
    assert summarize_policy_events([], since_index=0) is None
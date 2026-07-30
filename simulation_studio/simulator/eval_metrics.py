from __future__ import annotations

from typing import Any

DETAIL_METRICS = [
    "episode_score",
    "reward_total",
    "shortage_rate",
    "base_shortage_rate",
    "service_rate",
    "base_service_rate",
    "conservation_rate",
    "donor_attempts",
    "total_shortage",
    "total_transfused",
    "total_donated",
    "total_collected",
    "total_released",
    "total_rejected",
    "total_lab_rejected",
    "total_no_show",
    "rejection_rate",
    "total_base_requested",
    "total_net_requested",
    "total_expired",
    "total_external_units",
    "total_conserved",
    "transfer_units",
    "active_action_cost",
    "active_action_count",
    "controller_decisions",
    "avg_travel_min",
    "avg_wait_min",
    "budget_total",
    "budget_remaining",
    "budget_spent",
    "budget_exhausted_hours",
    "forecast_pressure",
    "recent_priority_pressure",
    "transport_congestion",
    "priority_weighted_shortage",
    "priority_weighted_requested",
    "total_exact_match_units",
    "total_compatible_substitution_units",
    "total_incompatible_fulfillment_units",
    "exact_match_rate",
    "compatible_substitution_rate",
    "incompatible_fulfillment_rate",
    "shortage_rbc",
    "shortage_platelets",
    "shortage_plasma",
]

PLOT_METRICS = [
    ("episode_score_mean", "Episode Score", "viridis", "{:.1f}"),
    ("shortage_rate_mean", "Shortage Rate (%)", "viridis_r", "{:.1f}"),
    ("total_external_units_mean", "External Units", "Reds_r", "{:.0f}"),
    ("active_action_cost_mean", "Action Cost", "YlOrBr_r", "{:.0f}"),
    ("budget_spent_mean", "Budget Spent", "YlOrBr_r", "{:.0f}"),
    ("total_expired_mean", "Total Expired", "Oranges_r", "{:.0f}"),
    ("exact_match_rate_mean", "Exact Match Rate (%)", "Greens", "{:.1f}"),
    (
        "compatible_substitution_rate_mean",
        "Compatible Substitution Rate (%)",
        "PuBu",
        "{:.1f}",
    ),
]

SCENARIO_PLOT_METRICS = [
    ("episode_score_mean", "Episode Score"),
    ("shortage_rate_mean", "Shortage Rate (%)"),
    ("total_external_units_mean", "External Units"),
    ("active_action_cost_mean", "Action Cost"),
    ("budget_spent_mean", "Budget Spent"),
    ("total_expired_mean", "Total Expired"),
    ("exact_match_rate_mean", "Exact Match Rate (%)"),
    ("compatible_substitution_rate_mean", "Compatible Substitution Rate (%)"),
]

AGENT_COMPARE_PLOT_METRICS = [
    ("reward_total_mean", "Reward Total"),
    ("shortage_rate_mean", "Shortage Rate (%)"),
    ("total_net_requested_mean", "Net Requested"),
    ("active_action_cost_mean", "Action Cost"),
    ("budget_spent_mean", "Budget Spent"),
    ("exact_match_rate_mean", "Exact Match Rate (%)"),
]

METRIC_LABELS = {
    "episode_score_mean": "Episode Score",
    "reward_total_mean": "Total Reward",
    "shortage_rate_mean": "Shortage Rate (%)",
    "base_shortage_rate_mean": "Base Shortage Rate (%)",
    "service_rate_mean": "Demand Fulfilled (%)",
    "base_service_rate_mean": "Base Demand Fulfilled (%)",
    "conservation_rate_mean": "Demand Conserved (%)",
    "donor_attempts_mean": "Donor Attempts",
    "total_shortage_mean": "Total Shortage",
    "total_transfused_mean": "Total Transfused",
    "total_donated_mean": "Total Donated (Collected)",
    "total_collected_mean": "Total Collected",
    "total_released_mean": "Total Released to Inventory",
    "total_rejected_mean": "Deferred / Rejected",
    "total_lab_rejected_mean": "Lab Rejected (Post-Collection)",
    "total_no_show_mean": "No-Shows",
    "rejection_rate_mean": "Rejection Rate (%)",
    "total_base_requested_mean": "Base Requested",
    "total_net_requested_mean": "Net Requested",
    "total_expired_mean": "Total Expired",
    "total_external_units_mean": "External Units",
    "total_conserved_mean": "Conserved Units",
    "transfer_units_mean": "Transfer Units",
    "active_action_cost_mean": "Action Cost",
    "active_action_count_mean": "Active Action Count",
    "controller_decisions_mean": "Controller Decisions",
    "avg_travel_min_mean": "Average Travel (min)",
    "avg_wait_min_mean": "Average Wait (min)",
    "budget_total_mean": "Budget Total",
    "budget_remaining_mean": "Budget Remaining",
    "budget_spent_mean": "Budget Spent",
    "budget_exhausted_hours_mean": "Budget Exhausted (h)",
    "forecast_pressure_mean": "Forecast Pressure",
    "recent_priority_pressure_mean": "Priority Pressure",
    "transport_congestion_mean": "Transport Congestion",
    "priority_weighted_shortage_mean": "Priority-Weighted Shortage",
    "priority_weighted_requested_mean": "Priority-Weighted Requested",
    "total_exact_match_units_mean": "Exact Match Units",
    "total_compatible_substitution_units_mean": "Compatible Substitution Units",
    "total_incompatible_fulfillment_units_mean": "Incompatible Fulfillment Units",
    "exact_match_rate_mean": "Exact Match Rate (%)",
    "compatible_substitution_rate_mean": "Compatible Substitution Rate (%)",
    "incompatible_fulfillment_rate_mean": "Incompatible Fulfillment Rate (%)",
    "shortage_rbc_mean": "RBC Shortage",
    "shortage_platelets_mean": "Platelet Shortage",
    "shortage_plasma_mean": "Plasma Shortage",
}


def extract_eval_metrics(summary: dict[str, Any], state) -> dict[str, float | int]:
    total_transfused = float(summary["total_transfused"])
    total_base_requested = float(summary["total_base_requested"])
    total_net_requested = float(summary["total_net_requested"])
    total_conserved = float(summary["total_conserved"])
    total_exact_match_units = float(summary["total_exact_match_units"])
    total_compatible_substitution_units = float(
        summary["total_compatible_substitution_units"]
    )
    total_incompatible_fulfillment_units = float(
        summary["total_incompatible_fulfillment_units"]
    )
    shortage_by_component = summary["shortage_by_component"]
    # total_donated now equals total_collected (counted at blood draw);
    # total_released is units that passed quarantine and entered inventory.
    total_collected = float(
        summary.get("total_collected", 0) or summary["total_donated"]
    )
    total_released = float(summary.get("total_released", 0))
    total_lab_rejected = float(summary.get("total_lab_rejected", 0))
    donor_attempts = float(
        summary["total_donated"] + summary["total_rejected"] + summary["total_no_show"]
    )
    service_rate = total_transfused / max(total_net_requested, 1.0) * 100.0
    base_service_rate = total_transfused / max(total_base_requested, 1.0) * 100.0
    conservation_rate = total_conserved / max(total_base_requested, 1.0) * 100.0
    exact_match_rate = total_exact_match_units / max(total_transfused, 1.0) * 100.0
    compatible_substitution_rate = (
        total_compatible_substitution_units / max(total_transfused, 1.0) * 100.0
    )
    incompatible_fulfillment_rate = (
        total_incompatible_fulfillment_units / max(total_transfused, 1.0) * 100.0
    )

    return {
        "episode_score": float(summary["episode_score"]),
        "reward_total": float(summary["reward"].total),
        "shortage_rate": float(summary["shortage_rate"]),
        "base_shortage_rate": float(summary["base_shortage_rate"]),
        "service_rate": float(service_rate),
        "base_service_rate": float(base_service_rate),
        "conservation_rate": float(conservation_rate),
        "donor_attempts": int(donor_attempts),
        "total_shortage": int(summary["total_shortage"]),
        "total_transfused": int(summary["total_transfused"]),
        "total_donated": int(summary["total_donated"]),
        "total_collected": int(total_collected),
        "total_released": int(total_released),
        "total_rejected": int(summary["total_rejected"]),
        "total_lab_rejected": int(total_lab_rejected),
        "total_no_show": int(summary["total_no_show"]),
        "rejection_rate": float(summary["rejection_rate"]),
        "total_base_requested": int(summary["total_base_requested"]),
        "total_net_requested": int(summary["total_net_requested"]),
        "total_expired": int(summary["total_expired"]),
        "total_external_units": int(summary["total_external_units"]),
        "total_conserved": int(summary["total_conserved"]),
        "transfer_units": int(state.total_transfer_units),
        "active_action_cost": float(summary["active_action_cost"]),
        "active_action_count": int(len(summary["active_action_keys"])),
        "controller_decisions": int(summary["controller_decisions"]),
        "avg_travel_min": float(summary["avg_travel_min"]),
        "avg_wait_min": float(summary["avg_wait_min"]),
        "budget_total": float(summary["budget_total"]),
        "budget_remaining": float(summary["budget_remaining"]),
        "budget_spent": float(summary["budget_spent"]),
        "budget_exhausted_hours": float(summary["budget_exhausted_hours"]),
        "forecast_pressure": float(summary["forecast_pressure"]),
        "recent_priority_pressure": float(summary["recent_priority_pressure"]),
        "transport_congestion": float(summary["transport_congestion"]),
        "priority_weighted_shortage": float(summary["priority_weighted_shortage"]),
        "priority_weighted_requested": float(summary["priority_weighted_requested"]),
        "total_exact_match_units": int(summary["total_exact_match_units"]),
        "total_compatible_substitution_units": int(
            summary["total_compatible_substitution_units"]
        ),
        "total_incompatible_fulfillment_units": int(
            summary["total_incompatible_fulfillment_units"]
        ),
        "exact_match_rate": float(exact_match_rate),
        "compatible_substitution_rate": float(compatible_substitution_rate),
        "incompatible_fulfillment_rate": float(incompatible_fulfillment_rate),
        "shortage_rbc": int(shortage_by_component.get("RBC", 0)),
        "shortage_platelets": int(shortage_by_component.get("PLATELETS", 0)),
        "shortage_plasma": int(shortage_by_component.get("PLASMA", 0)),
    }

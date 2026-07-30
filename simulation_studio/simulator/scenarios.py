"""
scenarios.py - Scenario definitions, action catalogue and reward settings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from calibration import (
    QC_AVG_UNITS_PER_ORDER_EST,
    QC_BASE_BUFFER_DAYS,
    QC_BASE_DEMAND_INTERARRIVAL_H,
    QC_BASE_DONOR_INTERARRIVAL_H,
    QC_DAILY_COMPLETED_DONATIONS_EST,
    QC_DAILY_HOSPITAL_ORDERS_EST,
    QC_DAILY_LABILE_PRODUCTS_EST,
    REAL_WORLD_NOTES,
)


@dataclass(frozen=True)
class Action:
    key: str
    name: str
    description: str
    operational_cost: float = 0.0
    donor_arrival_factor: float = 1.0
    donor_show_up_factor: float = 1.0
    donor_eligibility_factor: float = 1.0
    demand_management_factor: float = 1.0
    committed_donor_share: float = 0.0
    extra_nurses: int = 0
    extra_lab_staff: int = 0
    extra_processing_staff: int = 0
    lab_speed_factor: float = 1.0
    release_delay_factor: float = 1.0
    mobile_release_delay_h: float = 0.0
    mobile_units: int = 0
    hours_extension_h: float = 0.0
    emergency_share: bool = False
    reserve_threshold: Optional[int] = None
    transport_relief_factor: float = 1.0
    regional_replenishment_factor: float = 1.0
    replenishment_interval_h: Optional[float] = None
    activation_delay_h: float = 0.0
    ramp_h: float = 0.0
    active_h: Optional[float] = None


ACTION_CATALOG = {
    "campaign": Action(
        key="campaign",
        name="Targeted donor campaign",
        description=(
            "Trigger SMS, appointment outreach and registry callbacks. It sharply lifts "
            "arrivals and show-up, but also brings in more marginal first-time donors."
        ),
        operational_cost=60,
        donor_arrival_factor=0.50,
        donor_show_up_factor=1.20,
        donor_eligibility_factor=0.93,
        committed_donor_share=0.42,
        activation_delay_h=12.0,
        ramp_h=120.0,
        active_h=120.0,
    ),
    "mobile_unit": Action(
        key="mobile_unit",
        name="Deploy one mobile unit",
        description=(
            "Stand up two temporary collection sites near dense neighbourhood demand. "
            "Access improves, but mobile collections take longer to release into inventory."
        ),
        operational_cost=55,
        donor_arrival_factor=0.80,
        mobile_units=2,
        mobile_release_delay_h=12.0,
        activation_delay_h=24.0,
        ramp_h=96.0,
        active_h=120.0,
    ),
    "extend_hours": Action(
        key="extend_hours",
        name="Extend donor-centre hours",
        description=(
            "Open fixed sites three extra hours on both ends. This captures more donors "
            "but also raises staffing fatigue unless surge staff are added."
        ),
        operational_cost=35,
        donor_arrival_factor=0.88,
        donor_show_up_factor=1.05,
        hours_extension_h=3.0,
        activation_delay_h=12.0,
        ramp_h=72.0,
        active_h=72.0,
    ),
    "lab_fast_track": Action(
        key="lab_fast_track",
        name="Fast-track lab workflow",
        description=(
            "Fund rapid testing, prioritized component release and emergency QC lanes so "
            "fresh donations become usable much sooner."
        ),
        operational_cost=75,
        lab_speed_factor=0.30,
        release_delay_factor=0.18,
        activation_delay_h=6.0,
        ramp_h=48.0,
        active_h=72.0,
    ),
    "surge_staff": Action(
        key="surge_staff",
        name="Add surge collection and lab staff",
        description=(
            "Staff all active collection sites with extra nurses, lab benches and "
            "processing staff to prevent campaign volume from clogging the system."
        ),
        operational_cost=70,
        extra_nurses=4,
        extra_lab_staff=2,
        extra_processing_staff=2,
        activation_delay_h=12.0,
        ramp_h=48.0,
        active_h=72.0,
    ),
    "emergency_share": Action(
        key="emergency_share",
        name="Emergency sharing and courier routing",
        description=(
            "Allow deeper reserve use, prioritize hospital-bound transfers and unlock "
            "emergency provincial courier support when a hospital is at risk."
        ),
        operational_cost=65,
        emergency_share=True,
        reserve_threshold=0,
        transport_relief_factor=0.45,
        regional_replenishment_factor=8.00,
        replenishment_interval_h=4.0,
        activation_delay_h=24.0,
        ramp_h=96.0,
        active_h=60.0,
    ),
    "clinical_conservation": Action(
        key="clinical_conservation",
        name="Hospital blood conservation protocol",
        description=(
            "Activate restrictive transfusion thresholds, postpone elective cases and "
            "triage scarce blood to highest-acuity patients first."
        ),
        operational_cost=55,
        demand_management_factor=0.28,
        activation_delay_h=6.0,
        ramp_h=36.0,
        active_h=48.0,
    ),
    "national_mutual_aid": Action(
        key="national_mutual_aid",
        name="Provincial and national mutual aid",
        description=(
            "Escalate beyond the local network to bring in sustained outside support, "
            "including inter-regional stock, extra courier lanes and coordinated reserve use."
        ),
        operational_cost=85,
        transport_relief_factor=0.80,
        regional_replenishment_factor=1.80,
        activation_delay_h=48.0,
        ramp_h=120.0,
        active_h=120.0,
    ),
    "rapid_courier": Action(
        key="rapid_courier",
        name="Rapid courier and priority routing",
        description=(
            "Stand up dedicated courier lanes, donor shuttle capacity and priority routing "
            "so donors, existing inventory and provincial inbound stock keep moving during "
            "winter disruptions."
        ),
        operational_cost=48,
        donor_arrival_factor=0.82,
        donor_show_up_factor=1.12,
        transport_relief_factor=0.35,
        regional_replenishment_factor=1.75,
        replenishment_interval_h=4.0,
        activation_delay_h=2.0,
        ramp_h=12.0,
        active_h=36.0,
    ),
}


@dataclass(frozen=True)
class RewardWeights:
    fulfilled_unit: float = 12.0
    shortage_rbc: float = -80.0
    shortage_platelets: float = -70.0
    shortage_plasma: float = -40.0
    priority_shortage_unit: float = -45.0
    conserved_unit: float = -4.0
    expired_unit: float = -18.0
    transfer_unit: float = -7.0
    exact_match_unit: float = 1.5
    compatible_substitution_unit: float = 0.35
    incompatible_fulfillment_unit: float = -10.0
    action_cost_unit: float = -1.2
    budget_spent_unit: float = -0.25
    budget_blocked_control: float = -20.0
    budget_exhausted_hour: float = -18.0
    avg_wait_minute: float = -0.05
    avg_travel_minute: float = -0.03


@dataclass(frozen=True)
class Strategy:
    name: str
    description: str
    action_keys: tuple[str, ...] = field(default_factory=tuple)
    controller_key: Optional[str] = None
    controller_action_keys: tuple[str, ...] = field(default_factory=tuple)
    controller_interval_h: float = 6.0

    @property
    def actions(self) -> tuple[Action, ...]:
        return tuple(ACTION_CATALOG[key] for key in self.action_keys)

    @property
    def action_space(self) -> tuple[Action, ...]:
        keys = self.controller_action_keys or self.action_keys
        return tuple(ACTION_CATALOG[key] for key in keys)

    @property
    def is_adaptive(self) -> bool:
        return self.controller_key is not None

    @property
    def action_cost(self) -> float:
        return sum(action.operational_cost for action in self.actions)

    @property
    def donor_arrival_factor(self) -> float:
        return math.prod(action.donor_arrival_factor for action in self.actions) or 1.0

    @property
    def donor_show_up_factor(self) -> float:
        return math.prod(action.donor_show_up_factor for action in self.actions) or 1.0

    @property
    def donor_eligibility_factor(self) -> float:
        return (
            math.prod(action.donor_eligibility_factor for action in self.actions) or 1.0
        )

    @property
    def committed_donor_share(self) -> float:
        return min(sum(action.committed_donor_share for action in self.actions), 0.90)

    @property
    def demand_management_factor(self) -> float:
        return (
            math.prod(action.demand_management_factor for action in self.actions) or 1.0
        )

    @property
    def extra_nurses(self) -> int:
        return sum(action.extra_nurses for action in self.actions)

    @property
    def extra_lab_staff(self) -> int:
        return sum(action.extra_lab_staff for action in self.actions)

    @property
    def extra_processing_staff(self) -> int:
        return sum(action.extra_processing_staff for action in self.actions)

    @property
    def lab_speed_factor(self) -> float:
        return math.prod(action.lab_speed_factor for action in self.actions) or 1.0

    @property
    def release_delay_factor(self) -> float:
        return math.prod(action.release_delay_factor for action in self.actions) or 1.0

    @property
    def mobile_release_delay_h(self) -> float:
        return sum(action.mobile_release_delay_h for action in self.actions)

    @property
    def mobile_units(self) -> int:
        return sum(action.mobile_units for action in self.actions)

    @property
    def reserve_threshold(self) -> int:
        thresholds = [
            action.reserve_threshold
            for action in self.actions
            if action.reserve_threshold is not None
        ]
        return min(thresholds) if thresholds else 4

    @property
    def emergency_share(self) -> bool:
        return any(action.emergency_share for action in self.actions)

    @property
    def hours_extension_h(self) -> float:
        return sum(action.hours_extension_h for action in self.actions)

    @property
    def transport_relief_factor(self) -> float:
        return (
            math.prod(action.transport_relief_factor for action in self.actions) or 1.0
        )

    @property
    def regional_replenishment_factor(self) -> float:
        return (
            math.prod(action.regional_replenishment_factor for action in self.actions)
            or 1.0
        )

    @property
    def replenishment_interval_h(self) -> float:
        intervals = [
            action.replenishment_interval_h
            for action in self.actions
            if action.replenishment_interval_h is not None
        ]
        return min(intervals) if intervals else 12.0


STRATEGIES = {
    "baseline": Strategy(
        name="Baseline",
        description="Run the network with standing resources only.",
    ),
    "mass_campaign": Strategy(
        name="Mass Donor Campaign",
        description=(
            "Pair a targeted donor campaign with longer hours and one temporary mobile "
            "collection unit."
        ),
        action_keys=("campaign", "mobile_unit", "extend_hours"),
    ),
    "lab_investment": Strategy(
        name="Lab Infrastructure Investment",
        description=(
            "Invest in faster lab turnaround so collected blood becomes available sooner."
        ),
        action_keys=("lab_fast_track",),
    ),
    "emergency_network": Strategy(
        name="Emergency Sharing Network",
        description=(
            "Add surge staffing, hospital conservation protocols and emergency transfers "
            "so scarce stock can be conserved and routed to the highest-priority need."
        ),
        action_keys=("surge_staff", "emergency_share", "clinical_conservation"),
    ),
    "full_response": Strategy(
        name="Full Crisis Response",
        description=(
            "Activate campaign, mobile collection, longer hours, fast lab processing, "
            "hospital conservation and emergency transfers together."
        ),
        action_keys=(
            "campaign",
            "mobile_unit",
            "extend_hours",
            "lab_fast_track",
            "surge_staff",
            "emergency_share",
            "clinical_conservation",
            "national_mutual_aid",
        ),
    ),
    "ppo_shortage_minimizer": Strategy(
        name="PPO Shortage Minimizer",
        description=(
            "Adaptive PPO-style controller that watches live shortages, reserve coverage "
            "and transport pressure, then activates the least-cost shortage-mitigation "
            "actions as conditions worsen."
        ),
        controller_key="ppo_shortage_minimizer",
        controller_action_keys=(
            "campaign",
            "lab_fast_track",
            "emergency_share",
            "clinical_conservation",
            "national_mutual_aid",
        ),
        controller_interval_h=6.0,
    ),
    "dreamerv3_official": Strategy(
        name="Official DreamerV3",
        description=(
            "Adaptive official DreamerV3 controller that loads a trained checkpoint "
            "and chooses interventions directly from live simulator state."
        ),
        controller_key="dreamerv3_official",
        controller_action_keys=tuple(sorted(ACTION_CATALOG.keys())),
        controller_interval_h=6.0,
    ),
    "dreamerv4": Strategy(
        name="DreamerV4",
        description=(
            "Adaptive DreamerV4-compatible controller that plugs into the simulator "
            "through the continuous Dreamer policy interface and chooses "
            "interventions directly from live simulator state."
        ),
        controller_key="dreamerv4",
        controller_action_keys=tuple(sorted(ACTION_CATALOG.keys())),
        controller_interval_h=6.0,
    ),
    "ppo_continuous": Strategy(
        name="PPO Continuous",
        description=(
            "Adaptive PPO agent trained with continuous actions in the same "
            "Box(-1, 1) action space as DreamerV3.  The agent adjusts intervention "
            "intensities, hospital routing and component allocation dials every "
            "control step."
        ),
        controller_key="ppo_continuous",
        controller_action_keys=tuple(sorted(ACTION_CATALOG.keys())),
        controller_interval_h=6.0,
    ),
    "sac_continuous": Strategy(
        name="SAC Continuous",
        description=(
            "Adaptive Soft Actor-Critic controller trained off-policy in the same "
            "continuous Dreamer-style action space. It reuses replay-buffered "
            "experience to tune intervention intensity, routing and allocation."
        ),
        controller_key="sac_continuous",
        controller_action_keys=tuple(sorted(ACTION_CATALOG.keys())),
        controller_interval_h=6.0,
    ),
    "iql_offline": Strategy(
        name="IQL Offline",
        description=(
            "Implicit Q-Learning policy trained entirely from a fixed simulator "
            "dataset, without online environment interaction during optimization."
        ),
        controller_key="iql_offline",
        controller_action_keys=tuple(sorted(ACTION_CATALOG.keys())),
        controller_interval_h=6.0,
    ),
    "cql_offline": Strategy(
        name="CQL Offline",
        description=(
            "Conservative Q-Learning policy trained from a static simulator dataset "
            "with pessimistic value regularization to stay near supported actions."
        ),
        controller_key="cql_offline",
        controller_action_keys=tuple(sorted(ACTION_CATALOG.keys())),
        controller_interval_h=6.0,
    ),
}


@dataclass
class ScenarioParams:
    name: str = "Baseline"
    description: str = "Normal operating conditions."
    narrative: str = ""
    sim_hours: int = 336
    donor_inter_arrival_h: float = QC_BASE_DONOR_INTERARRIVAL_H
    donor_show_factor: float = 1.0
    eligible_rate: float = (
        0.93  # Donor-population baseline; Héma-Québec 7-15% deferral target
    )
    demand_rate_h: float = QC_BASE_DEMAND_INTERARRIVAL_H
    demand_surge_factor: float = 1.0
    avg_units_per_order: float = QC_AVG_UNITS_PER_ORDER_EST
    transport_penalty: float = 1.0
    forced_weather: Optional[str] = None
    lab_time_factor: float = 1.0
    proc_time_factor: float = 1.0
    initial_inventory_days: float = QC_BASE_BUFFER_DAYS
    reserve_target_days: float = QC_BASE_BUFFER_DAYS
    regional_replenishment_rate: float = 0.45
    demand_forecast_noise: float = 0.12
    demand_forecast_interval_h: float = 12.0
    demand_shock_scale: float = 0.10
    transport_lead_time_noise: float = 0.15
    congestion_threshold_units: float = 10.0
    congestion_delay_factor: float = 0.06
    episode_budget: float = 7000.0
    strategy_key: str = "baseline"
    scenario_key: str = "baseline"  # <--- CRITICAL FIX: explicitly track the scenario
    is_holdout: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    # --- Sensitivity-sweep overrides; defaults leave the model unchanged. ---
    # delay_scale/ramp_scale multiply every action's activation-delay/ramp-up
    # time; reward_tier_weights scales the three composite reward tiers
    # {"safety","matching","waste"}. All are no-ops at their defaults.
    delay_scale: float = 1.0
    ramp_scale: float = 1.0
    reward_tier_weights: Optional[dict] = None

    @property
    def strategy(self) -> Strategy:
        return STRATEGIES[self.strategy_key]


NORMAL_NARRATIVE = (
    "Baseline winter operations for the Quebec City blood network, calibrated from "
    "public Hema-Quebec and Statistics Canada data rather than hand-picked magic "
    "numbers.\n\n"
    f"  • Quebec City estimated completed donations: {QC_DAILY_COMPLETED_DONATIONS_EST:.1f} "
    "per day\n"
    f"  • Quebec City estimated hospital demand: {QC_DAILY_HOSPITAL_ORDERS_EST:.1f} "
    "orders/day\n"
    f"  • Quebec City estimated blood products consumed: {QC_DAILY_LABILE_PRODUCTS_EST:.1f} "
    "units/day\n"
    f"  • Average units per order proxy: {QC_AVG_UNITS_PER_ORDER_EST:.2f}\n\n"
    "The simulator now treats the donor figure as successful completed donations, then "
    "backs into the required arrival pace. A routine provincial replenishment flow also "
    "tops up city inventory when reserves fall below target."
)


SCENARIOS = {
    "baseline": ScenarioParams(
        name="Normal Operations",
        description="Balanced Quebec City winter baseline using scaled real-world demand and donor flow.",
        narrative=NORMAL_NARRATIVE,
        sim_hours=336,
        donor_inter_arrival_h=round(QC_BASE_DONOR_INTERARRIVAL_H, 2),
        donor_show_factor=1.0,
        eligible_rate=0.93,
        demand_rate_h=round(QC_BASE_DEMAND_INTERARRIVAL_H, 2),
        demand_surge_factor=1.0,
        avg_units_per_order=round(QC_AVG_UNITS_PER_ORDER_EST, 2),
        initial_inventory_days=QC_BASE_BUFFER_DAYS,
        reserve_target_days=QC_BASE_BUFFER_DAYS,
        regional_replenishment_rate=0.45,
        demand_forecast_noise=0.10,
        demand_shock_scale=0.08,
        transport_lead_time_noise=0.10,
        congestion_threshold_units=12.0,
        congestion_delay_factor=0.04,
        episode_budget=6500.0,
        strategy_key="baseline",
        scenario_key="baseline",
    ),
    "donor_decrease": ScenarioParams(
        name="Donor Decline Crisis",
        description=(
            "A multi-week donor attrition crisis where absenteeism and fatigue reduce "
            "collection throughput while hospitals stay near baseline use."
        ),
        narrative=(
            f"{NORMAL_NARRATIVE}\n\n"
            "Stressors applied:\n"
            "  • Donor arrivals slow by 80%\n"
            "  • Show-up rate falls by 30%\n"
            "  • Eligibility drops modestly due to illness and travel\n"
            "  • Persistent snow adds friction but does not fully halt the network\n"
            "  • Provincial replenishment still operates, but at a reduced cadence"
        ),
        sim_hours=720,
        donor_inter_arrival_h=round(QC_BASE_DONOR_INTERARRIVAL_H * 1.8, 2),
        donor_show_factor=0.70,
        eligible_rate=0.88,
        demand_rate_h=round(QC_BASE_DEMAND_INTERARRIVAL_H, 2),
        demand_surge_factor=1.08,
        avg_units_per_order=round(QC_AVG_UNITS_PER_ORDER_EST, 2),
        forced_weather="snow",
        initial_inventory_days=QC_BASE_BUFFER_DAYS * 1.15,
        reserve_target_days=QC_BASE_BUFFER_DAYS,
        regional_replenishment_rate=0.35,
        demand_forecast_noise=0.14,
        demand_shock_scale=0.11,
        transport_lead_time_noise=0.18,
        congestion_threshold_units=10.0,
        congestion_delay_factor=0.06,
        episode_budget=7200.0,
        strategy_key="mass_campaign",
        scenario_key="donor_decrease",
    ),
    "demand_surge": ScenarioParams(
        name="Mass Casualty Demand Surge",
        description=(
            "A prolonged multi-week trauma surge that requires repeated hospital response, "
            "not just a single bad day."
        ),
        narrative=(
            f"{NORMAL_NARRATIVE}\n\n"
            "Stressors applied:\n"
            "  • Hospital orders arrive 45% faster than baseline\n"
            "  • Demand per order rises 35%\n"
            "  • Transport penalty increases slightly from traffic and emergency routing\n"
            "  • Provincial replenishment helps, but cannot fully offset the demand spike"
        ),
        sim_hours=720,
        donor_inter_arrival_h=round(QC_BASE_DONOR_INTERARRIVAL_H, 2),
        donor_show_factor=1.0,
        eligible_rate=0.93,
        demand_rate_h=round(QC_BASE_DEMAND_INTERARRIVAL_H * 0.55, 2),
        demand_surge_factor=1.35,
        avg_units_per_order=round(QC_AVG_UNITS_PER_ORDER_EST * 1.35, 2),
        transport_penalty=1.15,
        forced_weather="clear",
        initial_inventory_days=QC_BASE_BUFFER_DAYS * 1.2,
        reserve_target_days=QC_BASE_BUFFER_DAYS,
        regional_replenishment_rate=0.28,
        demand_forecast_noise=0.18,
        demand_shock_scale=0.16,
        transport_lead_time_noise=0.15,
        congestion_threshold_units=8.0,
        congestion_delay_factor=0.08,
        episode_budget=7600.0,
        strategy_key="emergency_network",
        scenario_key="demand_surge",
    ),
    "transport_disruption": ScenarioParams(
        name="Transport & Infrastructure Disruption",
        description=(
            "A long-running winter infrastructure disruption that slows donors, staff and "
            "inter-site movement for weeks rather than hours."
        ),
        narrative=(
            f"{NORMAL_NARRATIVE}\n\n"
            "Stressors applied:\n"
            "  • Travel times more than double\n"
            "  • Donor arrival pace slows by 90%\n"
            "  • Show-up rate drops 40%\n"
            "  • Lab processing slows because staffing is thinner\n"
            "  • Provincial replenishment is delayed by the same transport disruption"
        ),
        sim_hours=504,
        donor_inter_arrival_h=round(QC_BASE_DONOR_INTERARRIVAL_H * 1.9, 2),
        donor_show_factor=0.60,
        eligible_rate=0.88,
        demand_rate_h=round(QC_BASE_DEMAND_INTERARRIVAL_H * 0.9, 2),
        demand_surge_factor=1.15,
        avg_units_per_order=round(QC_AVG_UNITS_PER_ORDER_EST, 2),
        transport_penalty=2.2,
        forced_weather="ice_storm",
        lab_time_factor=1.15,
        initial_inventory_days=QC_BASE_BUFFER_DAYS * 1.35,
        reserve_target_days=QC_BASE_BUFFER_DAYS,
        regional_replenishment_rate=0.20,
        demand_forecast_noise=0.16,
        demand_shock_scale=0.12,
        transport_lead_time_noise=0.32,
        congestion_threshold_units=7.0,
        congestion_delay_factor=0.12,
        episode_budget=8000.0,
        strategy_key="full_response",
        scenario_key="transport_disruption",
    ),
    "combined_crisis": ScenarioParams(
        name="Combined Crisis",
        description=(
            "A multi-week compound emergency where demand spike, donor attrition and "
            "transport friction hit the network at the same time."
        ),
        narrative=(
            f"{NORMAL_NARRATIVE}\n\n"
            "Stressors applied:\n"
            "  • Donor arrivals slow by 120%\n"
            "  • Show-up rate drops 45%\n"
            "  • Hospital orders arrive 30% faster\n"
            "  • Demand per order rises 20%\n"
            "  • Blizzard conditions impose a 1.8x travel penalty\n"
            "  • The wider provincial network is also strained, so replenishment is limited"
        ),
        sim_hours=720,
        donor_inter_arrival_h=round(QC_BASE_DONOR_INTERARRIVAL_H * 2.2, 2),
        donor_show_factor=0.55,
        eligible_rate=0.84,
        demand_rate_h=round(QC_BASE_DEMAND_INTERARRIVAL_H * 0.7, 2),
        demand_surge_factor=1.20,
        avg_units_per_order=round(QC_AVG_UNITS_PER_ORDER_EST * 1.20, 2),
        transport_penalty=1.8,
        forced_weather="blizzard",
        lab_time_factor=1.20,
        initial_inventory_days=QC_BASE_BUFFER_DAYS * 1.4,
        reserve_target_days=QC_BASE_BUFFER_DAYS,
        regional_replenishment_rate=0.16,
        demand_forecast_noise=0.22,
        demand_shock_scale=0.20,
        transport_lead_time_noise=0.38,
        congestion_threshold_units=6.0,
        congestion_delay_factor=0.15,
        episode_budget=8600.0,
        strategy_key="full_response",
        scenario_key="combined_crisis",
    ),
    "holiday_flu_wave": ScenarioParams(
        name="Holiday Flu Wave",
        description=(
            "A realistic seasonal holdout where respiratory illness, holiday travel and "
            "routine snow friction depress donation while hospitals stay mildly elevated."
        ),
        narrative=(
            f"{NORMAL_NARRATIVE}\n\n"
            "Holdout benchmark stressors applied:\n"
            "  • Donor arrivals slow by 45% because regular donors travel or stay home sick\n"
            "  • Show-up rate falls 22% and eligibility drops as febrile donors are deferred\n"
            "  • Hospital orders arrive 14% faster with a modest acuity lift per order\n"
            "  • Routine snow slows transport without creating a full infrastructure outage\n"
            "  • Provincial replenishment still exists, but the wider network is also stretched"
        ),
        sim_hours=504,
        donor_inter_arrival_h=round(QC_BASE_DONOR_INTERARRIVAL_H * 1.45, 2),
        donor_show_factor=0.78,
        eligible_rate=0.84,
        demand_rate_h=round(QC_BASE_DEMAND_INTERARRIVAL_H * 0.86, 2),
        demand_surge_factor=1.10,
        avg_units_per_order=round(QC_AVG_UNITS_PER_ORDER_EST * 1.05, 2),
        transport_penalty=1.22,
        forced_weather="snow",
        lab_time_factor=1.05,
        initial_inventory_days=QC_BASE_BUFFER_DAYS * 1.20,
        reserve_target_days=QC_BASE_BUFFER_DAYS,
        regional_replenishment_rate=0.30,
        demand_forecast_noise=0.20,
        demand_shock_scale=0.16,
        transport_lead_time_noise=0.22,
        congestion_threshold_units=9.0,
        congestion_delay_factor=0.08,
        episode_budget=7800.0,
        strategy_key="full_response",
        scenario_key="holiday_flu_wave",
        is_holdout=True,
        tags=("holdout", "seasonal", "donor", "demand", "transport"),
    ),
    "regional_testing_backlog": ScenarioParams(
        name="Regional Testing Backlog",
        description=(
            "A holdout scenario where collection continues but reagent shortages and shared "
            "lab congestion slow blood release across the region."
        ),
        narrative=(
            f"{NORMAL_NARRATIVE}\n\n"
            "Holdout benchmark stressors applied:\n"
            "  • Collection remains near normal, but appointments soften slightly\n"
            "  • Regional testing capacity is constrained, so lab and processing times both rise\n"
            "  • Hospitals remain close to baseline demand, creating a realistic release bottleneck\n"
            "  • Outside replenishment is limited because the same lab backlog affects neighbouring sites"
        ),
        sim_hours=432,
        donor_inter_arrival_h=round(QC_BASE_DONOR_INTERARRIVAL_H * 1.10, 2),
        donor_show_factor=0.92,
        eligible_rate=0.92,
        demand_rate_h=round(QC_BASE_DEMAND_INTERARRIVAL_H * 0.94, 2),
        demand_surge_factor=1.08,
        avg_units_per_order=round(QC_AVG_UNITS_PER_ORDER_EST * 1.04, 2),
        transport_penalty=1.12,
        forced_weather="cloudy",
        lab_time_factor=1.65,
        proc_time_factor=1.45,
        initial_inventory_days=QC_BASE_BUFFER_DAYS * 1.45,
        reserve_target_days=QC_BASE_BUFFER_DAYS * 1.10,
        regional_replenishment_rate=0.22,
        demand_forecast_noise=0.15,
        demand_shock_scale=0.11,
        transport_lead_time_noise=0.20,
        congestion_threshold_units=8.0,
        congestion_delay_factor=0.10,
        episode_budget=7600.0,
        strategy_key="lab_investment",
        scenario_key="regional_testing_backlog",
        is_holdout=True,
        tags=("holdout", "lab", "processing", "reagents"),
    ),
    "provincial_supply_crunch": ScenarioParams(
        name="Provincial Supply Crunch",
        description=(
            "A holdout scenario where Quebec City faces only moderate local stress, but the "
            "provincial network cannot provide the usual replenishment safety net."
        ),
        narrative=(
            f"{NORMAL_NARRATIVE}\n\n"
            "Holdout benchmark stressors applied:\n"
            "  • Local donor turnout is only moderately weaker than baseline\n"
            "  • Hospitals run slightly hotter than usual for multiple weeks\n"
            "  • The key stressor is upstream scarcity: provincial replenishment nearly disappears\n"
            "  • This tests whether policies overfit to outside rescue rather than local resilience"
        ),
        sim_hours=672,
        donor_inter_arrival_h=round(QC_BASE_DONOR_INTERARRIVAL_H * 1.18, 2),
        donor_show_factor=0.88,
        eligible_rate=0.90,
        demand_rate_h=round(QC_BASE_DEMAND_INTERARRIVAL_H * 0.88, 2),
        demand_surge_factor=1.10,
        avg_units_per_order=round(QC_AVG_UNITS_PER_ORDER_EST * 1.05, 2),
        transport_penalty=1.15,
        forced_weather="cloudy",
        lab_time_factor=1.05,
        initial_inventory_days=QC_BASE_BUFFER_DAYS * 1.05,
        reserve_target_days=QC_BASE_BUFFER_DAYS * 1.10,
        regional_replenishment_rate=0.08,
        demand_forecast_noise=0.18,
        demand_shock_scale=0.14,
        transport_lead_time_noise=0.20,
        congestion_threshold_units=9.0,
        congestion_delay_factor=0.07,
        episode_budget=8000.0,
        strategy_key="emergency_network",
        scenario_key="provincial_supply_crunch",
        is_holdout=True,
        tags=("holdout", "regional", "replenishment", "supply"),
    ),
    "post_storm_backlog": ScenarioParams(
        name="Post-Storm Surgical Backlog",
        description=(
            "A holdout recovery-phase scenario: roads reopen after a storm, but postponed "
            "surgeries rebound faster than donor behaviour and courier reliability recover."
        ),
        narrative=(
            f"{NORMAL_NARRATIVE}\n\n"
            "Holdout benchmark stressors applied:\n"
            "  • Hospitals work through a backlog of postponed procedures, pushing demand above baseline\n"
            "  • Donor behaviour recovers slowly, with lingering cancellations and thinner walk-in flow\n"
            "  • Transport is no longer frozen, but lead times remain noisy and partially congested\n"
            "  • The city starts with a slightly depleted inventory because the disruption happened before the episode"
        ),
        sim_hours=504,
        donor_inter_arrival_h=round(QC_BASE_DONOR_INTERARRIVAL_H * 1.30, 2),
        donor_show_factor=0.82,
        eligible_rate=0.90,
        demand_rate_h=round(QC_BASE_DEMAND_INTERARRIVAL_H * 0.78, 2),
        demand_surge_factor=1.18,
        avg_units_per_order=round(QC_AVG_UNITS_PER_ORDER_EST * 1.10, 2),
        transport_penalty=1.35,
        forced_weather="cloudy",
        lab_time_factor=1.12,
        proc_time_factor=1.08,
        initial_inventory_days=QC_BASE_BUFFER_DAYS * 0.95,
        reserve_target_days=QC_BASE_BUFFER_DAYS * 1.05,
        regional_replenishment_rate=0.22,
        demand_forecast_noise=0.24,
        demand_shock_scale=0.18,
        transport_lead_time_noise=0.28,
        congestion_threshold_units=7.5,
        congestion_delay_factor=0.09,
        episode_budget=8200.0,
        strategy_key="full_response",
        scenario_key="post_storm_backlog",
        is_holdout=True,
        tags=("holdout", "recovery", "demand", "transport", "inventory"),
    ),
}


TRAINING_SCENARIO_KEYS = tuple(
    key for key, params in SCENARIOS.items() if not params.is_holdout
)
HOLDOUT_SCENARIO_KEYS = tuple(
    key for key, params in SCENARIOS.items() if params.is_holdout
)
SCENARIO_GROUPS = {
    "train": TRAINING_SCENARIO_KEYS,
    "holdout": HOLDOUT_SCENARIO_KEYS,
    "all": tuple(SCENARIOS.keys()),
}


REAL_WORLD_ASSUMPTIONS_TEXT = "\n".join(f"  • {line}" for line in REAL_WORLD_NOTES)

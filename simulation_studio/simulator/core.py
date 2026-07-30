"""
core.py — Blood Supply Chain Core Entities
Quebec City Blood Bank Simulation
"""

import math
import os
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional

import networkx as nx
import osmnx as ox
import simpy
from calibration import COMPONENT_COLLECTION_WEIGHTS

# ─────────────────────────────────────────────────────────
# Geography
# ─────────────────────────────────────────────────────────
CITY = "Quebec City, Quebec, Canada"
DEFAULT_CITY_BOUNDS = (46.90, 46.70, -71.10, -71.35)


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value not in {"", "0", "false", "no", "off"}


def offline_city_graph():
    north, south, east, west = DEFAULT_CITY_BOUNDS
    return None, north, south, east, west


def load_city_graph():
    if _env_flag("PIOS_SIM_OFFLINE"):
        return offline_city_graph()

    gdf = ox.geocode_to_gdf(CITY)
    north = gdf.bbox_north[0]
    south = gdf.bbox_south[0]
    east = gdf.bbox_east[0]
    west = gdf.bbox_west[0]
    G = ox.graph_from_place(CITY, network_type="drive")
    ox.add_edge_speeds(G)
    ox.add_edge_travel_times(G)
    return G, north, south, east, west


# ─────────────────────────────────────────────────────────
# Blood
# ─────────────────────────────────────────────────────────
BLOOD_DISTRIBUTION = {
    "O+": 0.44,
    "A+": 0.30,
    "B+": 0.12,
    "AB+": 0.05,
    "O-": 0.04,
    "A-": 0.03,
    "B-": 0.015,
    "AB-": 0.005,
}
BLOOD_TYPES = list(BLOOD_DISTRIBUTION.keys())

SHELF_LIFE = {
    "RBC": 42 * 24,
    "PLATELETS": 5 * 24,
    "PLASMA": 365 * 24,
}

STEP_TIMES = {
    "registration": (0.083, 0.25),
    "screening": (0.083, 0.167),
    "collection": (0.167, 0.25),
    "lab_testing": (1.0, 3.0),
    "processing": (2.0, 4.0),
    "quarantine": (12.0, 36.0),  # Modern NAT testing at Héma-Québec: 12-36 h (was 24-72 h pre-NAT)
}

QC_HOLIDAYS = {
    datetime(2024, 12, 25),
    datetime(2024, 12, 26),
    datetime(2025, 1, 1),
    datetime(2025, 2, 17),
    datetime(2025, 4, 18),
    datetime(2025, 4, 21),
    datetime(2025, 5, 19),
    datetime(2025, 6, 24),
    datetime(2025, 7, 1),
    datetime(2025, 8, 4),
    datetime(2025, 9, 1),
    datetime(2025, 10, 13),
}

MEDICATIONS_DISQUALIFYING = [
    "isotretinoin",
    "finasteride",
    "dutasteride",
    "warfarin",
    "immunosuppressants",
    "HIV_antiretrovirals",
]
MEDICATIONS_BENIGN = ["aspirin", "statins", "vitamins", "antihypertensives", "none"]

OCCUPATIONS = [
    "student",
    "healthcare",
    "office",
    "manual_labor",
    "retired",
    "unemployed",
]
OCCUPATION_DIST = [0.15, 0.12, 0.30, 0.18, 0.15, 0.10]

# ─────────────────────────────────────────────────────────
# Context helpers
# ─────────────────────────────────────────────────────────
SIM_START_DT = datetime(2024, 12, 1, 8, 0)


def sim_datetime(env_now: float) -> datetime:
    return SIM_START_DT + timedelta(hours=env_now)


def hour_of_day(env_now: float) -> int:
    return sim_datetime(env_now).hour


def hour_in_day(env_now: float) -> float:
    dt = sim_datetime(env_now)
    return dt.hour + (dt.minute / 60.0) + (dt.second / 3600.0)


def is_holiday(env_now: float) -> bool:
    dt = sim_datetime(env_now)
    return dt.replace(hour=0, minute=0, second=0, microsecond=0) in QC_HOLIDAYS


def is_weekend(env_now: float) -> bool:
    return sim_datetime(env_now).weekday() >= 5


# ─────────────────────────────────────────────────────────
# Weather
# ─────────────────────────────────────────────────────────
WEATHER_STATES = ["clear", "cloudy", "snow", "ice_storm", "blizzard"]
WEATHER_WEIGHTS = [0.30, 0.30, 0.25, 0.10, 0.05]

WEATHER_SPEED = {
    "clear": 1.00,
    "cloudy": 0.95,
    "snow": 0.70,
    "ice_storm": 0.45,
    "blizzard": 0.25,
}


class WeatherEngine:
    def __init__(self, forced_state: Optional[str] = None):
        self.state = forced_state or "clear"
        self.forced = forced_state is not None
        self.changed_at = 0.0

    def update(self, env_now: float):
        if self.forced:
            return
        if env_now - self.changed_at >= random.expovariate(1 / 6):
            self.state = random.choices(WEATHER_STATES, WEATHER_WEIGHTS)[0]
            self.changed_at = env_now

    def speed_multiplier(self, env_now: float) -> float:
        self.update(env_now)
        return WEATHER_SPEED[self.state]


# ─────────────────────────────────────────────────────────
# Context multipliers
# ─────────────────────────────────────────────────────────
def time_of_day_multiplier(env_now: float) -> float:
    h = hour_of_day(env_now)
    if 7 <= h <= 9 or 16 <= h <= 18:
        return 0.60
    if 22 <= h or h <= 5:
        return 1.40
    return 1.00


def demand_surge_multiplier(
    env_now: float, weather: WeatherEngine, surge_factor: float = 1.0
) -> float:
    h = hour_of_day(env_now)
    base = 1.0
    if 8 <= h <= 12 or 14 <= h <= 18:
        base *= 1.30
    if is_holiday(env_now):
        base *= 1.20
    if weather.state in ("ice_storm", "blizzard"):
        base *= 1.50
    return base * surge_factor


def donor_show_probability(
    env_now: float, weather: WeatherEngine, donor_factor: float = 1.0
) -> float:
    p = 0.90 * donor_factor
    if is_holiday(env_now) or is_weekend(env_now):
        p *= 0.88
    p *= weather.speed_multiplier(env_now)
    h = hour_of_day(env_now)
    if h < 8 or h > 20:
        p *= 0.50
    return min(max(p, 0.02), 1.0)


# ─────────────────────────────────────────────────────────
# Travel
# ─────────────────────────────────────────────────────────
def nearest_node(G, lat, lon):
    return ox.nearest_nodes(G, lon, lat)


def road_travel_hours(
    G,
    o_lat,
    o_lon,
    d_lat,
    d_lon,
    env_now: float,
    weather,
    transport_penalty: float = 1.0,
) -> float:
    # ── 1. Smooth distance (no graph) ─────────────────────
    dx = (o_lat - d_lat) * 111  # km approx
    dy = (o_lon - d_lon) * 85  # longitude scaling (Canada-ish)
    dist_km = math.sqrt(dx * dx + dy * dy)

    # ── 2. Smooth speed model ─────────────────────────────
    base_speed = 40.0  # km/h urban average

    # smooth congestion instead of piecewise
    h = hour_in_day(env_now)
    congestion = (
        1.0 - 0.4 * math.exp(-((h - 8) ** 2) / 6) - 0.4 * math.exp(-((h - 17) ** 2) / 6)
    )

    # weather multiplier (already smooth-ish)
    weather_factor = weather.speed_multiplier(env_now)

    speed = base_speed * congestion * weather_factor

    # avoid zero / instability
    speed = max(speed, 5.0)

    # ── 3. Travel time ────────────────────────────────────
    travel_time = (dist_km / speed) * transport_penalty

    # ── 4. Add small noise (keeps exploration alive) ──────
    noise = 0.05 * random.random()
    return max(travel_time + noise, 0.05)


# ─────────────────────────────────────────────────────────
# Blood unit
# ─────────────────────────────────────────────────────────
class BloodUnit:
    def __init__(self, env, blood_type: str, component: str, center_name: str):
        self.id = str(uuid.uuid4())
        self.blood_type = blood_type
        self.component = component
        self.collection_time = env.now
        self.expiry = env.now + SHELF_LIFE[component]
        self.center = center_name


def donation_components() -> list[str]:
    """
    Whole blood always yields RBC.
    Plasma is normally recovered, while platelets are modelled as a smaller pooled share.
    """
    components = ["RBC", "PLASMA"]
    if random.random() < COMPONENT_COLLECTION_WEIGHTS["PLATELETS"] / max(
        COMPONENT_COLLECTION_WEIGHTS["RBC"], 1e-6
    ):
        components.append("PLATELETS")
    return components


def blood_abo(blood_type: str) -> str:
    return blood_type[:-1]


def blood_rh(blood_type: str) -> str:
    return blood_type[-1]


RBC_COMPATIBILITY = {
    "O-": ["O-"],
    "O+": ["O+", "O-"],
    "A-": ["A-", "O-"],
    "A+": ["A+", "A-", "O+", "O-"],
    "B-": ["B-", "O-"],
    "B+": ["B+", "B-", "O+", "O-"],
    "AB-": ["AB-", "A-", "B-", "O-"],
    "AB+": ["AB+", "AB-", "A+", "A-", "B+", "B-", "O+", "O-"],
}

PLASMA_COMPATIBILITY_ABO = {
    "O": ["O", "A", "B", "AB"],
    "A": ["A", "AB"],
    "B": ["B", "AB"],
    "AB": ["AB"],
}


@lru_cache(maxsize=None)
def compatible_donor_types(recipient_type: str, component: str) -> list[str]:
    if component == "RBC":
        return RBC_COMPATIBILITY[recipient_type]

    if component == "PLASMA":
        abo_options = PLASMA_COMPATIBILITY_ABO[blood_abo(recipient_type)]
        ordered = []
        for abo in abo_options:
            exact = f"{abo}{blood_rh(recipient_type)}"
            alt = f"{abo}{'+' if blood_rh(recipient_type) == '-' else '-'}"
            ordered.extend([exact, alt])
        seen = set()
        return [bt for bt in ordered if not (bt in seen or seen.add(bt))]

    # Platelets: prefer ABO-identical, then RBC-compatible donors, Rh as a soft match.
    ordered = [recipient_type]
    recipient_abo = blood_abo(recipient_type)
    recipient_rh = blood_rh(recipient_type)

    same_abo = [
        bt
        for bt in BLOOD_TYPES
        if blood_abo(bt) == recipient_abo and bt != recipient_type
    ]
    ordered.extend(
        sorted(same_abo, key=lambda bt: 0 if blood_rh(bt) == recipient_rh else 1)
    )
    ordered.extend(bt for bt in RBC_COMPATIBILITY[recipient_type] if bt not in ordered)
    ordered.extend(bt for bt in BLOOD_TYPES if bt not in ordered)
    return ordered


# ─────────────────────────────────────────────────────────
# Person / Donor
# ─────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────
# Person / Donor
# ─────────────────────────────────────────────────────────
class Person:
    def __init__(
        self,
        south,
        north,
        west,
        east,
        eligible_rate: float = 1.0,
        committed: bool = False,
    ):
        """
        Donor-population calibrated model.

        Distributions reflect a *self-selected donor population*, not the
        general public.  People who walk into Héma-Québec already believe they
        are eligible, so hemoglobin, weight and health-flag base rates are
        tighter than census-level statistics.

        Calibration targets (Héma-Québec / CBS published data):
            • Overall deferral rate        :  7–15 %
            • Hemoglobin deferral          :  3– 5 %
            • Weight-based deferral        :  1– 2 %
            • Other medical/travel/tattoo  :  3– 8 % (combined)

        eligible_rate — scenario-level deferral pressure (0–1).
            Each disqualifying flag is scaled by (1 - eligible_rate) so that
            crisis scenarios can raise deferrals without changing the
            population model itself.

        committed — if True, this donor is pre-screened (called from registry):
            guarantees high motivation and clears transient deferrals.
        """
        self.id = str(uuid.uuid4())
        self.committed = committed
        self.age = int(random.triangular(17, 75, 30))
        self.sex = random.choice(["M", "F"])
        self.blood_type = random.choices(
            BLOOD_TYPES, weights=list(BLOOD_DISTRIBUTION.values())
        )[0]
        # Donor-population weight: self-selected above the 50 kg floor.
        # Male μ=82 σ=10 → P(<50 kg) ≈ 0.07 %;  Female μ=68 σ=9 → P(<50 kg) ≈ 2.3 %
        self.weight_kg = random.gauss(
            82 if self.sex == "M" else 68, 10 if self.sex == "M" else 9
        )
        self.lat = random.uniform(south, north)
        self.lon = random.uniform(west, east)
        self.occupation = random.choices(OCCUPATIONS, weights=OCCUPATION_DIST)[0]
        # Donor-population hemoglobin (narrower than census; donors self-select).
        # Male  μ=15.2 σ=0.9  → P(<13.0) ≈ 0.7 %   (CBS: 1-3 % male deferral)
        # Female μ=13.8 σ=0.8 → P(<12.5) ≈ 5.2 %   (CBS: 5-8 % female deferral)
        # Blended ≈ 3 % hemoglobin deferral — matches Héma-Québec 3-5 % benchmark.
        self.hemoglobin = random.gauss(
            15.2 if self.sex == "M" else 13.8, 0.9 if self.sex == "M" else 0.8
        )

        # Deferral pressure scalar: base rates target the 7-15 % composite
        # deferral range published by Héma-Québec and Canadian Blood Services.
        # eligible_rate=1.0 → minimal extra flags (healthy walk-in cohort).
        # eligible_rate=0.5 → higher flag rates (crisis / flu season).
        dp = 1.0 - eligible_rate  # deferral pressure  (0 = no extra risk)

        base_illness = 0.02  # 2 % baseline; rises with dp in crisis scenarios

        if committed:
            # Committed donors are pre-registered; transient conditions cleared
            self.recent_illness = random.random() < base_illness * eligible_rate
            self.recent_tattoo = random.random() < 0.008 * max(eligible_rate, 0.5)
            self.hiv_risk = random.random() < 0.001 * max(eligible_rate, 0.5)
            self.heart_disease = random.random() < 0.005 * max(1.0 - 0.5 * dp, 0.5)
            self.cancer = random.random() < 0.003 * max(1.0 - 0.5 * dp, 0.5)
            self.recent_travel = False
            self.pregnancy = False
            self.diabetes = random.random() < 0.004 * max(1.0 - 0.5 * dp, 0.5)
            self.last_donation = random.choice([None, 90, 180, 365])
            self.motivation = random.uniform(0.70, 1.00)
        else:
            # Base rates calibrated so the joint deferral ≈ 10-12 % at
            # eligible_rate=0.93 (baseline).  Each rate rises with dp
            # so crisis scenarios naturally push deferrals toward 15-20 %.
            self.recent_illness = random.random() < (0.02 + 0.06 * dp)
            self.recent_tattoo = random.random() < (0.010 + 0.015 * dp)
            self.recent_travel = random.random() < (0.006 + 0.020 * dp)
            self.pregnancy = (self.sex == "F") and (random.random() < 0.012)
            self.hiv_risk = random.random() < 0.002
            self.diabetes = random.random() < (0.005 + 0.006 * dp)
            self.heart_disease = random.random() < (0.008 + 0.008 * dp)
            self.cancer = random.random() < (0.004 + 0.004 * dp)
            # Repeat-donor interval: 56-day minimum.  Most walk-ins last
            # donated > 90 days ago; only ~1.5 % arrive within the 56-day
            # exclusion window (CBS repeat-donor scheduling data).
            self.last_donation = random.choices(
                [None, 30, 60, 90, 180, 365],
                weights=[0.50, 0.01, 0.04, 0.18, 0.17, 0.10],
            )[0]
            self.motivation = random.betavariate(2, 3)

        # Medication — disqualifying meds rare in donor population
        # (people on warfarin / immunosuppressants rarely attempt to donate)
        r = random.random()
        if r < 0.008:
            self.medication = random.choice(MEDICATIONS_DISQUALIFYING)
        elif r < 0.40:
            self.medication = random.choice(MEDICATIONS_BENIGN)
        else:
            self.medication = "none"

    def eligibility(self) -> tuple[bool, list[str]]:
        reasons = []
        if not (17 <= self.age <= 71):
            reasons.append(f"age {self.age}")
        if self.weight_kg < 50:
            reasons.append(f"weight {self.weight_kg:.0f}kg")
        min_hb = 13.0 if self.sex == "M" else 12.5
        if self.hemoglobin < min_hb:
            reasons.append(f"Hb {self.hemoglobin:.1f}")
        if self.recent_illness:
            reasons.append("recent illness")
        if self.recent_tattoo:
            reasons.append("tattoo <6m")
        if self.recent_travel:
            reasons.append("travel risk zone")
        if self.pregnancy:
            reasons.append("pregnancy/postpartum")
        if self.hiv_risk:
            reasons.append("HIV risk")
        if self.diabetes:
            reasons.append("insulin diabetes")
        if self.heart_disease:
            reasons.append("heart disease")
        if self.cancer:
            reasons.append("cancer history")
        if self.medication in MEDICATIONS_DISQUALIFYING:
            reasons.append(f"med:{self.medication}")
        if self.last_donation and self.last_donation < 56:
            reasons.append(f"donated {self.last_donation}d ago")
        return len(reasons) == 0, reasons


# ─────────────────────────────────────────────────────────
# Donation Center
# ─────────────────────────────────────────────────────────
CENTER_CONFIGS = [
    {
        "name": "Héma-Québec Lebourgneuf",
        "type": "blood_bank",
        "lat": 46.8412,
        "lon": -71.3022,
        "capacity": {"nurses": 6, "lab": 4, "processing": 3},
        "hours": (7.25, 19.75),
    },
    {
        "name": "CHU Enfant-Jésus",
        "type": "hospital",
        "lat": 46.8326,
        "lon": -71.2457,
        "capacity": {"nurses": 4, "lab": 3, "processing": 2},
        "hours": (0, 24),
    },
    {
        "name": "Héma-Québec Sainte-Foy",
        "type": "blood_bank",
        "lat": 46.7740,
        "lon": -71.2926,
        "capacity": {"nurses": 4, "lab": 2, "processing": 2},
        "hours": (7.25, 20.25),
    },
    {
        "name": "Hôtel-Dieu de Lévis",
        "type": "hospital",
        "lat": 46.8079,
        "lon": -71.1756,
        "capacity": {"nurses": 3, "lab": 2, "processing": 1},
        "hours": (0, 24),
    },
    {
        "name": "Mobile Unit – Université Laval",
        "type": "mobile",
        "lat": 46.7815,
        "lon": -71.2747,
        "capacity": {"nurses": 2, "lab": 1, "processing": 1},
        "hours": (10, 16),
    },
]


class DonationCenter:
    def __init__(self, env, config: dict):
        self.env = env
        self.name = config["name"]
        self.ctype = config["type"]
        self.original_type = config.get("original_type") or config["type"]
        self.lat = config["lat"]
        self.lon = config["lon"]
        self.open_h = config["hours"][0]
        self.close_h = config["hours"][1]
        c = config["capacity"]
        self.nurses = simpy.Resource(env, capacity=c["nurses"])
        self.lab = simpy.Resource(env, capacity=c["lab"])
        self.processing = simpy.Resource(env, capacity=c["processing"])
        self.inventory: list[BloodUnit] = []
        self._inventory_counts = {"RBC": 0, "PLATELETS": 0, "PLASMA": 0}
        self.capacity = {"nurses": self.nurses.count, "labs": self.lab.count}
        self.stats = {
            "donated": 0,
            "collected": 0,
            "rejected": 0,
            "lab_rejected": 0,
            "no_show": 0,
            "orders": 0,
            "shortage": 0,
            "transfused": 0,
            "transfers_in": 0,
            "expired": 0,
            "travel_times": [],
            "travel_time_total": 0.0,
            "travel_time_count": 0,
            "wait_times": [],
            "wait_time_total": 0.0,
            "wait_time_count": 0,
        }

    def is_open(self, env_now: float) -> bool:
        h = hour_in_day(env_now)
        if self.close_h == 24:
            return True
        return self.open_h <= h < self.close_h

    def add_unit(self, unit: BloodUnit):
        self.inventory.append(unit)
        self._inventory_counts[unit.component] = (
            self._inventory_counts.get(unit.component, 0) + 1
        )

    def remove_expired(self) -> int:
        active_inventory: list[BloodUnit] = []
        removed = 0
        for unit in self.inventory:
            if unit.expiry > self.env.now:
                active_inventory.append(unit)
                continue
            removed += 1
            self._inventory_counts[unit.component] = max(
                self._inventory_counts.get(unit.component, 0) - 1,
                0,
            )

        self.inventory = active_inventory
        self.stats["expired"] += removed
        return removed

    def compatible_units(self, blood_type: str, component: str) -> list[BloodUnit]:
        donor_types = compatible_donor_types(blood_type, component)
        donor_rank = {donor_type: idx for idx, donor_type in enumerate(donor_types)}
        compatible = [
            unit
            for unit in self.inventory
            if unit.component == component and unit.blood_type in donor_rank
        ]
        compatible.sort(
            key=lambda unit: (
                donor_rank[unit.blood_type],
                unit.expiry,
                unit.collection_time,
            )
        )
        return compatible

    def compatible_unit_count(self, blood_type: str, component: str) -> int:
        donor_types = set(compatible_donor_types(blood_type, component))
        return sum(
            1
            for unit in self.inventory
            if unit.component == component and unit.blood_type in donor_types
        )

    def component_count(self, component: str) -> int:
        return int(self._inventory_counts.get(component, 0))

    def request_unit(self, blood_type: str, component: str) -> Optional[BloodUnit]:
        self.remove_expired()
        compatible = self.compatible_units(blood_type, component)
        if not compatible:
            return None
        unit = compatible[0]
        self.inventory.remove(unit)
        self._inventory_counts[unit.component] = max(
            self._inventory_counts.get(unit.component, 0) - 1,
            0,
        )
        return unit

    def inventory_by_component(self) -> dict:
        return dict(self._inventory_counts)

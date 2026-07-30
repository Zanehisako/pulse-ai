"""
Blood Supply Chain Simulation — Quebec City
============================================
Features:
- Detailed donor profiles affecting eligibility
- Multiple donation centers (blood banks, hospitals, mobile units)
- Travel time from donor to nearest center via OSMnx road network
- Real-world contextual modifiers: weather, holidays, time-of-day, crises
- Quebec City–calibrated timing statistics for each step
"""

import math
import random
import uuid
from datetime import datetime, timedelta

import networkx as nx
import osmnx as ox
import simpy

# ─────────────────────────────────────────────
# Simulation constants
# ─────────────────────────────────────────────
RANDOM_SEED = 42
SIM_TIME = 500  # simulation time units (each unit ≈ 1 hour)
SIM_START_DT = datetime(2024, 12, 1, 8, 0)  # start at 08:00 on Dec 1

# ─────────────────────────────────────────────
# Quebec City geography
# ─────────────────────────────────────────────
CITY = "Quebec City, Quebec, Canada"
print("⏳ Loading Quebec City road network …")
gdf = ox.geocode_to_gdf(CITY)
NORTH = gdf.bbox_north[0]
SOUTH = gdf.bbox_south[0]
EAST = gdf.bbox_east[0]
WEST = gdf.bbox_west[0]
print(f"   Lat: {SOUTH:.4f}°N – {NORTH:.4f}°N")
print(f"   Lon: {WEST:.4f}°W – {EAST:.4f}°W")

G = ox.graph_from_place(CITY, network_type="drive")
ox.add_edge_speeds(G)
ox.add_edge_travel_times(G)
print("✅ Road network loaded.\n")

# ─────────────────────────────────────────────
# Blood type distribution (Héma-Québec data)
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# Shelf life (hours, per Héma-Québec standards)
# ─────────────────────────────────────────────
SHELF_LIFE = {
    "RBC": 42 * 24,  # 42 days
    "PLATELETS": 5 * 24,  #  5 days
    "PLASMA": 365 * 24,  # 365 days (frozen)
}

# ─────────────────────────────────────────────
# Quebec City–calibrated step durations (hours)
# Based on Héma-Québec / MSSS published benchmarks
# ─────────────────────────────────────────────
STEP_TIMES = {
    "registration": (0.083, 0.25),  # 5–15 min
    "screening": (0.083, 0.167),  # 5–10 min
    "collection": (0.167, 0.25),  # 10–15 min
    "lab_testing": (1.0, 3.0),  # 1–3 h  (serology + NAT)
    "processing": (2.0, 4.0),  # 2–4 h  (component separation)
    "quarantine": (24.0, 72.0),  # 24–72 h (mandatory hold)
}

# ─────────────────────────────────────────────
# Quebec City public holidays 2024–2025
# ─────────────────────────────────────────────
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
    datetime(2025, 11, 11),
}


# ─────────────────────────────────────────────
# Context helpers
# ─────────────────────────────────────────────
def sim_datetime(env_now: float) -> datetime:
    """Convert simulation hours → real datetime."""
    return SIM_START_DT + timedelta(hours=env_now)


def hour_of_day(env_now: float) -> int:
    return sim_datetime(env_now).hour


def is_holiday(env_now: float) -> bool:
    dt = sim_datetime(env_now)
    return dt.replace(hour=0, minute=0, second=0, microsecond=0) in QC_HOLIDAYS


def is_weekend(env_now: float) -> bool:
    return sim_datetime(env_now).weekday() >= 5


# ─────────────────────────────────────────────
# Weather simulation (Quebec City climate)
# ─────────────────────────────────────────────
WEATHER_STATES = ["clear", "cloudy", "snow", "ice_storm", "blizzard"]
WEATHER_WEIGHTS = [0.30, 0.30, 0.25, 0.10, 0.05]  # Dec–Mar bias

_current_weather = {"state": "clear", "changed_at": 0}


def update_weather(env_now: float):
    """Weather changes every ~6 hours on average."""
    if env_now - _current_weather["changed_at"] >= random.expovariate(1 / 6):
        _current_weather["state"] = random.choices(WEATHER_STATES, WEATHER_WEIGHTS)[0]
        _current_weather["changed_at"] = env_now


def weather_speed_multiplier(env_now: float) -> float:
    update_weather(env_now)
    return {
        "clear": 1.00,
        "cloudy": 0.95,
        "snow": 0.70,
        "ice_storm": 0.45,
        "blizzard": 0.25,
    }[_current_weather["state"]]


def time_of_day_multiplier(env_now: float) -> float:
    h = hour_of_day(env_now)
    if 7 <= h <= 9:
        return 0.60  # morning rush
    elif 16 <= h <= 18:
        return 0.60  # evening rush
    elif 22 <= h or h <= 5:
        return 1.40  # night
    else:
        return 1.00


def crisis_multiplier(env_now: float) -> float:
    """Crises: holiday + blizzard combo simulates systemic stress."""
    if is_holiday(env_now) and _current_weather["state"] == "blizzard":
        return 0.40  # severe supply disruption
    return 1.00


def demand_context_multiplier(env_now: float) -> float:
    """Hospital demand rises during rush hours and crises."""
    h = hour_of_day(env_now)
    base = 1.0
    if 8 <= h <= 12 or 14 <= h <= 18:
        base *= 1.30  # peak surgical hours
    if is_holiday(env_now):
        base *= 1.20  # trauma spikes on holidays
    if _current_weather["state"] in ("ice_storm", "blizzard"):
        base *= 1.50  # road accidents
    return base


def donor_show_up_probability(env_now: float) -> float:
    """Base probability a scheduled donor actually shows up."""
    p = 0.80
    if is_holiday(env_now) or is_weekend(env_now):
        p *= 0.70
    p *= weather_speed_multiplier(env_now)
    h = hour_of_day(env_now)
    if h < 8 or h > 20:
        p *= 0.50
    return min(max(p, 0.05), 1.0)


# ─────────────────────────────────────────────
# OSMnx travel time
# ─────────────────────────────────────────────
def nearest_node(lat, lon):
    return ox.nearest_nodes(G, lon, lat)


def road_travel_hours(origin_lat, origin_lon, dest_lat, dest_lon, env_now=0) -> float:
    """Shortest-path travel time in hours, adjusted for context."""
    try:
        o_node = nearest_node(origin_lat, origin_lon)
        d_node = nearest_node(dest_lat, dest_lon)
        if o_node == d_node:
            return 0.0
        route = ox.shortest_path(G, o_node, d_node, weight="travel_time")
        if route is None:
            return random.uniform(0.25, 1.0)
        # sum edge travel times (seconds) → hours
        base_seconds = sum(
            G[u][v][0].get("travel_time", 60) for u, v in zip(route[:-1], route[1:])
        )
        base_hours = base_seconds / 3600
        # apply context multipliers
        adj = base_hours / (
            weather_speed_multiplier(env_now) * time_of_day_multiplier(env_now)
        )
        return max(adj, 0.05)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return random.uniform(0.25, 1.0)


# ─────────────────────────────────────────────
# Donation Centers
# ─────────────────────────────────────────────
CENTER_CONFIGS = [
    {
        "name": "Héma-Québec Centre-Ville",
        "type": "blood_bank",
        "lat": 46.8139,
        "lon": -71.2080,
        "capacity": {"nurses": 6, "lab": 4, "processing": 3},
        "hours": (8, 20),
    },
    {
        "name": "CHU de Québec – Hôpital de l'Enfant-Jésus",
        "type": "hospital",
        "lat": 46.8326,
        "lon": -71.2457,
        "capacity": {"nurses": 4, "lab": 3, "processing": 2},
        "hours": (0, 24),  # 24 h
    },
    {
        "name": "Héma-Québec Sainte-Foy",
        "type": "blood_bank",
        "lat": 46.7740,
        "lon": -71.2926,
        "capacity": {"nurses": 4, "lab": 2, "processing": 2},
        "hours": (9, 18),
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
        self.lat = config["lat"]
        self.lon = config["lon"]
        self.open_hour = config["hours"][0]
        self.close_hour = config["hours"][1]
        c = config["capacity"]
        self.nurses = simpy.Resource(env, capacity=c["nurses"])
        self.lab = simpy.Resource(env, capacity=c["lab"])
        self.processing = simpy.Resource(env, capacity=c["processing"])
        self.inventory = []
        self.stats = {"donated": 0, "rejected": 0, "shortage": 0, "transfused": 0}

    def is_open(self, env_now: float) -> bool:
        h = hour_of_day(env_now)
        if self.open_hour < self.close_hour:
            return self.open_hour <= h < self.close_hour
        return True  # 24h

    def add_unit(self, unit):
        self.inventory.append(unit)

    def remove_expired(self):
        before = len(self.inventory)
        self.inventory = [u for u in self.inventory if u.expiry > self.env.now]
        return before - len(self.inventory)

    def request_unit(self, blood_type, component):
        self.remove_expired()
        for u in self.inventory:
            if u.blood_type == blood_type and u.component == component:
                self.inventory.remove(u)
                return u
        return None

    def summary(self):
        return (
            f"  [{self.name}] donated={self.stats['donated']} "
            f"rejected={self.stats['rejected']} "
            f"transfused={self.stats['transfused']} "
            f"shortage={self.stats['shortage']} "
            f"inventory={len(self.inventory)}"
        )


# ─────────────────────────────────────────────
# Blood unit
# ─────────────────────────────────────────────
class BloodUnit:
    def __init__(self, env, blood_type, component, center_name):
        self.id = str(uuid.uuid4())
        self.blood_type = blood_type
        self.component = component
        self.collection_time = env.now
        self.expiry = env.now + SHELF_LIFE[component]
        self.center = center_name


# ─────────────────────────────────────────────
# Detailed Person / Donor profile
# ─────────────────────────────────────────────
OCCUPATIONS = [
    "student",
    "healthcare",
    "office",
    "manual_labor",
    "retired",
    "unemployed",
]
OCCUPATION_DIST = [0.15, 0.12, 0.30, 0.18, 0.15, 0.10]

MEDICATIONS_DISQUALIFYING = [
    "isotretinoin",
    "finasteride",
    "dutasteride",
    "warfarin",
    "immunosuppressants",
    "HIV_antiretrovirals",
]
MEDICATIONS_BENIGN = ["aspirin", "statins", "vitamins", "antihypertensives", "none"]


class Person:
    """Realistic Quebec City resident profile."""

    def __init__(self):
        self.id = str(uuid.uuid4())
        self.age = int(random.triangular(17, 75, 30))
        self.sex = random.choice(["M", "F"])
        self.blood_type = random.choices(
            BLOOD_TYPES, weights=list(BLOOD_DISTRIBUTION.values())
        )[0]
        self.weight_kg = random.gauss(78 if self.sex == "M" else 65, 12)
        self.lat = random.uniform(SOUTH, NORTH)
        self.lon = random.uniform(WEST, EAST)
        self.occupation = random.choices(OCCUPATIONS, weights=OCCUPATION_DIST)[0]

        # Health flags
        self.hemoglobin_g_dl = random.gauss(14.5 if self.sex == "M" else 13.0, 1.5)
        self.recent_illness = random.random() < 0.08
        self.recent_tattoo = random.random() < 0.10  # <6 months → deferral
        self.recent_travel = random.random() < 0.05  # malaria-risk zone
        self.pregnancy_or_12m = (self.sex == "F") and (random.random() < 0.07)
        self.hiv_risk_behavior = random.random() < 0.02
        self.diabetes_insulin = random.random() < 0.04
        self.heart_disease = random.random() < 0.06
        self.cancer_history = random.random() < 0.03
        self.last_donation_days = random.choice([None, 30, 60, 90, 180, 365])

        # Medication
        p_disqual = 0.05
        p_benign = 0.35
        r = random.random()
        if r < p_disqual:
            self.medication = random.choice(MEDICATIONS_DISQUALIFYING)
        elif r < p_disqual + p_benign:
            self.medication = random.choice(MEDICATIONS_BENIGN)
        else:
            self.medication = "none"

        # Motivation to donate (affects show-up)
        self.motivation_score = random.betavariate(2, 3)  # 0–1, skewed low

    def eligibility_assessment(self) -> tuple[bool, list[str]]:
        """
        Returns (eligible: bool, deferral_reasons: list[str])
        Based on Héma-Québec/Canadian Blood Services criteria.
        """
        reasons = []

        if not (17 <= self.age <= 71):
            reasons.append(f"age {self.age} out of 17–71 range")
        if self.weight_kg < 50:
            reasons.append(f"weight {self.weight_kg:.1f} kg < 50 kg minimum")
        min_hb = 13.0 if self.sex == "M" else 12.5
        if self.hemoglobin_g_dl < min_hb:
            reasons.append(f"hemoglobin {self.hemoglobin_g_dl:.1f} g/dL below {min_hb}")
        if self.recent_illness:
            reasons.append("recent illness (< 2 weeks)")
        if self.recent_tattoo:
            reasons.append("tattoo/piercing < 6 months")
        if self.recent_travel:
            reasons.append("travel to malaria-risk zone")
        if self.pregnancy_or_12m:
            reasons.append("pregnancy or postpartum < 12 months")
        if self.hiv_risk_behavior:
            reasons.append("HIV risk behavior — permanent deferral")
        if self.diabetes_insulin:
            reasons.append("insulin-dependent diabetes")
        if self.heart_disease:
            reasons.append("active heart disease")
        if self.cancer_history:
            reasons.append("cancer history")
        if self.medication in MEDICATIONS_DISQUALIFYING:
            reasons.append(f"disqualifying medication: {self.medication}")
        if self.last_donation_days is not None and self.last_donation_days < 56:
            reasons.append(
                f"last donation only {self.last_donation_days} days ago (min 56)"
            )

        eligible = len(reasons) == 0
        return eligible, reasons

    def profile_summary(self) -> str:
        eligible, reasons = self.eligibility_assessment()
        status = "✅ ELIGIBLE" if eligible else f"❌ DEFERRED ({'; '.join(reasons)})"
        return (
            f"Person({self.id[:8]}) "
            f"age={self.age} sex={self.sex} BT={self.blood_type} "
            f"Hb={self.hemoglobin_g_dl:.1f} wt={self.weight_kg:.0f}kg "
            f"occ={self.occupation} med={self.medication} → {status}"
        )


# ─────────────────────────────────────────────
# Simulation processes
# ─────────────────────────────────────────────


def nearest_open_center(centers: list[DonationCenter], lat, lon, env_now: float):
    """Return closest center that is currently open."""
    open_c = [c for c in centers if c.is_open(env_now)]
    if not open_c:
        open_c = centers  # fall back to any center

    def dist(c):
        return math.hypot(c.lat - lat, c.lon - lon)

    return min(open_c, key=dist)


def donor_process(env, person: Person, centers: list[DonationCenter]):
    """Full donor journey: travel → registration → screening → collection → lab → processing."""

    # Contextual show-up check
    base_show_up = donor_show_up_probability(env.now) * person.motivation_score
    if random.random() > base_show_up:
        return  # no-show

    center = nearest_open_center(centers, person.lat, person.lon, env.now)

    # ── Travel ──────────────────────────────
    travel_h = road_travel_hours(
        person.lat, person.lon, center.lat, center.lon, env.now
    )
    w = _current_weather["state"]
    dt = sim_datetime(env.now)
    print(
        f"[{env.now:6.1f}h | {dt:%d-%b %H:%M}] 🚗 Donor {person.id[:8]} "
        f"→ {center.name} | travel={travel_h * 60:.0f}min "
        f"| weather={w} | {person.blood_type}"
    )
    yield env.timeout(travel_h)

    # ── Eligibility screening ────────────────
    eligible, reasons = person.eligibility_assessment()
    reg_time = random.uniform(*STEP_TIMES["registration"])
    scrn_time = random.uniform(*STEP_TIMES["screening"])
    yield env.timeout(reg_time + scrn_time)

    if not eligible:
        print(
            f"[{env.now:6.1f}h] ❌ DEFERRED at {center.name}: {'; '.join(reasons[:2])}"
        )
        center.stats["rejected"] += 1
        return

    # ── Collection ───────────────────────────
    with center.nurses.request() as req:
        yield req
        coll_time = random.uniform(*STEP_TIMES["collection"])
        yield env.timeout(coll_time)

    # ── Lab testing ──────────────────────────
    with center.lab.request() as req:
        yield req
        lab_time = random.uniform(*STEP_TIMES["lab_testing"])
        # Lab is slower on weekends/holidays
        if is_holiday(env.now) or is_weekend(env.now):
            lab_time *= 1.40
        yield env.timeout(lab_time)

    # Infectious disease rejection (Héma-Québec: ~0.5% after screening)
    if random.random() < 0.005:
        print(f"[{env.now:6.1f}h] 🧪 Lab REJECTION at {center.name}")
        center.stats["rejected"] += 1
        return

    # ── Processing ───────────────────────────
    with center.processing.request() as req:
        yield req
        proc_time = random.uniform(*STEP_TIMES["processing"])
        yield env.timeout(proc_time)

    # ── Mandatory quarantine ─────────────────
    quar_time = random.uniform(*STEP_TIMES["quarantine"])
    yield env.timeout(quar_time)

    # ── Add components to inventory ──────────
    for comp in ["RBC", "PLASMA", "PLATELETS"]:
        unit = BloodUnit(env, person.blood_type, comp, center.name)
        center.add_unit(unit)

    center.stats["donated"] += 1
    print(
        f"[{env.now:6.1f}h] ✅ DONATION complete → {center.name} "
        f"({person.blood_type}, {person.id[:8]})"
    )


# def committed_donor_call(state: SimState, n_calls: int = 5):
#     """Every 12h, call committed donors directly — higher show-up rate."""
#     while True:
#         yield state.env.timeout(12)
#         for _ in range(n_calls):
#             person = Person(state.south, state.north, state.west, state.east)
#             # Committed donors: pre-screened, high motivation, guaranteed eligible
#             person.motivation = random.uniform(0.7, 1.0)
#             person.recent_illness = False
#             person.recent_tattoo = False
#             person.hiv_risk = False
#             person.heart_disease = False
#             person.last_donation = random.choice([None, 90, 180])
#             state.env.process(donor_process(state, person))


def donor_generator(
    env, centers: list[DonationCenter], eligible_rate=0.60, inter_arrival_h=2.0
):
    """Generate donor arrivals (Poisson process, ~Héma-Québec volume)."""
    while True:
        # Inter-arrival: exponential, modulated by context
        ctx = demand_context_multiplier(env.now)
        lam = 1 / (inter_arrival_h / max(ctx * 0.5, 0.1))
        yield env.timeout(random.expovariate(lam))

        # Only generate if there's a reasonable chance of donors being out
        h = hour_of_day(env.now)
        if h < 7 or h > 21:
            continue  # very few donors at night

        person = Person()
        # env.process(committed_donor_call(state, n_calls=8))
        env.process(donor_process(env, person, centers))


def hospital_demand(env, centers: list[DonationCenter], base_rate_h=3.0):
    """Hospitals request blood components; shortages are logged."""
    while True:
        ctx = demand_context_multiplier(env.now)
        rate = base_rate_h / ctx  # more demand → shorter interval
        yield env.timeout(random.expovariate(1 / rate))

        blood_type = random.choices(
            BLOOD_TYPES, weights=list(BLOOD_DISTRIBUTION.values())
        )[0]
        component = random.choices(
            ["RBC", "PLATELETS", "PLASMA"], weights=[0.60, 0.25, 0.15]
        )[0]

        # Try centers from largest inventory to smallest
        centers_sorted = sorted(centers, key=lambda c: len(c.inventory), reverse=True)
        filled = False
        for center in centers_sorted:
            unit = center.request_unit(blood_type, component)
            if unit:
                center.stats["transfused"] += 1
                filled = True
                dt = sim_datetime(env.now)
                print(
                    f"[{env.now:6.1f}h | {dt:%d-%b %H:%M}] 💉 Transfused "
                    f"{component} {blood_type} from {center.name}"
                )
                break

        if not filled:
            # Try any compatible type (simplified compatibility)
            for center in centers_sorted:
                # O- is universal donor for RBC
                if component == "RBC":
                    unit = center.request_unit("O-", "RBC")
                    if unit:
                        center.stats["transfused"] += 1
                        filled = True
                        print(
                            f"[{env.now:6.1f}h] 💉 Transfused O- RBC (universal) from {center.name}"
                        )
                        break
            if not filled:
                random.choice(centers).stats["shortage"] += 1
                print(
                    f"[{env.now:6.1f}h] ⚠️  SHORTAGE: {component} {blood_type} — "
                    f"weather={_current_weather['state']}"
                )


def monitor(env, centers: list[DonationCenter]):
    """Periodic report every 24 simulation hours."""
    while True:
        yield env.timeout(24)
        dt = sim_datetime(env.now)
        expired_total = sum(c.remove_expired() for c in centers)
        total_inv = sum(len(c.inventory) for c in centers)
        print(f"\n{'─' * 70}")
        print(f"📊 REPORT  [{env.now:.0f}h | {dt:%d-%b-%Y %H:%M}]")
        print(
            f"   Weather: {_current_weather['state']}  |  Holiday: {is_holiday(env.now)}"
        )
        print(f"   Total inventory: {total_inv}  |  Expired removed: {expired_total}")
        for c in centers:
            print(c.summary())
        print(f"{'─' * 70}\n")


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
random.seed(RANDOM_SEED)
env = simpy.Environment()

centers = [DonationCenter(env, cfg) for cfg in CENTER_CONFIGS]

env.process(donor_generator(env, centers, eligible_rate=0.60, inter_arrival_h=1.5))
env.process(hospital_demand(env, centers, base_rate_h=2.5))
env.process(monitor(env, centers))

print("🔴 Starting simulation …\n")
env.run(until=SIM_TIME)

print("\n" + "═" * 70)
print("FINAL SUMMARY")
print("═" * 70)
for c in centers:
    print(c.summary())

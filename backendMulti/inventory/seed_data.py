"""
Deterministic Quebec-specific seed catalogs and helpers for inventory demo data.

The seed commands use these shared constants to keep hospitals, donors, blood
supply, and prediction seasonality aligned around real Quebec regions while
remaining stable across repeated runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_FLOOR
import random

SEED = 42

BLOOD_TYPES = ("O+", "A+", "B+", "O-", "A-", "B-", "AB+", "AB-")

BLOOD_TYPE_WEIGHTS = {
    "O+": 39,
    "A+": 36,
    "B+": 8,
    "O-": 7,
    "A-": 6,
    "B-": 2,
    "AB+": 2,
    "AB-": 1,
}

BLOOD_TYPE_PROFILES = {
    "O+": {"stock_base": 128.0, "usage_base": 17.5, "scarcity": 0.45},
    "A+": {"stock_base": 114.0, "usage_base": 14.5, "scarcity": 0.5},
    "B+": {"stock_base": 91.0, "usage_base": 7.4, "scarcity": 0.7},
    "O-": {"stock_base": 74.0, "usage_base": 5.6, "scarcity": 0.95},
    "A-": {"stock_base": 67.0, "usage_base": 4.6, "scarcity": 1.0},
    "B-": {"stock_base": 56.0, "usage_base": 3.4, "scarcity": 1.2},
    "AB+": {"stock_base": 48.0, "usage_base": 2.8, "scarcity": 1.1},
    "AB-": {"stock_base": 30.0, "usage_base": 1.9, "scarcity": 1.35},
}

HOSPITAL_SIZE_PROFILES = {
    "large": {
        "stock_scale": 1.75,
        "usage_scale": 1.7,
        "volatility": 0.08,
        "lead_time_days": 2,
        "surgery_base": 24,
        "trauma_base": 8,
        "shortage_penalty": 0.3,
    },
    "urban": {
        "stock_scale": 1.3,
        "usage_scale": 1.2,
        "volatility": 0.12,
        "lead_time_days": 3,
        "surgery_base": 17,
        "trauma_base": 5,
        "shortage_penalty": 0.9,
    },
    "regional": {
        "stock_scale": 0.86,
        "usage_scale": 0.74,
        "volatility": 0.2,
        "lead_time_days": 4,
        "surgery_base": 9,
        "trauma_base": 3,
        "shortage_penalty": 1.8,
    },
    "remote": {
        "stock_scale": 0.72,
        "usage_scale": 0.62,
        "volatility": 0.26,
        "lead_time_days": 5,
        "surgery_base": 7,
        "trauma_base": 3,
        "shortage_penalty": 2.4,
    },
}


@dataclass(frozen=True)
class HospitalSeed:
    hospital_id: str
    name: str
    wilaya: str
    city: str
    latitude: float
    longitude: float
    size_tier: str


@dataclass(frozen=True)
class CitySeed:
    city_id: str
    city: str
    wilaya: str
    latitude: float
    longitude: float
    cluster_id: int
    population_weight: int
    latitude_jitter: float
    longitude_jitter: float


QUEBEC_HOSPITALS: tuple[HospitalSeed, ...] = (
    # Québec region
    HospitalSeed("H001", "CHU de Québec - Hôpital de l'Enfant-Jésus", "Québec", "Québec", 46.8382, -71.2224, "large"),
    HospitalSeed("H002", "CHU de Québec - Hôtel-Dieu de Québec", "Québec", "Québec", 46.8146, -71.2141, "large"),
    # Montréal region
    HospitalSeed("H003", "Centre hospitalier de l'Université de Montréal (CHUM)", "Montréal", "Montréal", 45.5075, -73.5540, "large"),
    HospitalSeed("H004", "McGill University Health Centre (MUHC)", "Montréal", "Montréal", 45.4737, -73.5998, "large"),
    HospitalSeed("H005", "Hôpital du Sacré-Cœur de Montréal", "Montréal", "Montréal", 45.5358, -73.7166, "urban"),
    HospitalSeed("H006", "Hôpital Maisonneuve-Rosemont", "Montréal", "Montréal", 45.5767, -73.5475, "urban"),
    HospitalSeed("H007", "CHU Sainte-Justine", "Montréal", "Montréal", 45.4973, -73.6195, "urban"),
)

HOSPITALS_BY_ID = {row.hospital_id: row for row in QUEBEC_HOSPITALS}
WILAYAS = tuple(dict.fromkeys(row.wilaya for row in QUEBEC_HOSPITALS))

QUEBEC_CITIES: tuple[CitySeed, ...] = (
    CitySeed("C000", "Québec", "Québec", 46.8139, -71.2080, 0, 850_000, 0.08, 0.11),
    CitySeed("C001", "Montréal", "Montréal", 45.5017, -73.5673, 1, 2_050_000, 0.07, 0.09),
)

CITIES_BY_ID = {row.city_id: row for row in QUEBEC_CITIES}
CITIES_BY_WILAYA = {
    wilaya: tuple(city for city in QUEBEC_CITIES if city.wilaya == wilaya)
    for wilaya in WILAYAS
}

FRENCH_QUEBEC_GIVEN_NAMES = (
    "Alexandre",
    "Amélie",
    "Antoine",
    "Audrey",
    "Camille",
    "Catherine",
    "Charles",
    "Charlotte",
    "David",
    "Émilie",
    "Félix",
    "Gabriel",
    "Isabelle",
    "Jean-François",
    "Julie",
    "Karine",
    "Laurence",
    "Louis",
    "Marc-André",
    "Marie-Ève",
    "Mathieu",
    "Maxime",
    "Nathalie",
    "Nicolas",
    "Olivier",
    "Pascal",
    "Sophie",
    "Stéphanie",
    "Vincent",
    "Xavier",
)

FRENCH_QUEBEC_SURNAMES = (
    "Bouchard",
    "Cote",
    "Fortin",
    "Gagne",
    "Gagnon",
    "Lefebvre",
    "Lemieux",
    "Leroux",
    "Lavoie",
    "Martin",
    "Morin",
    "Ouellet",
    "Pelletier",
    "Roy",
    "Savard",
    "Simard",
    "Tremblay",
    "Beaulieu",
    "Bergeron",
    "Caron",
)


def seeded_random(*parts: object) -> random.Random:
    return random.Random("|".join(str(part) for part in (SEED, *parts)))


def allocate_counts(total: int, weighted_rows: list[tuple[str, int]]) -> dict[str, int]:
    if total <= 0:
        return {key: 0 for key, _weight in weighted_rows}

    total_weight = sum(weight for _key, weight in weighted_rows)
    if total_weight <= 0:
        raise ValueError("Total weight must be positive.")

    allocations: dict[str, int] = {}
    remainders: list[tuple[Decimal, str]] = []
    assigned = 0

    for key, weight in weighted_rows:
        raw = Decimal(total * weight) / Decimal(total_weight)
        whole = int(raw.to_integral_value(rounding=ROUND_FLOOR))
        allocations[key] = whole
        assigned += whole
        remainders.append((raw - whole, key))

    for _remainder, key in sorted(remainders, reverse=True):
        if assigned >= total:
            break
        allocations[key] += 1
        assigned += 1

    return allocations


def quebec_seasonality_modifier(day: date) -> float:
    modifier = 1.0

    if day.month in (1, 2):
        modifier *= 0.90

    if day.month in (6, 7, 8):
        modifier *= 0.85

    if day.month in (2, 3, 9, 10, 11):
        modifier *= 1.10

    if (day.month == 12 and 24 <= day.day <= 31) or (day.month == 6 and day.day == 24):
        modifier *= 0.80

    return modifier

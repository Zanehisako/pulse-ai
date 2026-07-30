"""
Generate Quebec donor forecasting features and labels.

Outputs:
    quebec_donor_features.parquet
    quebec_donor_labels.parquet
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_repo_root = Path(__file__).resolve().parents[2]
_backend_path = _repo_root.parent / "backendMulti"
if str(_backend_path) not in sys.path:
    sys.path.insert(0, str(_backend_path))

from inventory.seed_data import (  # noqa: E402
    BLOOD_TYPES,
    BLOOD_TYPE_PROFILES,
    BLOOD_TYPE_WEIGHTS,
    QUEBEC_CITIES,
    QUEBEC_HOSPITALS,
)


RNG = np.random.default_rng(seed=20260508)
START_DATE = datetime(2024, 1, 1, tzinfo=timezone.utc)
SNAPSHOT_STEP_DAYS = 45
N_DONORS_DEFAULT = 12_000
N_DONORS_FULL = 60_000
N_SNAPSHOTS_DEFAULT = 6
N_SNAPSHOTS_FULL = 10

LANGUAGE_PROFILES = ("fr", "fr", "fr", "en", "fr_en", "other")


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    lat1_r, lon1_r, lat2_r, lon2_r = np.radians([lat1, lon1, lat2, lon2])
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return float(2.0 * radius * np.arcsin(np.sqrt(a)))


def _nearest_hospital(latitude: float, longitude: float):
    distances = [
        (
            _haversine_km(latitude, longitude, hospital.latitude, hospital.longitude),
            hospital,
        )
        for hospital in QUEBEC_HOSPITALS
    ]
    return min(distances, key=lambda row: row[0])


def _sample_blood_types(size: int) -> np.ndarray:
    weights = np.array([BLOOD_TYPE_WEIGHTS[blood_type] for blood_type in BLOOD_TYPES], dtype=float)
    weights = weights / weights.sum()
    return RNG.choice(np.array(BLOOD_TYPES), size=size, p=weights)


def _logistic(value: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-value))


def _donor_static_rows(n_donors: int) -> pd.DataFrame:
    city_weights = np.array([city.population_weight for city in QUEBEC_CITIES], dtype=float)
    city_weights = city_weights / city_weights.sum()
    city_indices = RNG.choice(np.arange(len(QUEBEC_CITIES)), size=n_donors, p=city_weights)
    blood_types = _sample_blood_types(n_donors)

    rows = []
    for index, city_index in enumerate(city_indices, start=1):
        city = QUEBEC_CITIES[int(city_index)]
        latitude = float(RNG.normal(city.latitude, city.latitude_jitter / 3.0))
        longitude = float(RNG.normal(city.longitude, city.longitude_jitter / 3.0))
        distance_km, hospital = _nearest_hospital(latitude, longitude)
        age = int(np.clip(RNG.normal(42, 14), 18, 70))
        blood_type = str(blood_types[index - 1])
        chronic_condition = int(RNG.random() < (0.04 + max(age - 50, 0) * 0.004))
        rows.append(
            {
                "donor_id": f"QD{index:07d}",
                "age": age,
                "sex": str(RNG.choice(["F", "M", "X"], p=[0.51, 0.48, 0.01])),
                "country_code": "CA",
                "province": "Quebec",
                "region": city.wilaya,
                "wilaya": city.wilaya,
                "city_id": city.city_id,
                "city": city.city,
                "latitude": round(latitude, 5),
                "longitude": round(longitude, 5),
                "urbanicity": "metro" if city.population_weight > 1_000_000 else "urban",
                "language_preference": str(RNG.choice(LANGUAGE_PROFILES, p=[0.55, 0.18, 0.08, 0.11, 0.06, 0.02])),
                "blood_type": blood_type,
                "is_rare_type": int(blood_type in {"O-", "A-", "B-", "AB-", "AB+"}),
                "blood_type_country_prevalence": BLOOD_TYPE_WEIGHTS[blood_type] / 100.0,
                "blood_type_scarcity": BLOOD_TYPE_PROFILES[blood_type]["scarcity"],
                "bmi": round(float(np.clip(RNG.normal(25.4, 4.1), 18.0, 39.5)), 1),
                "smoker": int(RNG.random() < 0.13),
                "chronic_condition_flag": chronic_condition,
                "nearest_hospital_id": hospital.hospital_id,
                "nearest_hospital_name": hospital.name,
                "nearest_hospital_region": hospital.wilaya,
                "nearest_collection_center_id": f"HQ-{hospital.city[:3].upper()}-{hospital.hospital_id}",
                "center_distance_km": round(distance_km, 2),
                "travel_time_min": round(distance_km / float(RNG.uniform(0.45, 0.78)) + RNG.uniform(4, 18), 1),
                "route_feasibility_flag": int(distance_km <= 45.0),
                "public_transit_access_score": round(float(np.clip(RNG.normal(0.78 if city.population_weight > 1_000_000 else 0.62, 0.16), 0, 1)), 3),
                "winter_route_risk_score": round(float(np.clip(distance_km / 75.0 + RNG.normal(0.08, 0.06), 0, 1)), 3),
                "cold_chain_quality_score": round(float(np.clip(RNG.normal(0.91, 0.05), 0.65, 1.0)), 3),
                "transport_capacity_slots": int(RNG.integers(6, 28)),
                "transfer_cost_usd": round(float(18.0 + distance_km * RNG.uniform(0.55, 1.25)), 2),
                "area_deprivation_index": round(float(np.clip(RNG.normal(0.38, 0.18), 0, 1)), 3),
                "digital_contactability_score": round(float(np.clip(RNG.beta(4, 2), 0, 1)), 3),
            }
        )
    return pd.DataFrame(rows)


def _build_snapshots(static: pd.DataFrame, n_snapshots: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    first_donation_year = RNG.integers(2008, 2024, size=len(static))
    lifetime_counts = RNG.poisson(8, size=len(static)) + RNG.binomial(18, 0.22, size=len(static))
    regular_flags = RNG.binomial(1, 0.42, size=len(static))

    for donor_index, donor in static.reset_index(drop=True).iterrows():
        donor_history_strength = float(
            np.clip(
                0.7 * regular_flags[donor_index]
                + 0.08 * lifetime_counts[donor_index]
                + RNG.normal(0, 0.35),
                0,
                4,
            )
        )
        typical_interval = float(np.clip(RNG.normal(115 - donor_history_strength * 12, 28), 56, 240))
        base_last_gap = float(np.clip(RNG.normal(typical_interval, 38), 20, 420))

        for snapshot_index in range(n_snapshots):
            event_timestamp = START_DATE + timedelta(days=SNAPSHOT_STEP_DAYS * snapshot_index)
            seasonal = 0.18 * np.sin(2.0 * np.pi * event_timestamp.timetuple().tm_yday / 365.25)
            recency_days = max(0.0, base_last_gap + SNAPSHOT_STEP_DAYS * snapshot_index - RNG.normal(0, 18))
            donation_count_last_12m = int(
                np.clip(
                    RNG.poisson(max(0.2, 3.4 - typical_interval / 70.0 + donor_history_strength * 0.35)),
                    0,
                    6,
                )
            )
            days_until_eligible = max(0.0, 56.0 - recency_days)
            eligible = int(
                days_until_eligible <= 0
                and donor["chronic_condition_flag"] == 0
                and 18 <= int(donor["age"]) <= 70
                and 18.5 <= float(donor["bmi"]) <= 38.0
            )
            route_feasible = int(donor["route_feasibility_flag"])
            readiness_score = (
                1.5 * eligible
                + 0.45 * donation_count_last_12m
                + 0.55 * int(regular_flags[donor_index])
                + 0.30 * donor_history_strength
                + seasonal
                - 0.30 * int(donor["smoker"])
            )
            mobility_score = (
                1.15 * route_feasible
                + 0.75 * (float(donor["travel_time_min"]) <= 30)
                + 0.45 * float(donor["public_transit_access_score"])
                + 0.20 * float(donor["cold_chain_quality_score"])
                - 0.45 * float(donor["winter_route_risk_score"])
            )
            rare_priority = (
                1.0
                + 1.5 * int(donor["is_rare_type"])
                + float(donor["blood_type_scarcity"])
                - float(donor["blood_type_country_prevalence"])
            )
            travel_burden = float(donor["travel_time_min"]) * (1.0 + float(donor["center_distance_km"]) / 25.0)
            commitment_score = donation_count_last_12m + 2.5 * int(regular_flags[donor_index]) + donor_history_strength
            response_readiness = (
                0.42 * readiness_score
                + 0.22 * mobility_score
                + 0.22 * commitment_score
                + 0.14 * rare_priority
                - 0.012 * travel_burden
            )
            probability_90d = float(_logistic(-2.55 + 0.88 * response_readiness + 0.72 * eligible))
            probability_180d = float(_logistic(-1.65 + 0.82 * response_readiness + 0.62 * eligible))
            timing_pressure = (
                0.92 * response_readiness
                + 0.48 * eligible
                + 0.18 * float(donor["digital_contactability_score"])
                + 0.12 * seasonal
                - 0.010 * travel_burden
                - 0.34 * int(donor["chronic_condition_flag"])
                - 0.18 * int(donor["smoker"])
            )
            expected_days = (
                295.0
                - 44.0 * timing_pressure
                + 0.58 * days_until_eligible
                + 0.30 * float(donor["travel_time_min"])
                - 8.0 * donation_count_last_12m
                + RNG.normal(0, 10)
            )
            days_until_next = float(np.clip(expected_days, 1.0, 420.0))
            if RNG.random() < 0.04 * probability_90d:
                days_until_next = float(np.clip(days_until_next - RNG.uniform(18, 42), 1.0, 420.0))
            elif RNG.random() < 0.03 * (1.0 - probability_180d):
                days_until_next = float(np.clip(days_until_next + RNG.uniform(25, 70), 1.0, 420.0))

            ideal_probability = float(
                np.clip(
                    _logistic(
                        -2.1
                        + 0.95 * readiness_score
                        + 0.48 * mobility_score
                        + 0.26 * rare_priority
                        + 0.18 * commitment_score
                        - 0.014 * travel_burden
                        - 0.55 * int(donor["smoker"])
                    ),
                    0.0,
                    1.0,
                )
            )
            feature_row = donor.to_dict()
            feature_row.update(
                {
                    "event_timestamp": event_timestamp,
                    "snapshot_month": event_timestamp.month,
                    "snapshot_quarter": (event_timestamp.month - 1) // 3 + 1,
                    "snapshot_is_winter": int(event_timestamp.month in (12, 1, 2, 3)),
                    "years_since_first_donation": round(max(0.0, event_timestamp.year - int(first_donation_year[donor_index])), 2),
                    "first_donation_year": int(first_donation_year[donor_index]),
                    "lifetime_donation_count": int(lifetime_counts[donor_index] + donation_count_last_12m),
                    "last_donation_date": event_timestamp - timedelta(days=int(recency_days)),
                    "recency_days": round(recency_days, 1),
                    "days_since_last_donation_derived": round(recency_days, 1),
                    "snapshot_days_until_eligible": round(days_until_eligible, 1),
                    "snapshot_overdue_days": round(max(0.0, recency_days - typical_interval), 1),
                    "snapshot_frequency_365": donation_count_last_12m,
                    "donation_count_last_12m": donation_count_last_12m,
                    "donation_velocity": round(donation_count_last_12m / (1.0 + max(1.0, event_timestamp.year - int(first_donation_year[donor_index]))), 4),
                    "is_regular_donor": int(regular_flags[donor_index]),
                    "eligible_to_donate": eligible,
                    "eligibility_status": "eligible" if eligible else "deferred",
                    "deferral_reason": "none" if eligible else ("medical" if int(donor["chronic_condition_flag"]) else "recent_donation"),
                    "readiness_score": round(readiness_score, 4),
                    "mobility_score": round(mobility_score, 4),
                    "rare_type_priority_score": round(rare_priority, 4),
                    "recency_eligibility_gap": round(max(recency_days - days_until_eligible, 0.0), 2),
                    "travel_burden_score": round(travel_burden, 3),
                    "bmi_margin": round(abs(float(donor["bmi"]) - 24.0), 3),
                    "commitment_score": round(commitment_score, 4),
                    "friction_adjusted_commitment": round(commitment_score / (1.0 + float(donor["travel_time_min"])), 6),
                    "eligibility_buffer": round(max(recency_days - 56.0, 0.0), 2),
                    "rare_mobility_synergy": round(rare_priority * (1.0 + mobility_score), 4),
                    "response_readiness_index": round(response_readiness, 4),
                    "donor_momentum": round(donation_count_last_12m + donor_history_strength + max(recency_days - typical_interval, 0.0) / 180.0, 4),
                    "quebec_access_score": round((float(donor["public_transit_access_score"]) + route_feasible + float(donor["cold_chain_quality_score"])) / 3.0, 4),
                    "french_language_access_score": int(str(donor["language_preference"]).startswith("fr")),
                    "emergency_outreach_priority": round(rare_priority * response_readiness * eligible, 4),
                    "operational_outreach_priority": round(
                        (rare_priority * response_readiness * eligible)
                        * float(donor["digital_contactability_score"])
                        * eligible,
                        4,
                    ),
                    "distance_risk_band": pd.cut(
                        pd.Series([float(donor["center_distance_km"])]),
                        bins=[-1, 5, 15, 30, 60, 999],
                        labels=["local", "city", "regional", "remote", "extreme"],
                        include_lowest=True,
                    ).astype(str).iloc[0],
                    "donation_propensity_score": round(probability_180d * 100.0, 4),
                }
            )
            rows.append(feature_row)
            labels.append(
                {
                    "donor_id": donor["donor_id"],
                    "event_timestamp": event_timestamp,
                    "donated_within_0_3m": int(days_until_next <= 90.0),
                    "donated_within_3_6m": int(90.0 < days_until_next <= 180.0),
                    "donated_within_6m_plus": int(days_until_next > 180.0),
                    "days_until_next_donation": round(days_until_next, 2),
                    "ideal_donor_probability": round(ideal_probability, 6),
                    "donated_next_6m": int(days_until_next <= 180.0),
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(labels)


def generate_quebec_donor_dataset(*, full: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    donors = N_DONORS_FULL if full else N_DONORS_DEFAULT
    snapshots = N_SNAPSHOTS_FULL if full else N_SNAPSHOTS_DEFAULT
    static = _donor_static_rows(donors)
    features, labels = _build_snapshots(static, snapshots)
    features = features.sort_values(["donor_id", "event_timestamp"]).reset_index(drop=True)
    labels = labels.sort_values(["donor_id", "event_timestamp"]).reset_index(drop=True)
    return features, labels


def _save(df: pd.DataFrame, stem: str, directory: Path) -> None:
    df.to_parquet(directory / f"{stem}.parquet", index=False)
    df.to_csv(directory / f"{stem}.csv", index=False)
    print(f"  {stem:<35} -> {df.shape[0]:,} rows x {df.shape[1]} cols")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Quebec donor forecasting dataset")
    parser.add_argument("--full", action="store_true", help="Generate full donor snapshot dataset")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parents[1]
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Generating Quebec donor dataset...")
    features, labels = generate_quebec_donor_dataset(full=args.full)
    _save(features, "quebec_donor_features", out_dir)
    _save(labels, "quebec_donor_labels", out_dir)
    print(f"  Donors:       {features['donor_id'].nunique():,}")
    print(f"  Regions:      {features['region'].nunique():,}")
    print(f"  Blood types:  {features['blood_type'].nunique():,}")
    print(f"  Target mean:  {labels['donated_next_6m'].mean():.3f}")


if __name__ == "__main__":
    main()

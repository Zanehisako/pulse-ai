# Cell 2 + 2b — Vectorized Québec-faithful dataset generation (inventory + donor)
# Extracted from the SOTA Kaggle notebook.

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260511
OUTPUT_DIR = Path(__file__).resolve().parent.parent

CONFIG = {
    "inventory_days": 455,
    "inventory_hospitals": 42,
    "n_donors": 90000,
    "donor_snapshots": 8,
}

SMOKE = False  # set True for quick test


def downcast_frame(df):
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast="float")
        elif pd.api.types.is_integer_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast="integer")
    return df


def generate_quebec_inventory_dataset(days=None, n_hospitals=None):
    if days is None:
        days = 90 if SMOKE else CONFIG["inventory_days"]
    if n_hospitals is None:
        n_hospitals = 12 if SMOKE else CONFIG["inventory_hospitals"]
    rng = np.random.default_rng(SEED + 11)
    profiles = [
        ("CHUM", "Montréal", "Montréal", "large"),
        ("CUSM-MUHC", "Montréal", "Montréal", "large"),
        ("CHU de Québec", "Capitale-Nationale", "Québec", "large"),
        ("CISSS Laval", "Laval", "Laval", "urban"),
        ("CISSS Outaouais", "Outaouais", "Gatineau", "regional"),
        ("CIUSSS Estrie-CHUS", "Estrie", "Sherbrooke", "large"),
        ("CISSS Saguenay-Lac-Saint-Jean", "Saguenay–Lac-Saint-Jean", "Saguenay", "remote"),
        ("CISSS Bas-Saint-Laurent", "Bas-Saint-Laurent", "Rimouski", "remote"),
        ("CISSS Abitibi-Témiscamingue", "Abitibi-Témiscamingue", "Rouyn-Noranda", "remote"),
        ("CISSS Côte-Nord", "Côte-Nord", "Sept-Îles", "remote"),
        ("CRSSS Baie-James", "Nord-du-Québec", "Chibougamau", "remote"),
    ]
    blood_types = ["O+", "A+", "B+", "AB+", "O-", "A-", "B-", "AB-"]
    components = ["Red blood cells", "Plasma", "Platelets"]
    comp_factor = {"Red blood cells": 1.0, "Plasma": 0.58, "Platelets": 0.46}
    shelf_life = {"Red blood cells": 42, "Plasma": 365, "Platelets": 5}
    waste_rate = {"Red blood cells": 0.020, "Plasma": 0.007, "Platelets": 0.080}
    scarcity = {"O+": 0.45, "A+": 0.48, "B+": 0.68, "AB+": 0.78,
                "O-": 0.95, "A-": 0.88, "B-": 0.96, "AB-": 1.0}
    size_map = {"large": 1.55, "urban": 1.18, "regional": 0.82, "remote": 0.54}

    hospitals = []
    for i in range(n_hospitals):
        base = profiles[i % len(profiles)]
        hospitals.append({
            "hospital_id": f"QC-HOSP-{i+1:03d}",
            "hospital_name": base[0] + (f" satellite {i//len(profiles)+1}" if i >= len(profiles) else ""),
            "region_admin": base[1], "city": base[2], "hospital_size_tier": base[3],
        })

    n_combos = n_hospitals * len(blood_types) * len(components)
    combo_size = np.zeros(n_combos)
    combo_cf = np.zeros(n_combos)
    combo_wr = np.zeros(n_combos)
    combo_sc = np.zeros(n_combos)
    combo_shelf = np.zeros(n_combos, dtype=int)
    combo_stress = np.zeros(n_combos)
    hospital_ids, hospital_names, region_admins, cities = [], [], [], []
    combo_bt, combo_comp, combo_tier = [], [], []

    idx = 0
    hospital_stresses = rng.beta(2.8, 3.6, n_hospitals)
    for hi, hp in enumerate(hospitals):
        hs = hp["hospital_size_tier"]
        sz = size_map[hs]
        stress = hospital_stresses[hi]
        for bt in blood_types:
            for comp in components:
                combo_size[idx] = sz
                combo_cf[idx] = comp_factor[comp]
                combo_wr[idx] = waste_rate[comp]
                combo_sc[idx] = scarcity[bt]
                combo_shelf[idx] = shelf_life[comp]
                combo_stress[idx] = stress
                hospital_ids.append(hp["hospital_id"])
                hospital_names.append(hp["hospital_name"])
                region_admins.append(hp["region_admin"])
                cities.append(hp["city"])
                combo_bt.append(bt)
                combo_comp.append(comp)
                combo_tier.append(hs)
                idx += 1

    combo_remote = 1 + 0.55*(np.array(combo_tier) == "remote").astype(float) + 0.25*(np.array(combo_tier) == "regional").astype(float)
    combo_rare = 1 + 0.55*np.array([1.0 if bt in ["O-", "A-", "B-", "AB-"] else 0.0 for bt in combo_bt])
    combo_platelet = 1 + 0.30*np.array([1.0 if c == "Platelets" else 0.0 for c in combo_comp])

    safety = np.maximum(2.0, 13.0*combo_size*combo_cf*(0.85+combo_sc)*combo_remote*combo_rare*combo_platelet*(0.95+0.50*combo_stress))
    inv = np.maximum(safety*rng.uniform(0.90, 1.55, n_combos), rng.normal(safety*rng.uniform(1.25, 2.25, n_combos), safety*0.28))
    is_remote = (np.array(combo_tier) == "remote").astype(float)
    is_rare_bt = np.array([1.0 if bt in ["O-", "A-", "B-", "AB-"] else 0.0 for bt in combo_bt])
    is_platelet = np.array([1.0 if c == "Platelets" else 0.0 for c in combo_comp])

    disruption_memory = np.zeros(n_combos)
    replenishment_memory = np.zeros(n_combos)
    shortage_memory = np.zeros(n_combos)

    all_data = {k: [] for k in [
        "day_index", "month", "dow", "weekend", "holiday", "flu_season_flag",
        "major_event", "supply_disruption", "donor_shortfall", "current_inventory",
        "units_used", "units_collected", "wastage", "incoming_supply_scheduled_7d",
        "lead_time_days", "route_disruption_score", "trauma_cases",
        "scheduled_surgeries", "blood_shortage",
    ]}

    start = pd.Timestamp("2024-01-01", tz="UTC")
    for d in range(days):
        ts = start + pd.Timedelta(days=d)
        dow, month = ts.dayofweek, ts.month
        winter = float(month in [12, 1, 2, 3])
        holiday = float((month == 12 and ts.day >= 24) or (month == 1 and ts.day <= 2))
        flu = float(month in [11, 12, 1, 2, 3])

        major = (rng.random(n_combos) < (0.020 + 0.034*combo_stress + 0.012*(dow >= 5) + 0.010*winter)).astype(float)
        supply_disruption = (rng.random(n_combos) < (0.045 + 0.070*combo_stress + 0.050*winter + 0.060*is_remote + 0.025*is_rare_bt)).astype(float)
        donor_shortfall = (rng.random(n_combos) < (0.060 + 0.050*winter + 0.040*combo_sc + 0.030*is_platelet)).astype(float)
        weather = np.maximum(0, rng.normal(3.0*winter, 2.35, n_combos))
        route = np.clip(0.42*weather + 1.35*holiday + 2.05*major + 2.60*supply_disruption + 0.70*disruption_memory + rng.normal(0, 0.70, n_combos), 0, 12)
        disruption_memory = 0.62*disruption_memory + 0.38*route

        trauma = rng.poisson(np.maximum(0.1, 1.25*combo_size + 0.42*(dow >= 5) + 0.30*route + 2.2*major))
        surgeries = rng.poisson(np.maximum(0.1, 7.8*combo_size*(dow < 5) + 1.4))
        base_usage = 3.85*combo_size*combo_cf + 0.42*trauma + 0.115*surgeries + 0.48*flu + 0.22*shortage_memory
        usage = np.maximum(0, rng.normal(base_usage*(1 + 0.24*major), 2.4))
        supply_mult = np.clip(0.90 - 0.42*supply_disruption - 0.26*donor_shortfall - 0.075*route - 0.14*is_remote - 0.06*is_rare_bt, 0.10, 1.05)
        planned_supply_7d = np.maximum(0, rng.normal(usage*4.85*supply_mult + 0.28*replenishment_memory, 5.8))
        lead = np.clip(rng.normal(2.35 + 0.34*route + 1.20*is_remote + 0.55*supply_disruption + 0.20*is_rare_bt, 0.9), 1, 12)
        wastage = np.maximum(0, inv*combo_wr*(1 + 0.10*winter) + rng.normal(0, 0.45, n_combos))
        coll_mult = np.clip(0.70 + 0.17*(1-combo_stress) - 0.10*winter - 0.18*donor_shortfall - 0.10*supply_disruption, 0.25, 0.95)
        collected = np.maximum(0, rng.normal((usage+wastage)*coll_mult + planned_supply_7d/11, 4.2))
        actual_used = usage + np.where(major > 0, rng.gamma(3.0, 1.7, n_combos), rng.gamma(1.2, 0.38, n_combos))
        inv = np.clip(inv + collected + planned_supply_7d/11 - actual_used - wastage - 0.12*route, 0, 320)
        projected_7d = inv + planned_supply_7d - 7*np.maximum(actual_used, 0) - 2*np.maximum(wastage, 0)
        shortage = ((inv <= safety) | (projected_7d <= 0.85*safety)).astype(float)
        shortage_memory = 0.70*shortage_memory + 1.8*shortage
        replenishment_memory = np.maximum(0, safety*3.2 - inv)*(0.12 + 0.06*(1-supply_disruption))

        platelet_topup = (shortage > 0) & (is_platelet > 0) & (rng.random(n_combos) < 0.34)
        other_topup = (shortage > 0) & (~(is_platelet > 0).astype(bool)) & (rng.random(n_combos) < 0.20)
        inv = np.where(platelet_topup, inv + rng.uniform(0.22, 0.58, n_combos)*safety, inv)
        inv = np.where(other_topup, inv + rng.uniform(0.10, 0.40, n_combos)*safety, inv)

        all_data["day_index"].append(np.full(n_combos, d, dtype=np.int16))
        all_data["month"].append(np.full(n_combos, month, dtype=np.int8))
        all_data["dow"].append(np.full(n_combos, dow, dtype=np.int8))
        all_data["weekend"].append(np.full(n_combos, int(dow >= 5), dtype=np.int8))
        all_data["holiday"].append(np.full(n_combos, int(holiday), dtype=np.int8))
        all_data["flu_season_flag"].append(np.full(n_combos, int(flu), dtype=np.int8))
        all_data["major_event"].append(major.astype(np.int8))
        all_data["supply_disruption"].append(supply_disruption.astype(np.int8))
        all_data["donor_shortfall"].append(donor_shortfall.astype(np.int8))
        all_data["current_inventory"].append(inv.astype(np.float32))
        all_data["units_used"].append(actual_used.astype(np.float32))
        all_data["units_collected"].append(collected.astype(np.float32))
        all_data["wastage"].append(wastage.astype(np.float32))
        all_data["incoming_supply_scheduled_7d"].append(planned_supply_7d.astype(np.float32))
        all_data["lead_time_days"].append(lead.astype(np.float32))
        all_data["route_disruption_score"].append(route.astype(np.float32))
        all_data["trauma_cases"].append(trauma.astype(np.int16))
        all_data["scheduled_surgeries"].append(surgeries.astype(np.int16))
        all_data["blood_shortage"].append(shortage.astype(np.int8))

    frame = pd.DataFrame({
        "hospital_id": np.tile(hospital_ids, days),
        "hospital_name": np.tile(hospital_names, days),
        "region_admin": np.tile(region_admins, days),
        "city": np.tile(cities, days),
        "hospital_size_tier": np.tile(combo_tier, days),
        "supplier_context": "Héma-Québec",
        "province": "Québec",
        "blood_type": np.tile(combo_bt, days),
        "component_type": np.tile(combo_comp, days),
        "shelf_life_days": np.tile(combo_shelf, days),
        "safety_threshold": np.tile(safety, days).astype(np.float32),
        "component_criticality_score": np.tile(combo_sc * combo_cf, days).astype(np.float32),
        "hospital_stress_index": np.tile(combo_stress, days).astype(np.float32),
    })
    timestamps = pd.date_range(start, periods=days, freq="D", tz="UTC")
    frame["event_timestamp"] = np.repeat(timestamps, n_combos)
    for k, v in all_data.items():
        frame[k] = np.concatenate(v)

    frame.sort_values(["hospital_id", "blood_type", "component_type", "event_timestamp"], inplace=True)
    frame.reset_index(drop=True, inplace=True)

    group_cols = ["hospital_id", "blood_type", "component_type"]
    g = frame.groupby(group_cols, sort=False)
    for w in [3, 7, 14, 30, 90]:
        frame[f"inventory_{w}d_mean"] = g["current_inventory"].transform(lambda s: s.rolling(w, min_periods=1).mean())
        frame[f"usage_{w}d_mean"] = g["units_used"].transform(lambda s: s.rolling(w, min_periods=1).mean())
        frame[f"collection_{w}d_mean"] = g["units_collected"].transform(lambda s: s.rolling(w, min_periods=1).mean())
        frame[f"wastage_{w}d_mean"] = g["wastage"].transform(lambda s: s.rolling(w, min_periods=1).mean())
        frame[f"stockout_count_{w}d"] = g["blood_shortage"].transform(lambda s: s.shift(1).rolling(w, min_periods=1).sum()).fillna(0)
    frame["inventory_runway_days"] = frame["current_inventory"]/(1+frame["units_used"])
    frame["collection_adjusted_runway_days"] = frame["current_inventory"]/(1+np.maximum(frame["units_used"]+frame["wastage"]-0.35*frame["units_collected"], 0))

    labels = frame[group_cols + ["event_timestamp"]].copy()
    for h in [1, 7, 30]:
        for col in ["wastage", "units_used", "units_collected"]:
            labels[f"{col}_next_{h}d"] = g[col].transform(
                lambda s: s.iloc[::-1].rolling(h, min_periods=1).sum().iloc[::-1].shift(-1)
            ).fillna(0).astype(np.float32)
    for name, horizon in [("stockout_within_0_7d", 7), ("stockout_within_8_30d", 30),
                          ("stockout_within_31_90d", 90), ("stockout_within_91_180d", 180)]:
        labels[name] = g["blood_shortage"].transform(
            lambda s: (s.iloc[::-1].rolling(horizon, min_periods=1).sum().iloc[::-1].shift(-1) > 0).astype(int)
        ).fillna(0).astype(np.int8)
    labels["days_until_stockout"] = 180.0

    frame.drop(columns=["blood_shortage"], inplace=True)
    return downcast_frame(frame), downcast_frame(labels)


def generate_quebec_donor_dataset(n=None, snapshots=None):
    if n is None:
        n = 8000 if SMOKE else CONFIG["n_donors"]
    if snapshots is None:
        snapshots = 3 if SMOKE else CONFIG["donor_snapshots"]
    rng = np.random.default_rng(SEED + 12)
    blood_types = np.array(["O+", "A+", "B+", "AB+", "O-", "A-", "B-", "AB-"])
    probs = np.array([.38, .34, .09, .03, .07, .06, .02, .01])
    regions = ["Montréal", "Montérégie", "Capitale-Nationale", "Laval", "Outaouais",
               "Estrie", "Saguenay–Lac-Saint-Jean", "Côte-Nord", "Gaspésie", "Nord-du-Québec"]
    region_probs = np.array([.28, .17, .12, .07, .07, .06, .05, .04, .03, .01])
    region_probs = region_probs / region_probs.sum()

    donor_ids = np.array([f"QC-DONOR-{i:07d}" for i in range(n)])
    reg = rng.choice(regions, n, p=region_probs)
    age = np.clip(rng.normal(42, 14, n), 18, 70).astype(int)
    sex = rng.choice(["F", "M", "X"], n, p=[.51, .48, .01])
    lang = rng.choice(["fr", "en", "fr_en"], n, p=[.78, .12, .10])
    bt = rng.choice(blood_types, n, p=probs)
    digital = rng.beta(4.2, 2.0, n).astype(np.float32)
    dist = np.clip(rng.gamma(2.0, 10.5, n), .5, 180).astype(np.float32)
    transit = np.clip(rng.normal(.66, .20, n), 0, 1).astype(np.float32)
    winter_risk = np.clip(rng.beta(2.2, 5.2, n), 0, 1).astype(np.float32)
    smoker = rng.binomial(1, .13, n).astype(np.int8)
    bmi = np.clip(rng.normal(25.4, 4.1, n), 18, 39.5).astype(np.float32)
    chronic = rng.binomial(1, .07, n).astype(np.int8)
    is_rare = np.isin(bt, ["O-", "A-", "B-", "AB-", "AB+"]).astype(np.int8)

    first_year = rng.integers(2008, 2024, n)
    lifetime = (rng.poisson(8, n) + rng.binomial(18, .22, n)).astype(np.int16)
    regular = rng.binomial(1, .42, n).astype(np.int8)
    hist = np.clip(.7*regular + .08*lifetime + rng.normal(0, .35, n), 0, 4).astype(np.float32)
    interval = np.clip(rng.normal(115 - hist*12, 28), 56, 240).astype(np.float32)
    base_gap = np.clip(rng.normal(interval, 38), 20, 420).astype(np.float32)

    start = pd.Timestamp("2024-01-01", tz="UTC")
    total = n * snapshots

    t_ids = np.tile(donor_ids, snapshots)
    t_reg = np.tile(reg, snapshots)
    t_age = np.tile(age, snapshots)
    t_sex = np.tile(sex, snapshots)
    t_lang = np.tile(lang, snapshots)
    t_bt = np.tile(bt, snapshots)
    t_digital = np.tile(digital, snapshots)
    t_dist = np.tile(dist, snapshots)
    t_transit = np.tile(transit, snapshots)
    t_winter_risk = np.tile(winter_risk, snapshots)
    t_smoker = np.tile(smoker, snapshots)
    t_bmi = np.tile(bmi, snapshots)
    t_chronic = np.tile(chronic, snapshots)
    t_is_rare = np.tile(is_rare, snapshots)
    t_regular = np.tile(regular, snapshots)
    t_lifetime = np.tile(lifetime, snapshots)
    t_hist = np.tile(hist, snapshots)
    t_first_year = np.tile(first_year, snapshots)
    t_base_gap = np.tile(base_gap, snapshots)

    snapshot_idx = np.repeat(np.arange(snapshots), n)
    timestamps = [start + pd.Timedelta(days=45*s) for s in range(snapshots)]
    t_ts = np.repeat(pd.DatetimeIndex(timestamps), n)
    t_month = np.array([ts.month for ts in timestamps], dtype=np.int8)
    t_snap_month = np.repeat(t_month, n)
    t_winter = np.repeat(np.array([int(ts.month in [12, 1, 2, 3]) for ts in timestamps], dtype=np.int8), n)
    t_years_since = np.maximum(0, np.repeat([ts.year for ts in timestamps], n) - t_first_year)

    recency = np.maximum(0, t_base_gap + 45*snapshot_idx - rng.normal(0, 18, total)).astype(np.float32)
    count12 = np.clip(rng.poisson(np.maximum(.2, 3.4 - np.tile(interval, snapshots)/70 + t_hist*.35)), 0, 6).astype(np.int8)
    eligible_days = np.maximum(0, 56 - recency).astype(np.float32)
    eligible = ((eligible_days <= 0) & (t_chronic == 0) & (t_age >= 18) & (t_age <= 70) & (t_bmi >= 18.5) & (t_bmi <= 38)).astype(np.int8)
    rare_priority = (1 + 1.5*t_is_rare).astype(np.float32)
    mobility = (1.15*(t_dist <= 55) + .45*t_transit - .55*t_winter_risk*(1+.45*t_winter)).astype(np.float32)
    readiness = (1.5*eligible + .45*count12 + .55*t_regular + .30*t_hist - .30*t_smoker).astype(np.float32)
    response = (.42*readiness + .22*mobility + .22*(count12+2.5*t_regular+t_hist) + .14*rare_priority - .012*t_dist).astype(np.float32)
    days_next = np.clip(295 - 44*response + .58*eligible_days + .30*t_dist - 8*count12 + rng.normal(0, 16, total), 1, 420).astype(np.float32)

    features = pd.DataFrame({
        "donor_id": t_ids, "province": "Québec", "supplier_context": "Héma-Québec",
        "region_admin": t_reg, "age": t_age, "sex": t_sex, "language_preference": t_lang,
        "blood_type": t_bt, "digital_contactability_score": t_digital,
        "center_distance_km": t_dist, "public_transit_access_score": t_transit,
        "winter_route_risk_score": t_winter_risk, "smoker": t_smoker, "bmi": t_bmi,
        "chronic_condition_flag": t_chronic, "is_rare_type": t_is_rare,
        "event_timestamp": t_ts, "snapshot_month": t_snap_month, "snapshot_is_winter": t_winter,
        "years_since_first_donation": t_years_since,
        "lifetime_donation_count": (t_lifetime + count12).astype(np.int16),
        "recency_days": recency, "snapshot_days_until_eligible": eligible_days,
        "snapshot_frequency_365": count12,
        "donation_velocity": (count12/(1+np.maximum(1, t_years_since))).astype(np.float32),
        "is_regular_donor": t_regular, "eligible_to_donate": eligible,
        "readiness_score": readiness, "mobility_score": mobility,
        "response_readiness_index": response,
        "donor_momentum": (count12 + t_hist).astype(np.float32),
        "operational_outreach_priority": (rare_priority*response*eligible*t_digital).astype(np.float32),
    })
    labels = pd.DataFrame({
        "donor_id": t_ids, "event_timestamp": t_ts,
        "days_until_next_donation": days_next,
        "donated_next_6m": (days_next <= 180).astype(np.int8),
        "donated_within_0_3m": (days_next <= 90).astype(np.int8),
        "donated_within_3_6m": ((days_next > 90) & (days_next <= 180)).astype(np.int8),
        "donated_within_6m_plus": (days_next > 180).astype(np.int8),
    })
    return downcast_frame(features), downcast_frame(labels)


def main():
    parser = argparse.ArgumentParser(description="Generate SOTA training datasets")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--smoke", action="store_true", help="Small dataset for testing")
    args = parser.parse_args()

    global SMOKE
    SMOKE = args.smoke

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    print("Generating inventory dataset...")
    inv_features, inv_labels = generate_quebec_inventory_dataset()
    print(f"  Inventory features: {inv_features.shape}")
    print(f"  Inventory labels: {inv_labels.shape}")
    print(f"  Stockout 0-7d rate: {float(inv_labels['stockout_within_0_7d'].mean()):.4f}")

    print("Generating donor dataset...")
    donor_features, donor_labels = generate_quebec_donor_dataset()
    print(f"  Donor features: {donor_features.shape}")
    print(f"  Donor labels: {donor_labels.shape}")

    inv_features.to_parquet(out / "sota_inventory_features.parquet", index=False)
    inv_labels.to_parquet(out / "sota_inventory_labels.parquet", index=False)
    donor_features.to_parquet(out / "sota_donor_features.parquet", index=False)
    donor_labels.to_parquet(out / "sota_donor_labels.parquet", index=False)
    print(f"Datasets saved to {out}/")


if __name__ == "__main__":
    main()

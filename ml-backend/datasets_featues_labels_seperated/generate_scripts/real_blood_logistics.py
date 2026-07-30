import os
import random
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

# --------------------
# CONFIG
# --------------------

N_DONORS = 5000
N_HOSPITALS = 25
DAYS = 365

START_DATE = datetime(2022, 1, 1, tzinfo=timezone.utc)

blood_groups = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

# RBC shelf life
EXPIRATION_DAYS = 42

# minimum donation interval
DONATION_INTERVAL = 56

# --------------------
# CREATE DONORS
# --------------------

donors = []

for i in range(N_DONORS):
    donors.append(
        {
            "donor_id": f"D{i:06}",
            "blood_group": random.choice(blood_groups),
            "lat": 30 + random.random(),
            "lon": 1 + random.random(),
            "last_donation": START_DATE - timedelta(days=random.randint(0, 120)),
        }
    )

donors = pd.DataFrame(donors)

# --------------------
# HOSPITALS
# --------------------

hospitals = []

for i in range(N_HOSPITALS):
    hospitals.append(
        {
            "hospital_id": f"H{i:03}",
            "lat": 30 + random.random(),
            "lon": 1 + random.random(),
            "base_demand": random.randint(5, 20),
        }
    )

hospitals = pd.DataFrame(hospitals)

# --------------------
# STORAGE — each hospital has its own inventory
# --------------------

# FIX 1: Per-hospital inventory instead of one global pool
hospital_ids = [f"H{i:03}" for i in range(N_HOSPITALS)]
hospital_inventories = {h_id: [] for h_id in hospital_ids}

donations = []
features_rows = []
labels_rows = []

# Track last 7-day donation count per hospital for a lagged feature
hospital_recent_donations = {h_id: [] for h_id in hospital_ids}

# --------------------
# SIMULATION LOOP
# --------------------

for day in range(DAYS):
    current_date = START_DATE + timedelta(days=day)

    # weather — shared across all hospitals on a given day
    temperature = np.random.normal(20, 7)
    rain = max(0, np.random.normal(3, 5))
    disaster = np.random.choice([0, 1], p=[0.98, 0.02])
    holiday = np.random.choice([0, 1], p=[0.9, 0.1])

    # ----------------
    # donor donations — distributed to hospitals randomly
    # ----------------
    for idx, donor in donors.iterrows():
        days_since = (current_date - donor["last_donation"]).days

        if days_since >= DONATION_INTERVAL:
            # FIX 2: lower donation rate + more randomness → realistic supply variability
            donate_prob = 0.015 if not holiday else 0.025
            if random.random() < donate_prob:
                donations.append(
                    {
                        "donor_id": donor["donor_id"],
                        "blood_group": donor["blood_group"],
                        "event_timestamp": current_date,
                    }
                )
                donors.at[idx, "last_donation"] = current_date

                # FIX 3: assign donation to a specific hospital, not global pool
                target_hospital = random.choice(hospital_ids)
                hospital_inventories[target_hospital].append(
                    {
                        "blood_group": donor["blood_group"],
                        "collection_date": current_date,
                        "expires": current_date + timedelta(days=EXPIRATION_DAYS),
                    }
                )
                hospital_recent_donations[target_hospital].append(current_date)

    # remove expired blood per hospital
    for h_id in hospital_ids:
        hospital_inventories[h_id] = [
            b for b in hospital_inventories[h_id]
            if b["expires"] > current_date
        ]
        # keep only last 7 days of donation timestamps for lagged feature
        cutoff = current_date - timedelta(days=7)
        hospital_recent_donations[h_id] = [
            d for d in hospital_recent_donations[h_id]
            if d >= cutoff
        ]

    # ----------------
    # hospital demand
    # ----------------
    for _, hospital in hospitals.iterrows():
        h_id = hospital["hospital_id"]
        surgeries = np.random.poisson(3)
        trauma = np.random.poisson(2)

        demand = hospital["base_demand"] + surgeries * 2 + trauma * 3

        if disaster:
            demand += np.random.randint(10, 30)  # variable disaster impact

        if holiday:
            demand = max(1, int(demand * random.uniform(0.6, 0.9)))  # lower elective on holidays

        stock = len(hospital_inventories[h_id])

        # FIX 4: remove current_inventory from features — it directly determines shortage
        # Use indirect/lagged signals instead:
        #   - stock_lag_ratio: stock vs rolling average demand (indirect)
        #   - donations_last_7d: supply-side trend
        #   - days_since_last_restock: temporal signal
        donations_last_7d = len(hospital_recent_donations[h_id])

        # rolling average demand proxy (base demand + seasonal effects)
        avg_demand_proxy = hospital["base_demand"] + 3 * 2 + 2 * 3  # using mean of poisson
        stock_lag_ratio = stock / max(avg_demand_proxy, 1)

        # days since last donation to this hospital
        if hospital_recent_donations[h_id]:
            last_restock = hospital_recent_donations[h_id][-1]
            days_since_restock = (current_date - last_restock).days
        else:
            days_since_restock = day  # never restocked

        # FIX 5: probabilistic label — not a hard deterministic rule
        # sigmoid over (demand - stock) with noise
        imbalance = demand - stock
        noise = random.gauss(0, 2.5)  # realistic noise
        shortage_logit = (imbalance + noise) / 6.0
        shortage_probability = 1 / (1 + np.exp(-shortage_logit))
        shortage = int(random.random() < shortage_probability)

        features_rows.append(
            {
                "hospital_id": h_id,
                "event_timestamp": current_date,
                # Environmental features
                "temperature": round(temperature, 2),
                "rain_mm": round(rain, 2),
                "holiday": holiday,
                "disaster": disaster,
                # Demand-side features
                "scheduled_surgeries": surgeries,
                "trauma_cases": trauma,
                # FIX 6: indirect supply signals (not raw stock)
                "stock_lag_ratio": round(stock_lag_ratio, 4),
                "donations_last_7d": donations_last_7d,
                "days_since_last_restock": days_since_restock,
            }
        )

        labels_rows.append(
            {
                "hospital_id": h_id,
                "event_timestamp": current_date,
                "blood_shortage": shortage,
            }
        )

        # consume blood from this hospital's inventory
        used = min(stock, demand)
        hospital_inventories[h_id] = hospital_inventories[h_id][used:]

# --------------------
# SAVE DATASETS
# --------------------

SCRIPTS_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
ROOT_DIRECTORY = os.path.dirname(SCRIPTS_DIRECTORY)

features_df = pd.DataFrame(features_rows)
labels_df = pd.DataFrame(labels_rows)
donations_df = pd.DataFrame(donations)

print("Features shape:", features_df.shape)
print("Labels shape:", labels_df.shape)
print("Donations shape:", donations_df.shape)
print("\nLabel distribution:")
print(labels_df["blood_shortage"].value_counts())
print(f"\nShortage rate: {labels_df['blood_shortage'].mean():.2%}")
print("\nFeature columns:", features_df.columns.tolist())
print("\nSample features:")
print(features_df.head())

donations_df.to_parquet(ROOT_DIRECTORY + "/donations.parquet", index=False)

features_df.to_parquet(
    ROOT_DIRECTORY + "/hospital_supply_features.parquet", index=False
)

labels_df.to_parquet(
    ROOT_DIRECTORY + "/hospital_supply_labels.parquet", index=False
)

print("\nDataset generated successfully")
import os
from datetime import datetime, timedelta, timezone
import random

import numpy as np
import pandas as pd

# ----------------------------
# CONFIG
# ----------------------------
N_ROWS = 10000
N_CITIES = 20
START_DATE = datetime(2022, 1, 1)

# ----------------------------
# RANDOM GENERATION HELPERS
# ----------------------------
blood_groups = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]


def random_timestamp():
    return START_DATE + timedelta(days=np.random.randint(0, 365))


# ----------------------------
# GENERATE DATA
# ----------------------------
rows = []

for i in range(N_ROWS):
    donor_id = f"D{i:07d}"
    city_id = f"C{np.random.randint(0, N_CITIES):03d}"

    lat = 30 + np.random.rand()
    lon = 1 + np.random.rand()

    availability = np.random.choice([0, 1], p=[0.3, 0.7])
    blood_group = np.random.choice(blood_groups)

    recency_days = np.random.randint(1, 365)
    frequency_365 = np.random.poisson(2)

    days_until_eligible = np.random.randint(0, 90)

    cluster_id = np.random.randint(0, 10)
    base_score = (
        (frequency_365 / 5.0)           # normalized
        + (1 - recency_days / 365.0)    # normalized
        + availability * 0.5
        - days_until_eligible / 90.0
    )
    noise = random.gauss(0, 0.3)
    probability = 1 / (1 + np.exp(-(base_score + noise)))
    good_donor = int(random.random() < probability)

    # target label: "good donor"
    good_donor = int(
        (frequency_365 >= 3) and (recency_days < 120) and (availability == 1)
    )
    rows.append(
        {
            "donor_id": donor_id,
            "event_timestamp": random_timestamp(),
            # features
            "city_id": city_id,
            "lat": lat,
            "lon": lon,
            "blood_group": blood_group,
            "recency_days": recency_days,
            "frequency_365": frequency_365,
            "days_until_eligible": days_until_eligible,
            "cluster_id": cluster_id,
            # label
            "good_donor": good_donor,
        }
    )

df = pd.DataFrame(rows)

# ----------------------------
# SPLIT FEATURES / LABELS
# ----------------------------
feature_cols = [
    "donor_id",
    "event_timestamp",
    "city_id",
    "lat",
    "lon",
    "blood_group",
    "recency_days",
    "frequency_365",
    "days_until_eligible",
    "cluster_id",
]

label_cols = ["donor_id", "event_timestamp", "good_donor"]

features_df = df[feature_cols]
labels_df = df[label_cols]

# ----------------------------
# SAVE PARQUET
# ----------------------------
SCRIPTS_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
ROOT_DIRECTORY = os.path.dirname(SCRIPTS_DIRECTORY)

features_df.to_parquet(ROOT_DIRECTORY + "/donor_features.parquet", index=False)
labels_df.to_parquet(ROOT_DIRECTORY + "/donor_labels.parquet", index=False)

print("Dataset generated successfully")
print("Features:", features_df.shape)
print("Labels:", labels_df.shape)

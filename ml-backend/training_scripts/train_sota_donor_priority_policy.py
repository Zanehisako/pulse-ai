# Cell 10 — Donor Model D: Priority policy (decision-support only)
# Usage: python train_sota_donor_priority_policy.py

import gc
import sys
from pathlib import Path

import numpy as np
import joblib

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from sota_training_utils import (
    CONFIG, OUTPUT_DIR, DATA_DIR, flush_gpu,
    donor_group_holdout_split, save_json,
    apply_donor_proxy_policy, all_model_metrics, readiness_gates,
)


def main():
    print("=" * 70)
    print("Donor Model D: Priority Policy (decision-support)")
    print("=" * 70)

    import pandas as pd
    don_f_path = DATA_DIR / "sota_donor_features.parquet"
    don_l_path = DATA_DIR / "sota_donor_labels.parquet"
    if not (don_f_path.exists() and don_l_path.exists()):
        print(f"Data not found at {DATA_DIR}. Run generate_sota_dataset.py first.")
        return
    donor_features = pd.read_parquet(don_f_path)
    donor_labels = pd.read_parquet(don_l_path)

    donor_df = donor_features.merge(donor_labels, on=["donor_id", "event_timestamp"], how="inner")

    donor_features_cols = [
        "province", "supplier_context", "region_admin", "age", "sex", "language_preference",
        "blood_type", "digital_contactability_score", "center_distance_km", "public_transit_access_score",
        "winter_route_risk_score", "smoker", "bmi", "chronic_condition_flag", "is_rare_type",
        "snapshot_month", "snapshot_is_winter", "years_since_first_donation", "lifetime_donation_count",
        "recency_days", "snapshot_days_until_eligible", "snapshot_frequency_365", "donation_velocity",
        "is_regular_donor", "eligible_to_donate", "readiness_score", "mobility_score",
        "response_readiness_index", "donor_momentum", "operational_outreach_priority"
    ]
    donor_features_cols = apply_donor_proxy_policy(
        [c for c in donor_features_cols if c in donor_df.columns], "donor_priority_policy"
    )
    train, valid, test = donor_group_holdout_split(donor_df)
    del donor_df
    gc.collect()

    priority_cols = ["operational_outreach_priority", "readiness_score", "response_readiness_index",
                     "is_rare_type", "eligible_to_donate"]
    if all(c in test.columns for c in priority_cols):
        ps = (test["operational_outreach_priority"].astype(float)*0.34
              + test["readiness_score"].astype(float)*0.26
              + test["response_readiness_index"].astype(float)*0.18
              + test["is_rare_type"].astype(float)*0.12
              + test["eligible_to_donate"].astype(float)*0.10)
        ps = (ps - ps.min()) / max(ps.max() - ps.min(), 1e-9)
        priority_metrics = {"urgent_contact_rate": float((ps >= .85).mean()), "proxy_policy": True}
    else:
        priority_metrics = {"urgent_contact_rate": None, "proxy_policy": True,
                            "disabled_reason": "missing_priority_columns"}

    policy = {"name": "global_sota_donor_priority_policy_model", "type": "policy",
              "inputs": priority_cols, "decision_support_only": True}
    joblib.dump(policy, OUTPUT_DIR / "global_sota_donor_priority_policy_model.joblib")
    save_json(priority_metrics, OUTPUT_DIR / "global_sota_donor_priority_policy_model_metrics.json")
    all_model_metrics["global_sota_donor_priority_policy_model"] = priority_metrics
    readiness_gates["global_sota_donor_priority_policy_model"] = {
        "operational_ready": False, "reason": "proxy_policy_requires_real_outreach_utility_validation"
    }

    print(f"\nDonor priority policy saved.")
    print(f"  Urgent contact rate (>=85th pct): {priority_metrics.get('urgent_contact_rate', 'N/A')}")


if __name__ == "__main__":
    main()

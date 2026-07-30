# Cell 10 — Donor Model C: Contact propensity (tabular only, proxy)
# Usage: python train_sota_donor_contact_propensity.py

import gc
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from sota_training_utils import (
    CONFIG, OUTPUT_DIR, DATA_DIR, flush_gpu,
    donor_group_holdout_split,
    encode_ohe, fit_tabular_classifier, best_f1_threshold, classification_metrics,
    fit_sota_classification_baselines, choose_winner,
    evaluate_classification_gate, save_bundle,
    apply_donor_proxy_policy, all_model_metrics, readiness_gates,
)


def main():
    print("=" * 70)
    print("Donor Model C: Contact Propensity (tabular)")
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
        [c for c in donor_features_cols if c in donor_df.columns], "donor_contact_propensity"
    )
    train, valid, test = donor_group_holdout_split(donor_df)
    del donor_df
    gc.collect()

    CONTACT_TARGET_CANDIDATES = {
        "donor_contact_propensity_model": ["contact_propensity_label", "contacted_next_campaign", "responded_to_contact"],
    }
    target = next((c for c in CONTACT_TARGET_CANDIDATES["donor_contact_propensity_model"] if c in train.columns), None)

    if target is None or len(np.unique(train[target].astype(int))) <= 1:
        print("No real contact propensity label found — saving proxy-only model.")
        proxy_metrics = {
            "task": "donor_contact_propensity_proxy", "winner": "tabular_gbdt",
            "proxy_only": True, "operational_ready": False,
            "note": "Requires real contact-propensity labels before operational use.",
        }
        bundle = {"tabular_model": None, "tabular_columns": [], "feature_cols": donor_features_cols,
                  "target": "proxy", "winner": "tabular_gbdt", "optional_sota_models": {}}
        save_bundle("global_sota_donor_contact_propensity_model", bundle, proxy_metrics)
        all_model_metrics["global_sota_donor_contact_propensity_model"] = proxy_metrics
        readiness_gates["global_sota_donor_contact_propensity_model"] = {
            "operational_ready": False, "reason": "proxy_only_without_real_contact_response_labels"
        }
        return

    ctx, cvx, ctex, ccols = encode_ohe(train, valid, test, donor_features_cols)
    contact_tab_name, contact_tab_model, _ = fit_tabular_classifier(
        ctx, train[target].astype(int), cvx, valid[target].astype(int)
    )
    contact_valid_prob = contact_tab_model.predict_proba(cvx)[:, 1]
    contact_thr = best_f1_threshold(valid[target].astype(int), contact_valid_prob)
    contact_valid_metrics = classification_metrics(valid[target].astype(int), contact_valid_prob, contact_thr)
    contact_prob = contact_tab_model.predict_proba(ctex)[:, 1]
    contact_tab_metrics = classification_metrics(test[target].astype(int), contact_prob, contact_thr)

    contact_sota = fit_sota_classification_baselines(train, valid, test, donor_features_cols, target, "donor_contact_propensity")
    contact_valid_comparison = {"tabular_gbdt": contact_valid_metrics, **contact_sota["valid"]}
    contact_test_comparison = {"tabular_gbdt": contact_tab_metrics, **contact_sota["test"]}
    contact_winner = choose_winner(contact_valid_comparison, "auc", higher_is_better=True)
    contact_metrics = {
        "task": target, "winner": contact_winner, "proxy_only": False,
        "selection_metric": "validation_auc", "validation_protocol": "donor_group_holdout",
        "validation_comparison": contact_valid_comparison, "test_comparison": contact_test_comparison,
        "tabular_overall": contact_tab_metrics,
        "sota_baselines": {"availability": contact_sota["availability"], "thresholds": contact_sota.get("thresholds", {})},
    }
    contact_bundle = {"tabular_model": contact_tab_model, "tabular_columns": ccols,
                      "feature_cols": donor_features_cols, "target": target, "winner": contact_winner,
                      "optional_sota_models": {k: v for k, v in contact_sota["models"].items() if k != "autogluon"}}
    save_bundle("global_sota_donor_contact_propensity_model", contact_bundle, contact_metrics)
    all_model_metrics["global_sota_donor_contact_propensity_model"] = contact_metrics
    readiness_gates["global_sota_donor_contact_propensity_model"] = evaluate_classification_gate(
        contact_test_comparison[contact_winner]
    )

    print(f"\nDonor contact propensity — Winner: {contact_winner}")
    print(f"  Tabular: AUC={contact_tab_metrics['auc']:.4f} F1={contact_tab_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()

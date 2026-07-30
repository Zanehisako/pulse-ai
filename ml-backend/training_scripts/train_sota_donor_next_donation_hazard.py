# Cell 10 — Donor Model A: Next-donation hazard (classification)
# Usage: python train_sota_donor_next_donation_hazard.py

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
    train_neural, neural_predict, classification_metrics, best_f1_threshold,
    encode_ohe, fit_tabular_classifier,
    fit_sota_classification_baselines, choose_winner,
    evaluate_classification_gate, save_bundle,
    apply_donor_proxy_policy, all_model_metrics, readiness_gates,
)


def main():
    print("=" * 70)
    print("Donor Model A: Next-Donation Hazard (6m)")
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
        [c for c in donor_features_cols if c in donor_df.columns], "donor_next_donation_hazard"
    )
    train, valid, test = donor_group_holdout_split(donor_df)
    del donor_df
    gc.collect()
    target = "donated_next_6m"

    neural_model, neural_prep, neural_info = train_neural(train, valid, donor_features_cols, target, task="classification", out_dim=1)
    valid_prob = neural_predict(neural_model, neural_prep, valid, task="classification").reshape(-1)
    thr = best_f1_threshold(valid[target], valid_prob)
    neural_valid_metrics = classification_metrics(valid[target], valid_prob, thr)
    prob = neural_predict(neural_model, neural_prep, test, task="classification").reshape(-1)
    donor_hazard_metrics = classification_metrics(test[target], prob, thr)

    neural_state = {k: v.cpu().clone() for k, v in neural_model.state_dict().items()}
    del neural_model
    flush_gpu()

    tx, vx, tex, cols = encode_ohe(train, valid, test, donor_features_cols)
    tab_name, tab_model, _ = fit_tabular_classifier(tx, train[target].astype(int), vx, valid[target].astype(int))
    tab_valid_prob = tab_model.predict_proba(vx)[:, 1]
    tab_thr = best_f1_threshold(valid[target].astype(int), tab_valid_prob)
    tab_valid_metrics = classification_metrics(valid[target].astype(int), tab_valid_prob, tab_thr)
    tab_prob = tab_model.predict_proba(tex)[:, 1]
    tab_metrics = classification_metrics(test[target], tab_prob, tab_thr)

    sota = fit_sota_classification_baselines(train, valid, test, donor_features_cols, target, "donor_next_donation_hazard")
    valid_comparison = {"neural_ft_transformer": neural_valid_metrics, "tabular_gbdt": tab_valid_metrics, **sota["valid"]}
    test_comparison = {"neural_ft_transformer": donor_hazard_metrics, "tabular_gbdt": tab_metrics, **sota["test"]}
    winner = choose_winner(valid_comparison, "auc", higher_is_better=True)
    metrics = {"task": "donor_next_donation_6m_hazard_proxy", "winner": winner,
               "selection_metric": "validation_auc", "validation_protocol": "donor_group_holdout",
               "validation_comparison": valid_comparison, "test_comparison": test_comparison,
               "sota_baselines": {"availability": sota["availability"], "thresholds": sota.get("thresholds", {})},
               "neural_overall": donor_hazard_metrics, "tabular_overall": tab_metrics, **neural_info}
    readiness_gates["global_sota_donor_next_donation_hazard_model"] = evaluate_classification_gate(test_comparison[winner])
    bundle = {"neural_model_state": neural_state, "neural_preprocessor": neural_prep,
              "tabular_model": tab_model, "tabular_columns": cols, "feature_cols": donor_features_cols,
              "target": target, "winner": winner,
              "optional_sota_models": {k: v for k, v in sota["models"].items() if k != "autogluon"}}
    save_bundle("global_sota_donor_next_donation_hazard_model", bundle, metrics)
    all_model_metrics["global_sota_donor_next_donation_hazard_model"] = metrics

    print(f"\nDonor next-donation hazard — Winner: {winner}")
    print(f"  Neural: AUC={donor_hazard_metrics['auc']:.4f} F1={donor_hazard_metrics['f1']:.4f}")

    del tx, vx, tex, bundle, neural_state, sota, train, valid, test
    flush_gpu()


if __name__ == "__main__":
    main()

# Cell 10b — Model A: Donor time-to-next-donation quantile regression
# Usage: python train_sota_donor_time_to_donation.py

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
    train_neural, neural_predict, quantile_metrics,
    encode_ohe, fit_tabular_regressor,
    fit_sota_regression_baselines, choose_winner,
    evaluate_regression_gate, save_bundle,
    apply_donor_proxy_policy, all_model_metrics, readiness_gates,
)


def main():
    print("=" * 70)
    print("Donor Model E: Time-to-Next-Donation (Quantile Regression)")
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
    donor_reg_cols = [
        "province", "supplier_context", "region_admin", "age", "sex", "language_preference",
        "blood_type", "digital_contactability_score", "center_distance_km", "public_transit_access_score",
        "winter_route_risk_score", "smoker", "bmi", "chronic_condition_flag", "is_rare_type",
        "snapshot_month", "snapshot_is_winter", "years_since_first_donation", "lifetime_donation_count",
        "recency_days", "snapshot_days_until_eligible", "snapshot_frequency_365", "donation_velocity",
        "is_regular_donor", "eligible_to_donate", "readiness_score", "mobility_score",
        "response_readiness_index", "donor_momentum", "operational_outreach_priority"
    ]
    donor_reg_cols = apply_donor_proxy_policy(
        [c for c in donor_reg_cols if c in donor_df.columns], "donor_time_to_donation"
    )
    train, valid, test = donor_group_holdout_split(donor_df)
    del donor_df
    gc.collect()
    target = "days_until_next_donation"
    baseline_val = np.full(len(test), float(train[target].median()))
    baseline_valid = np.full(len(valid), float(train[target].median()))

    neural_model, neural_prep, neural_info = train_neural(train, valid, donor_reg_cols, target, task="quantile", out_dim=3)
    qv = neural_predict(neural_model, neural_prep, valid, task="quantile")
    vq10, vq50, vq90 = np.minimum(qv[:, 0], qv[:, 1]), qv[:, 1], np.maximum(qv[:, 2], qv[:, 1])
    neural_valid_metrics = quantile_metrics(valid[target], vq10, vq50, vq90, baseline_valid)
    q = neural_predict(neural_model, neural_prep, test, task="quantile")
    q10, q50, q90 = np.minimum(q[:, 0], q[:, 1]), q[:, 1], np.maximum(q[:, 2], q[:, 1])
    neural_metrics = quantile_metrics(test[target], q10, q50, q90, baseline_val)

    neural_state = {k: v.cpu().clone() for k, v in neural_model.state_dict().items()}
    del neural_model
    flush_gpu()

    tx, vx, tex, cols = encode_ohe(train, valid, test, donor_reg_cols)
    tab_models, tab_preds, tab_valid_preds = {}, {}, {}
    for quant in [.1, .5, .9]:
        name, model, _ = fit_tabular_regressor(tx, train[target], vx, valid[target], objective="quantile", alpha=quant)
        tab_models[f"q{int(quant*100)}"] = model
        tab_valid_preds[quant] = model.predict(vx)
        tab_preds[quant] = model.predict(tex)
    tab_vq10 = np.minimum.reduce([tab_valid_preds[.1], tab_valid_preds[.5], tab_valid_preds[.9]])
    tab_vq90 = np.maximum.reduce([tab_valid_preds[.1], tab_valid_preds[.5], tab_valid_preds[.9]])
    tab_vq50 = np.clip(tab_valid_preds[.5], tab_vq10, tab_vq90)
    tab_valid_metrics = quantile_metrics(valid[target], tab_vq10, tab_vq50, tab_vq90, baseline_valid)
    tab_q10 = np.minimum.reduce([tab_preds[.1], tab_preds[.5], tab_preds[.9]])
    tab_q90 = np.maximum.reduce([tab_preds[.1], tab_preds[.5], tab_preds[.9]])
    tab_q50 = np.clip(tab_preds[.5], tab_q10, tab_q90)
    tab_metrics = quantile_metrics(test[target], tab_q10, tab_q50, tab_q90, baseline_val)

    sota = fit_sota_regression_baselines(train, valid, test, donor_reg_cols, target, "donor_time_to_donation")
    valid_comparison = {"neural_ft_transformer": neural_valid_metrics, "tabular_gbdt": tab_valid_metrics, **sota["valid"]}
    test_comparison = {"neural_ft_transformer": neural_metrics, "tabular_gbdt": tab_metrics, **sota["test"]}
    winner = choose_winner(valid_comparison, "wape", higher_is_better=False)
    metrics = {"task": "donor_days_until_next_donation", "primary_model": "neural_ft_transformer", "winner": winner,
               "selection_metric": "validation_wape", "validation_protocol": "donor_group_holdout",
               "validation_comparison": valid_comparison, "test_comparison": test_comparison,
               "sota_baselines": {"availability": sota["availability"], "thresholds": sota.get("thresholds", {})},
               "neural_overall": neural_metrics, "tabular_overall": tab_metrics, **neural_info}
    readiness_gates["global_sota_donor_time_to_donation_model"] = evaluate_regression_gate(test_comparison[winner])
    bundle = {"neural_model_state": neural_state, "neural_preprocessor": neural_prep,
              "tabular_models": tab_models, "tabular_columns": cols, "feature_cols": donor_reg_cols,
              "target": target, "winner": winner,
              "optional_sota_models": {k: v for k, v in sota["models"].items() if k != "autogluon"}}
    save_bundle("global_sota_donor_time_to_donation_model", bundle, metrics)
    all_model_metrics["global_sota_donor_time_to_donation_model"] = metrics

    print(f"\nDonor time-to-donation — Winner: {winner}")
    print(f"  Neural: RMSE={neural_metrics['rmse']:.1f} MAE={neural_metrics['mae']:.1f} "
          f"WAPE={neural_metrics['wape']:.4f} R²={neural_metrics['r2']:.4f}")
    print(f"  Tabular: RMSE={tab_metrics['rmse']:.1f} MAE={tab_metrics['mae']:.1f} "
          f"WAPE={tab_metrics['wape']:.4f} R²={tab_metrics['r2']:.4f}")
    print(f"  Baseline WAPE: {neural_metrics['baseline_wape']:.4f}")
    print(f"  Coverage (80% target): {neural_metrics['q10_q90_coverage']:.3f}")

    del tx, vx, tex, bundle, neural_state, tab_models, tab_preds, tab_valid_preds, sota, train, valid, test
    flush_gpu()


if __name__ == "__main__":
    main()

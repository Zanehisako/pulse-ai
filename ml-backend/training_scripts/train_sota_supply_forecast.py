# Cell 7 — Global Model 2: Supply quantile forecast
# Usage: python train_sota_supply_forecast.py

import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from sota_training_utils import (
    CONFIG, OUTPUT_DIR, DATA_DIR, flush_gpu,
    unseen_hospital_split, rolling_baseline,
    train_neural, neural_predict, quantile_metrics, group_metrics_quantile,
    encode_ohe, fit_tabular_regressor,
    fit_sota_regression_baselines, choose_winner,
    evaluate_regression_gate, save_bundle,
    all_model_metrics, readiness_gates,
)


def main():
    print("=" * 70)
    print("Global Model 2: Supply Quantile Forecast")
    print("=" * 70)

    import pandas as pd
    inv_f_path = DATA_DIR / "sota_inventory_features.parquet"
    inv_l_path = DATA_DIR / "sota_inventory_labels.parquet"
    if not (inv_f_path.exists() and inv_l_path.exists()):
        print(f"Data not found at {DATA_DIR}. Run generate_sota_dataset.py first.")
        return
    inv_features = pd.read_parquet(inv_f_path)
    inv_labels = pd.read_parquet(inv_l_path)

    forecast_features = [
        "hospital_id", "hospital_name", "region_admin", "city", "supplier_context", "province",
        "blood_type", "component_type", "hospital_size_tier", "month", "dow", "weekend", "holiday", "flu_season_flag",
        "major_event", "supply_disruption", "donor_shortfall", "shelf_life_days", "current_inventory", "units_used",
        "units_collected", "wastage", "incoming_supply_scheduled_7d", "lead_time_days", "route_disruption_score",
        "trauma_cases", "scheduled_surgeries", "component_criticality_score", "inventory_7d_mean", "usage_7d_mean",
        "wastage_7d_mean", "collection_7d_mean", "inventory_30d_mean", "usage_30d_mean", "collection_30d_mean", "stockout_count_30d"
    ]

    target = "units_collected_next_7d"
    inventory_df = inv_features.merge(
        inv_labels[["hospital_id", "blood_type", "component_type", "event_timestamp", target]],
        on=["hospital_id", "blood_type", "component_type", "event_timestamp"], how="inner"
    )
    train, valid, test = unseen_hospital_split(inventory_df)
    baseline_valid = rolling_baseline(train, valid, target)
    baseline = rolling_baseline(train, test, target)

    neural_model, neural_prep, neural_info = train_neural(train, valid, forecast_features, target, task="quantile", out_dim=3)
    qv = neural_predict(neural_model, neural_prep, valid, task="quantile")
    vq10, vq50, vq90 = np.minimum(qv[:, 0], qv[:, 1]), qv[:, 1], np.maximum(qv[:, 2], qv[:, 1])
    neural_valid_metrics = quantile_metrics(valid[target], vq10, vq50, vq90, baseline_valid)
    q = neural_predict(neural_model, neural_prep, test, task="quantile")
    q10, q50, q90 = np.minimum(q[:, 0], q[:, 1]), q[:, 1], np.maximum(q[:, 2], q[:, 1])
    neural_metrics = quantile_metrics(test[target], q10, q50, q90, baseline)
    neural_by_group = group_metrics_quantile(test, test[target], q10, q50, q90, baseline)

    neural_state = {k: v.cpu().clone() for k, v in neural_model.state_dict().items()}
    del neural_model
    flush_gpu()

    tx, vx, tex, cols = encode_ohe(train, valid, test, forecast_features)
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
    tab_metrics = quantile_metrics(test[target], tab_q10, tab_q50, tab_q90, baseline)
    tab_by_group = group_metrics_quantile(test, test[target], tab_q10, tab_q50, tab_q90, baseline)

    sota = fit_sota_regression_baselines(train, valid, test, forecast_features, target, "supply_quantile_forecast")
    valid_comparison = {"neural_ft_transformer": neural_valid_metrics, "tabular_gbdt": tab_valid_metrics, **sota["valid"]}
    test_comparison = {"neural_ft_transformer": neural_metrics, "tabular_gbdt": tab_metrics, **sota["test"]}
    winner = choose_winner(valid_comparison, "wape", higher_is_better=False)
    metrics = {"task": "supply_quantile_forecast", "primary_model": "neural_ft_transformer", "winner": winner,
               "selection_metric": "validation_wape", "validation_comparison": valid_comparison, "test_comparison": test_comparison,
               "sota_baselines": {"availability": sota["availability"], "thresholds": sota.get("thresholds", {})},
               "neural_overall": neural_metrics, "tabular_overall": tab_metrics,
               "neural_by_product_group": neural_by_group, "tabular_by_product_group": tab_by_group, **neural_info}
    readiness_gates["global_sota_supply_quantile_forecast_model"] = evaluate_regression_gate(
        test_comparison[winner],
        tab_by_group if winner == "tabular_gbdt" else neural_by_group if winner == "neural_ft_transformer" else None
    )
    bundle = {"neural_model_state": neural_state, "neural_preprocessor": neural_prep,
              "tabular_models": tab_models, "tabular_columns": cols, "feature_cols": forecast_features,
              "target": target, "winner": winner,
              "optional_sota_models": {k: v for k, v in sota["models"].items() if k != "autogluon"}}
    save_bundle("global_sota_supply_quantile_forecast_model", bundle, metrics)
    all_model_metrics["global_sota_supply_quantile_forecast_model"] = metrics

    print(f"\nSupply forecast — Winner: {winner}")
    print(f"  Neural: WAPE={neural_metrics['wape']:.4f} RMSE={neural_metrics['rmse']:.2f} R²={neural_metrics['r2']:.4f}")

    del tx, vx, tex, bundle, neural_state, tab_models, tab_preds, tab_valid_preds, sota, train, valid, test, inventory_df
    flush_gpu()


if __name__ == "__main__":
    main()

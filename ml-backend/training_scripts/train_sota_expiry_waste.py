# Cell 8 — Global Model 3: Expiry/waste forecast
# Usage: python train_sota_expiry_waste.py

import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from sota_training_utils import (
    CONFIG, OUTPUT_DIR, DATA_DIR, flush_gpu,
    unseen_hospital_split, rolling_baseline,
    train_neural, neural_predict, regression_metrics, group_metrics_reg,
    encode_ohe, fit_tabular_regressor,
    fit_sota_regression_baselines, choose_winner,
    evaluate_regression_gate, save_bundle,
    leakage_audit, all_model_metrics, readiness_gates,
)


def main():
    print("=" * 70)
    print("Global Model 3: Expiry/Waste Forecast")
    print("=" * 70)

    import pandas as pd
    inv_f_path = DATA_DIR / "sota_inventory_features.parquet"
    inv_l_path = DATA_DIR / "sota_inventory_labels.parquet"
    if not (inv_f_path.exists() and inv_l_path.exists()):
        print(f"Data not found at {DATA_DIR}. Run generate_sota_dataset.py first.")
        return
    inv_features = pd.read_parquet(inv_f_path)
    inv_labels = pd.read_parquet(inv_l_path)

    target = "wastage_next_7d"
    waste_features = [
        "hospital_id", "hospital_name", "region_admin", "city", "supplier_context", "province",
        "blood_type", "component_type", "hospital_size_tier", "month", "dow", "weekend", "holiday",
        "flu_season_flag", "major_event", "supply_disruption", "donor_shortfall", "shelf_life_days",
        "current_inventory", "wastage", "units_used", "units_collected", "route_disruption_score",
        "inventory_7d_mean", "wastage_7d_mean", "inventory_30d_mean", "wastage_30d_mean",
        "collection_adjusted_runway_days"
    ]
    inventory_df = inv_features.merge(
        inv_labels[["hospital_id", "blood_type", "component_type", "event_timestamp", target]],
        on=["hospital_id", "blood_type", "component_type", "event_timestamp"], how="inner"
    )
    leakage_audit(waste_features, ["*next*", "*future*", "*label*", "*target*"], "global_sota_waste")
    train, valid, test = unseen_hospital_split(inventory_df)
    baseline_valid = rolling_baseline(train, valid, target)
    baseline = rolling_baseline(train, test, target)

    neural_model, neural_prep, neural_info = train_neural(train, valid, waste_features, target, task="regression", out_dim=1)
    valid_pred = np.clip(neural_predict(neural_model, neural_prep, valid, task="regression").reshape(-1), 0, None)
    neural_valid_metrics = regression_metrics(valid[target], valid_pred, baseline_valid)
    pred = np.clip(neural_predict(neural_model, neural_prep, test, task="regression").reshape(-1), 0, None)
    neural_metrics = regression_metrics(test[target], pred, baseline)
    neural_by_group = group_metrics_reg(test, test[target], pred, baseline)

    neural_state = {k: v.cpu().clone() for k, v in neural_model.state_dict().items()}
    del neural_model
    flush_gpu()

    tx, vx, tex, cols = encode_ohe(train, valid, test, waste_features)
    tab_name, tab_model, _ = fit_tabular_regressor(tx, train[target], vx, valid[target], objective="regression")
    tab_valid_pred = np.clip(tab_model.predict(vx), 0, None)
    tab_valid_metrics = regression_metrics(valid[target], tab_valid_pred, baseline_valid)
    tab_pred = np.clip(tab_model.predict(tex), 0, None)
    tab_metrics = regression_metrics(test[target], tab_pred, baseline)
    tab_by_group = group_metrics_reg(test, test[target], tab_pred, baseline)

    sota = fit_sota_regression_baselines(train, valid, test, waste_features, target, "expiry_waste_forecast")
    valid_comparison = {"neural_ft_transformer": neural_valid_metrics, "tabular_gbdt": tab_valid_metrics, **sota["valid"]}
    test_comparison = {"neural_ft_transformer": neural_metrics, "tabular_gbdt": tab_metrics, **sota["test"]}
    winner = choose_winner(valid_comparison, "wape", higher_is_better=False)
    metrics = {"task": "expiry_waste_forecast", "primary_model": "neural_ft_transformer", "winner": winner,
               "selection_metric": "validation_wape", "validation_comparison": valid_comparison, "test_comparison": test_comparison,
               "sota_baselines": {"availability": sota["availability"], "thresholds": sota.get("thresholds", {})},
               "neural_overall": neural_metrics, "tabular_overall": tab_metrics,
               "neural_by_product_group": neural_by_group, "tabular_by_product_group": tab_by_group, **neural_info}
    readiness_gates["global_sota_expiry_waste_model"] = evaluate_regression_gate(
        test_comparison[winner],
        tab_by_group if winner == "tabular_gbdt" else neural_by_group if winner == "neural_ft_transformer" else None
    )
    bundle = {"neural_model_state": neural_state, "neural_preprocessor": neural_prep,
              "tabular_model": tab_model, "tabular_columns": cols, "feature_cols": waste_features,
              "target": target, "winner": winner,
              "optional_sota_models": {k: v for k, v in sota["models"].items() if k != "autogluon"}}
    save_bundle("global_sota_expiry_waste_model", bundle, metrics)
    all_model_metrics["global_sota_expiry_waste_model"] = metrics

    print(f"\nExpiry/Waste forecast — Winner: {winner}")
    print(f"  Neural: WAPE={neural_metrics['wape']:.4f} RMSE={neural_metrics['rmse']:.2f} R²={neural_metrics['r2']:.4f}")

    del tx, vx, tex, bundle, neural_state, sota, train, valid, test, inventory_df
    flush_gpu()


if __name__ == "__main__":
    main()

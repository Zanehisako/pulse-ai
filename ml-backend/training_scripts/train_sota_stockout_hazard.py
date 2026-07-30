# Cell 9 — Global Model 4: Stockout hazard (classification)
# Usage: python train_sota_stockout_hazard.py

import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from sota_training_utils import (
    CONFIG, OUTPUT_DIR, DATA_DIR, flush_gpu,
    unseen_hospital_split,
    train_neural, neural_predict, classification_metrics, best_f1_threshold, group_metrics_cls,
    encode_ohe, fit_tabular_classifier,
    fit_sota_classification_baselines, choose_winner,
    evaluate_classification_gate, save_bundle,
    all_model_metrics, readiness_gates,
)


def main():
    print("=" * 70)
    print("Global Model 4: Stockout Hazard (Classification)")
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

    stockout_df = inv_features.merge(
        inv_labels[["hospital_id", "blood_type", "component_type", "event_timestamp", "stockout_within_0_7d"]],
        on=["hospital_id", "blood_type", "component_type", "event_timestamp"], how="inner"
    )
    stockout_features = forecast_features + ["safety_threshold", "stockout_count_7d", "stockout_count_14d"]
    stockout_features = list(dict.fromkeys([c for c in stockout_features if c in stockout_df.columns]))
    target = "stockout_within_0_7d"
    train, valid, test = unseen_hospital_split(stockout_df)

    neural_model, neural_prep, neural_info = train_neural(train, valid, stockout_features, target, task="classification", out_dim=1)
    valid_prob = neural_predict(neural_model, neural_prep, valid, task="classification").reshape(-1)
    thr = best_f1_threshold(valid[target], valid_prob)
    neural_valid_metrics = classification_metrics(valid[target], valid_prob, thr)
    prob = neural_predict(neural_model, neural_prep, test, task="classification").reshape(-1)
    neural_metrics = classification_metrics(test[target], prob, thr)
    neural_by_group = group_metrics_cls(test, test[target], prob, thr)

    neural_state = {k: v.cpu().clone() for k, v in neural_model.state_dict().items()}
    del neural_model
    flush_gpu()

    tx, vx, tex, cols = encode_ohe(train, valid, test, stockout_features)
    tab_name, tab_model, _ = fit_tabular_classifier(tx, train[target].astype(int), vx, valid[target].astype(int))
    tab_valid_prob = tab_model.predict_proba(vx)[:, 1]
    tab_thr = best_f1_threshold(valid[target].astype(int), tab_valid_prob)
    tab_valid_metrics = classification_metrics(valid[target].astype(int), tab_valid_prob, tab_thr)
    tab_prob = tab_model.predict_proba(tex)[:, 1]
    tab_metrics = classification_metrics(test[target], tab_prob, tab_thr)
    tab_by_group = group_metrics_cls(test, test[target], tab_prob, tab_thr)

    sota = fit_sota_classification_baselines(train, valid, test, stockout_features, target, "stockout_hazard")
    valid_comparison = {"neural_ft_transformer": neural_valid_metrics, "tabular_gbdt": tab_valid_metrics, **sota["valid"]}
    test_comparison = {"neural_ft_transformer": neural_metrics, "tabular_gbdt": tab_metrics, **sota["test"]}
    winner = choose_winner(valid_comparison, "auc", higher_is_better=True)
    metrics = {"task": "stockout_hazard_0_7d", "primary_model": "neural_ft_transformer", "winner": winner,
               "selection_metric": "validation_auc", "validation_comparison": valid_comparison, "test_comparison": test_comparison,
               "sota_baselines": {"availability": sota["availability"], "thresholds": sota.get("thresholds", {})},
               "neural_overall": neural_metrics, "tabular_overall": tab_metrics,
               "neural_by_product_group": neural_by_group, "tabular_by_product_group": tab_by_group, **neural_info}
    readiness_gates["global_sota_stockout_hazard_model"] = evaluate_classification_gate(
        test_comparison[winner],
        tab_by_group if winner == "tabular_gbdt" else neural_by_group if winner == "neural_ft_transformer" else None
    )
    bundle = {"neural_model_state": neural_state, "neural_preprocessor": neural_prep,
              "tabular_model": tab_model, "tabular_columns": cols, "feature_cols": stockout_features,
              "target": target, "winner": winner,
              "optional_sota_models": {k: v for k, v in sota["models"].items() if k != "autogluon"}}
    save_bundle("global_sota_stockout_hazard_model", bundle, metrics)
    all_model_metrics["global_sota_stockout_hazard_model"] = metrics

    print(f"\nStockout hazard — Winner: {winner}")
    print(f"  Neural: AUC={neural_metrics['auc']:.4f} F1={neural_metrics['f1']:.4f} Brier={neural_metrics['brier']:.4f}")

    del tx, vx, tex, bundle, neural_state, sota, train, valid, test, stockout_df
    flush_gpu()


if __name__ == "__main__":
    main()

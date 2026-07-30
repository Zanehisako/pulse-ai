# Cell 11 — Hybrid model registry and inventory risk simulator
# Usage: python train_sota_inventory_risk_simulator.py

import sys
from pathlib import Path

import joblib

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from sota_training_utils import (
    CONFIG, OUTPUT_DIR, save_json, save_bundle,
    all_model_metrics, readiness_gates,
)


def select_by_group(metrics, metric, higher=False):
    n = metrics.get("neural_by_product_group", {})
    t = metrics.get("tabular_by_product_group", {})
    out = {}
    for k in sorted(set(n) | set(t)):
        if k not in n:
            out[k] = {"selected": "tabular_gbdt"}
            continue
        if k not in t:
            out[k] = {"selected": "neural_ft_transformer"}
            continue
        nv, tv = n[k].get(metric), t[k].get(metric)
        sel = "neural_ft_transformer" if (nv >= tv if higher else nv <= tv) else "tabular_gbdt"
        out[k] = {"selected": sel, "neural_metric": float(nv), "tabular_metric": float(tv), "metric": metric}
    return out


def main():
    print("=" * 70)
    print("Hybrid Model Registry & Inventory Risk Simulator")
    print("=" * 70)

    # Reload all metrics if not in memory
    metrics_path = OUTPUT_DIR / "all_model_metrics.json"
    if not all_model_metrics and metrics_path.exists():
        import json
        with open(metrics_path) as f:
            loaded = json.load(f)
            all_model_metrics.update(loaded)

    required_keys = [
        "global_sota_demand_quantile_forecast_model",
        "global_sota_supply_quantile_forecast_model",
        "global_sota_expiry_waste_model",
        "global_sota_stockout_hazard_model",
        "global_sota_donor_next_donation_hazard_model",
        "global_sota_donor_time_to_donation_model",
    ]
    missing = [k for k in required_keys if k not in all_model_metrics]
    if missing:
        print(f"Warning: Missing model metrics: {missing}")
        print("Run the training scripts first.")

    registry = {
        "demand": select_by_group(all_model_metrics.get("global_sota_demand_quantile_forecast_model", {}), "wape", higher=False),
        "supply": select_by_group(all_model_metrics.get("global_sota_supply_quantile_forecast_model", {}), "wape", higher=False),
        "waste": select_by_group(all_model_metrics.get("global_sota_expiry_waste_model", {}), "wape", higher=False),
        "stockout": select_by_group(all_model_metrics.get("global_sota_stockout_hazard_model", {}), "auc", higher=True),
        "donor": {
            "next_donation": all_model_metrics.get("global_sota_donor_next_donation_hazard_model", {}).get("winner", "unknown"),
            "time_to_donation": all_model_metrics.get("global_sota_donor_time_to_donation_model", {}).get("winner", "unknown"),
            "contact_response": "proxy_global_model",
            "contact_propensity": "proxy_global_model",
            "priority_policy": "policy_model",
        }
    }
    save_json(registry, OUTPUT_DIR / "sota_hybrid_selection_registry.json")
    joblib.dump(registry, OUTPUT_DIR / "sota_hybrid_selection_registry.joblib")

    simulator = {
        "name": "global_sota_inventory_risk_simulator",
        "type": "hybrid_simulator",
        "inputs": ["global demand q50", "global supply q50", "global waste prediction", "global stockout hazard probability"],
        "risk_rule": "hybrid_risk = max(mechanistic_inventory_risk, stockout_hazard_probability)",
        "horizons_days": [7, 30, 90, 180],
        "registry": "sota_hybrid_selection_registry.json",
    }
    joblib.dump(simulator, OUTPUT_DIR / "global_sota_inventory_risk_simulator.joblib")
    save_json({"artifact_type": "simulator"}, OUTPUT_DIR / "global_sota_inventory_risk_simulator_metrics.json")
    all_model_metrics["global_sota_inventory_risk_simulator"] = {"artifact_type": "simulator"}
    readiness_gates["global_sota_inventory_risk_simulator"] = {
        "structural_pass": True,
        "operational_ready": False,
        "reason": "requires real operational data for deployment validation",
    }

    save_json(all_model_metrics, OUTPUT_DIR / "all_model_metrics.json")
    print("\nHybrid registry and simulator saved.")
    for k, v in all_model_metrics.items():
        winner = v.get("winner", "saved") if isinstance(v, dict) and "winner" in v else "saved"
        print(f"  {k}: {winner}")


if __name__ == "__main__":
    main()

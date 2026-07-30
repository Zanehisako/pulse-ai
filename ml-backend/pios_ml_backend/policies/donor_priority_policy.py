"""
Donor priority policy pyfunc model.

Takes donor features and computes an expected-utility priority score
using configurable weights from donor_priority_policy_model.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

_ML_BACKEND = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _ML_BACKEND / "config" / "donor_priority_policy_model.json"


def _load_config() -> dict:
    return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))


class DonorPriorityPolicyModel(mlflow.pyfunc.PythonModel):
    def __init__(self) -> None:
        super().__init__()
        config = _load_config()
        formula = config.get("priority_formula", {})
        self.thresholds = config.get("action_thresholds", {})
        self.policy_inputs = config.get("policy_inputs", {}).get("donor_features", [])

        self.w_usable = float(formula.get("expected_usable_units_weight", 0.40))
        self.w_urgency = float(formula.get("urgency_weight", 0.25))
        self.w_compat = float(formula.get("compatibility_weight", 0.15))
        self.w_location = float(formula.get("location_feasibility_weight", 0.10))
        self.w_response = float(formula.get("response_likelihood_weight", 0.10))
        self.contact_cost = float(formula.get("contact_cost", 0.05))
        self.fatigue_penalty = float(formula.get("fatigue_penalty", 0.02))
        self.normalize = bool(formula.get("display_score_normalization", True))

    def _score_row(self, row: dict) -> float:
        eligible = float(row.get("eligible_to_donate", 0) or 0)
        if eligible <= 0:
            return 0.0

        is_regular = float(row.get("is_regular_donor", 0) or 0)
        is_rare = float(row.get("is_rare_type", 0) or 0)
        compatible = 0.5 + 0.5 * is_rare if is_rare > 0 else 0.5

        distance = float(row.get("center_distance_km", 50) or 50)
        travel = float(row.get("travel_time_min", 30) or 30)
        location_feasible = 1.0 / (1.0 + distance / 10.0 + travel / 30.0)

        response = max(0.0, min(1.0, float(row.get("response_readiness_index", 0) or 0) / 10.0 + 0.5))
        momentum = max(0.0, min(1.0, float(row.get("donor_momentum", 0) or 0) / 4.0))
        outreach = max(0.0, min(1.0, float(row.get("operational_outreach_priority", 0.5) or 0.5)))

        usable = 0.3 + 0.4 * is_regular + 0.3 * momentum
        urgency = 0.5 + 0.5 * outreach

        fatigue = self.fatigue_penalty if is_regular > 0 else 0.0

        score = (
            self.w_usable * usable
            + self.w_urgency * urgency
            + self.w_compat * compatible
            + self.w_location * location_feasible
            + self.w_response * response
            - self.contact_cost
            - fatigue
        )
        return max(0.0, min(1.0, score))

    def _action(self, score: float) -> str:
        thresholds = sorted(
            [(float(v), k) for k, v in self.thresholds.items()],
            key=lambda x: x[0],
            reverse=True,
        )
        for threshold, action in thresholds:
            if score >= threshold:
                return action.replace("_", " ")
        return "do_not_contact_low_score"

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            frame = model_input.copy()
        elif isinstance(model_input, (list, dict)):
            frame = pd.DataFrame([model_input] if isinstance(model_input, dict) else model_input)
        else:
            frame = pd.DataFrame(model_input)

        results = []
        for _, row in frame.iterrows():
            row_dict = row.to_dict()
            score = self._score_row(row_dict)
            action = self._action(score)
            display_score = round(score * 100) if self.normalize else round(score, 4)
            results.append({
                "donor_priority_score": round(score, 4),
                "display_donor_score": display_score,
                "recommended_action": action,
                "action_reason": self._reason(action, row_dict),
                "prediction": round(score, 4),
            })

        return pd.DataFrame(results)

    def _reason(self, action: str, row: dict) -> str:
        if action == "do not contact low score":
            return "Donor priority below minimum threshold"
        if action == "urgent contact":
            return f"High-priority donor (rare_type={row.get('is_rare_type', 0)}, momentum={row.get('donor_momentum', 0):.1f})"
        if action == "high priority contact":
            return "High expected donation yield"
        if action == "standard contact":
            return "Standard outreach priority"
        if action == "low priority":
            return "Low expected return on contact effort"
        return "Priority assessment based on donor features"

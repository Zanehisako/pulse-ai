from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def sample_from_quantiles(
    rng: np.random.Generator,
    n_paths: int,
    forecast: float,
    cv: float,
    quantile_rng: float = 0.0,
) -> np.ndarray:
    if forecast <= 0:
        return np.zeros(n_paths, dtype=float)
    sigma = max(cv, 0.001)
    mu = np.log(max(forecast, 1e-6)) - 0.5 * sigma**2
    if quantile_rng > 0:
        sigma_lower = max(sigma * (1.0 - quantile_rng), 0.001)
        sigma_upper = sigma * (1.0 + quantile_rng)
        sigmas = rng.uniform(sigma_lower, sigma_upper, size=n_paths)
        samples = np.array(
            [rng.lognormal(mu, s) for s in sigmas], dtype=float
        )
    else:
        samples = rng.lognormal(mu, sigma, size=n_paths)
    return np.clip(samples, 0.0, None)


class InventorySimulator:
    def __init__(
        self,
        n_paths: int = 1000,
        horizons_days: list[int] | None = None,
        default_cv: float = 0.15,
        safety_threshold: float = 1.0,
        seed: int | None = None,
    ):
        self.n_paths = n_paths
        self.horizons_days = horizons_days or [7, 30, 90, 180]
        self.default_cv = default_cv
        self.safety_threshold = safety_threshold
        self._rng = np.random.default_rng(seed)

    def run(
        self,
        current_stock: float,
        demand_forecast: float,
        supply_forecast: float,
        waste_forecast: float,
        demand_cv: float | None = None,
        supply_cv: float | None = None,
        waste_cv: float | None = None,
        safety_threshold: float | None = None,
    ) -> dict[str, Any]:
        cv_d = demand_cv if demand_cv is not None else self.default_cv
        cv_s = supply_cv if supply_cv is not None else self.default_cv
        cv_w = waste_cv if waste_cv is not None else self.default_cv

        max_horizon = max(self.horizons_days)
        threshold = self.safety_threshold if safety_threshold is None else float(safety_threshold)
        stock_paths = np.full(self.n_paths, current_stock, dtype=float)
        first_stockout_day = np.full(self.n_paths, -1, dtype=int)

        for day in range(1, max_horizon + 1):
            demand_samples = sample_from_quantiles(
                self._rng, self.n_paths, demand_forecast, cv_d
            )
            supply_samples = sample_from_quantiles(
                self._rng, self.n_paths, supply_forecast, cv_s
            )
            waste_samples = sample_from_quantiles(
                self._rng, self.n_paths, waste_forecast, cv_w
            )
            stock_paths = stock_paths + supply_samples - demand_samples - waste_samples
            newly_stocked_out = (stock_paths <= threshold) & (
                first_stockout_day == -1
            )
            first_stockout_day[newly_stocked_out] = day

        result: dict[str, Any] = {}
        stocked_out = first_stockout_day != -1
        for h in self.horizons_days:
            result[f"p_stockout_0_{h}d"] = float(
                np.mean(stocked_out & (first_stockout_day <= h))
            )
        result["p_stockout_0_maxd"] = float(np.mean(stocked_out))

        stockout_days = first_stockout_day[stocked_out]
        if len(stockout_days) > 0:
            restricted_days = np.where(stocked_out, first_stockout_day, max_horizon + 1)
            result["expected_days_until_stockout"] = float(np.mean(restricted_days))
        else:
            result["expected_days_until_stockout"] = float(max_horizon + 1)

        for label, quantile in (("p10", 0.10), ("p50", 0.50), ("p90", 0.90)):
            key = f"{label}_days_until_stockout"
            if result["p_stockout_0_maxd"] >= quantile and len(stockout_days) > 0:
                result[key] = float(np.percentile(stockout_days, quantile * 100.0))
            else:
                result[key] = None

        result["median_status"] = (
            "reached" if result["p50_days_until_stockout"] is not None else "not_reached"
        )
        return result


def run_monte_carlo_paths(
    current_stock: np.ndarray,
    demand_forecast: np.ndarray,
    supply_forecast: np.ndarray,
    waste_forecast: np.ndarray,
    n_paths: int = 1000,
    horizons_days: list[int] | None = None,
    demand_cv: float = 0.15,
    supply_cv: float = 0.20,
    waste_cv: float = 0.15,
    safety_threshold: float | np.ndarray = 1.0,
    seed: int | None = None,
) -> pd.DataFrame:
    horizons_days = horizons_days or [7, 30, 90, 180]
    sim = InventorySimulator(
        n_paths=n_paths,
        horizons_days=horizons_days,
        default_cv=demand_cv,
        safety_threshold=safety_threshold,
        seed=seed,
    )
    rows: list[dict[str, Any]] = []
    n_rows = len(current_stock)
    if np.isscalar(safety_threshold):
        thresholds = np.full(n_rows, float(safety_threshold), dtype=float)
    else:
        thresholds = np.asarray(safety_threshold, dtype=float)
    for i in range(n_rows):
        result = sim.run(
            current_stock=float(current_stock[i]),
            demand_forecast=float(demand_forecast[i]),
            supply_forecast=float(supply_forecast[i]),
            waste_forecast=float(waste_forecast[i]),
            demand_cv=demand_cv,
            supply_cv=supply_cv,
            waste_cv=waste_cv,
            safety_threshold=float(thresholds[i]),
        )
        rows.append(result)
    return pd.DataFrame(rows)

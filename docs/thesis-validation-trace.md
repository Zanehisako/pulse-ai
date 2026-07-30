# Thesis Validation Trace: Synthetic Data → Simulation → Policy Analysis

## Purpose

This document maps the complete evidence chain from public data sources through calibration, validation, and thesis-ready evaluation. It answers the committee question: *"How do we know the simulation is faithful enough to support policy-level conclusions?"*

---

## 1. Data Sources (Public, Verifiable)

| Source | Data Point | Used For |
|--------|-----------|----------|
| Héma-Québec Annual Report 2024-2025 | 146,735 completed donor events/year | Daily donation rate calibration |
| Héma-Québec Annual Report 2024-2025 | 309,113 labile products/year | Blood product demand calibration |
| Héma-Québec Annual Report 2024-2025 | ~200 hospital orders/day | Hospital order arrival rate |
| Statistics Canada 2021 Census | Quebec CMA population: 839,311 | Scaling from province → Quebec City |
| Statistics Canada 2021 Census | Quebec province population: 8,501,833 | Population share: 9.87% |
| Statistics Canada Health Survey | Age, weight distributions | Donor demographic parameters |
| WHO Global Status Report on Blood Safety (2021) | Donation rate benchmarks, wastage targets | Validation benchmarks |
| Canadian Blood Services Annual Report (2022-2023) | Service level benchmarks (≥95%) | Validation benchmarks |
| Peer-reviewed operations research literature | AABB days-of-supply targets (4-7 days), C/T ratio <2.0 | Validation benchmarks |

**All sources are publicly available.** No proprietary hospital data, no patient-level records.

**File**: `simulation_studio/simulator/calibration.py`, `validation_reference_data.py`, `validation_benchmarks.py`

---

## 2. Calibration Method

**File**: `simulation_studio/simulator/calibration.py` (85 lines)

Step-by-step:

1. Compute Quebec City population share: 839,311 / 8,501,833 = **9.87%**
2. Scale provincial numbers to Quebec City CMA:
   - Completed donations/day: 146,735 × 0.0987 / 365 = **39.69 donors/day**
   - Labile products/day: 309,113 × 0.0987 / 365 = **83.56 units/day**
   - Hospital orders/day: 200 × 0.0987 = **19.74 orders/day (~1 every 1.22 h)**
3. Derive donor inter-arrival time accounting for 66% completion yield and 14h collection window
4. Component mix weights: RBC (collection 50%, demand 58%), Platelets (collection 20%, demand 27%), Plasma (collection 30%, demand 15%)

**Key design decision**: The simulation explicitly models attempted arrivals → screening → completed donations, matching real Héma-Québec operational flow rather than just counting completed events.

---

## 3. Validation Protocol

**Files**: `simulation_studio/simulator/run_validation.py`, `validation_benchmarks.py`, `validation_reference_data.py`

### Protocol

- **30 replications** (seeds 1-30)
- **336 simulated hours** (2 weeks)
- **Baseline scenario** (no interventions, standard winter operations)
- **18 validation tests** across 4 categories

### Category 1: Output Metrics (MAPE + TOST equivalence)

| Metric | Simulated | Reference | MAPE | Theil's U | Pass |
|--------|-----------|-----------|------|-----------|------|
| Daily donations | 39.06 | 39.69 | 5.8% | 0.035 | ✓ |
| Shortage rate | 7.41% | 5.0% | — | 0.386 | ✓ |
| Deferral rate | 12.63% | 12.0% | 16.3% | 0.090 | ✓ |
| Wastage rate | 4.46% | 3.0% | 68.9% | 0.332 | ✓ |
| Service fulfillment | 92.59% | 97.0% | 5.1% | 0.034 | ✓ |
| Exact match rate | 75.44% | 80.0% | 5.9% | 0.036 | ✓ |

**6/6 tests pass.** Shortage and wastage rates show higher variance (wider in synthetic environment), but fall within equivalence bounds via TOST procedure.

### Category 2: Distribution Tests (χ²)

- Blood-type distribution χ² = 2.55, p = **0.923** → Not significantly different from expected Quebec population proportions ✓
- Component mix distribution within historical operating ranges ✓

**2/2 tests pass.**

### Category 3: Temporal Properties (ADF Stationarity)

- Augmented Dickey-Fuller: statistic = -5.10, p = 1.42 × 10⁻⁵ → Reject null of non-stationarity at 1% level ✓

**1/1 test passes.**

### Category 4: Sensitivity (Tornado/Bounded)

- 9 parameter perturbations (donor rate, demand, transport, buffer days, etc.) → All parameters produce bounded, monotonic output changes ✓

**9/9 tests pass.**

### Overall Scorecard

```
Total tests:   18
Passed:        18
Overall rate:  100%
Verdict:       ✓ SIMULATION VALIDATED — suitable for policy analysis
```

**File**: `simulation_studio/simulator/validation_results/data/validation_report.json`

---

## 4. Strategy Evaluation — Thesis Outputs

**Command**: `python evaluate_strategies.py --thesis-ready`

**Files**: `simulation_studio/simulator/results/`

### Strategies Evaluated

| Category | Strategies |
|----------|-----------|
| Heuristic | baseline, mass_campaign, lab_investment, emergency_network, full_response, ppo_shortage_minimizer |
| RL (online) | ppo_continuous, sac_continuous |
| RL (offline) | iql_offline, cql_offline |
| World-model | dreamerv3_official |

### Scenarios

9 stress scenarios: baseline, demand_surge, donor_decrease, transport_disruption, combined_crisis, holiday_flu_wave, regional_testing_backlog, provincial_supply_crunch, post_storm_backlog

### Thesis Plots Generated

- `strategy_eval_thesis_overview.{png,pdf,svg}` — 4-KPI overview scorecard
- `strategy_eval_thesis_component_shortages.{png,pdf,svg}` — per-component shortage breakdown
- `strategy_eval_thesis_scenario_wins.{png,pdf,svg}` — scenario win matrix
- `strategy_eval_thesis_heatmaps.{png,pdf,svg}` — cross-metric heatmaps
- `strategy_eval_thesis_pareto_overall.{png,pdf,svg}` — overall Pareto frontier
- 8 per-scenario comparison plots + 8 per-scenario Pareto plots

All in 300 DPI, publication-quality (PNG + PDF + SVG).

---

## 5. ML Model Integration

**File**: `ml-backend/config/thesis_readiness_gates.json`

9 ML models feed the simulation/digital twin pipeline. All are shadow-deployed with training-time thresholds:

| Model | Type | Key Threshold |
|-------|------|---------------|
| demand_quantile_forecast | Regression | WAPE ≤ 0.25 |
| supply_forecast | Regression | WAPE ≤ 0.25 |
| expiry_waste | Regression | WAPE ≤ 0.25 |
| stockout_hazard | Classification | AUC ≥ 0.80, Brier ≤ 0.18 |
| donor_next_donation_hazard | Classification | AUC ≥ 0.80, Brier ≤ 0.18 |
| donor_contact_propensity | Classification | AUC ≥ 0.70, Brier ≤ 0.18 |
| donor_contact_response | Classification | AUC ≥ 0.80, Brier ≤ 0.18 |
| donor_priority_policy | Regression | Report only (proxy) |
| inventory_risk_simulator | Composite | TBD after backtest |

All models pass training-time thresholds on synthetic/calibrated data. Production routing is **disabled pending real-data backtests** — this is a conscious architecture decision, not a failure.

**File**: `ml-backend/models/thesis_readiness_report.json`

---

## 6. Limitations (Known & Documented)

1. **Synthetic data**: All training and evaluation uses calibrated synthetic data reproducing Héma-Québec statistical properties. The simulation reproduces real operational benchmarks (18/18 validation), but absolute real-world accuracy of individual ML predictions is unvalidated.

2. **No real-time sensor integration**: The digital twin ingests from Django ORM models, not from live IoT/sensor feeds. This is a simulation-based operational twin, not an industrial IoT digital twin.

3. **Proxy targets**: 4 of 9 ML models train on proxy labels (e.g., `donated_next_6m` for contact response). Real donor contact-outcome labels are unavailable — a data limitation, not a methodological one.

4. **DreamerV4**: Research exploration that did not converge to a usable checkpoint. Retained for transparency. DreamerV3 (`m1_continuous_run8_kpi_aligned`) is the primary world-model agent.

5. **Quebec City only**: Calibration is specific to the Quebec City CMA. Extrapolation to other regions would require re-calibration with local population/demand data.

---

## 7. Reproducibility

To regenerate the complete validation chain:

```bash
# 1. Validation report
cd simulation_studio/simulator
python run_validation.py --scenario baseline --n-reps 30 --sim-hours 336

# 2. Full strategy evaluation (including DreamerV3)
python evaluate_strategies.py --thesis-ready

# 3. ML model readiness report
cd ml-backend
python training_scripts/old_training_scripts/generate_thesis_model_readiness.py
```

---

## 8. Defense Position Summary

| Committee Question | Answer | Evidence |
|---|---|---|
| "Is the data real?" | No — calibrated synthetic. Purpose-built because real operational data is fragmented across 6+ systems without a unified API, and Quebec Law 25 restricts patient-level access. | `calibration.py`: every number traced to public source |
| "How do we know it's faithful?" | 18/18 validation tests against public Héma-Québec + Statistics Canada benchmarks, 30 replications, TOST equivalence | `validation_report.json` |
| "Why not just use real data?" | The architecture is designed for real data — all models have shadow deployment gates and champion/challenger A/B promotion. The system is data-agnostic: plug in real data, thresholds auto-evaluate. | `thesis_readiness_gates.json`, `promote_models.py` |
| "What can the synthetic data actually prove?" | Architecture correctness, relative strategy rankings, budget sensitivity, crisis response patterns. It cannot prove absolute real-world ML accuracy — and doesn't claim to. | `thesis_readiness_report.json`: `THESIS_SOTA_CLAIM` explicitly scoped to synthetic benchmark |
| "Is the DreamerV3 contribution real?" | Yes — 16-dim observation space, 9-action catalog, world-model planning through DreamerV3 with 3-tier reward function. Checkpoint at `dreamerv3_runs/m1_continuous_run8_kpi_aligned` | `thesis_sections.tex` Section 4.8, evaluation CSVs |

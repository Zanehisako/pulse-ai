# Dynamic Orchestrator LLM Benchmark

Generated at: `2026-04-21T23:20:50.217974+00:00`

## Experimental Setup

This benchmark evaluates each locally available GGUF orchestrator model against natural-language prompts that require predictive-model inference, simulation, or db_search routing. The tool menu is varied per case to quantify whether the planner remains accurate as tools appear and disappear.

- Number of runs: `56`
- Number of prompt/menu cases: `14`
- Number of orchestrator LLMs: `4`

## Overall Summary

| orchestrator_model_id | orchestrator_model_name | runs | load_time_ms | planned_tool_match | executed_tool_match | execution_success | menu_adherence | argument_match | case_score | median_latency_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| glm_4_9b_0414_ud_iq3_xxs | Glm 4 9B 0414 Q3_XXS | 14 | 15598.07 | 0.9286 | 0.9286 | 1.0 | 1.0 | 1.0 | 0.9643 | 76509.135 |
| qwen2_7b_instruct_q4_0 | Qwen2 7B Instruct Q4_0 | 14 | 22878.65 | 0.9286 | 0.9286 | 1.0 | 1.0 | 0.9286 | 0.9536 | 41716.76 |
| xlam_7b_q4_k_m | xLAM 7B Q4_K_M | 14 | 36434.29 | 0.9286 | 0.9286 | 1.0 | 1.0 | 0.9286 | 0.9536 | 69656.105 |
| xlam_7b_q2_k | xLAM 7B Q2_K | 14 | 253127.47 | 0.8571 | 0.8571 | 1.0 | 1.0 | 0.9286 | 0.9179 | 336400.785 |

## Tool-Family Summary

| orchestrator_model_name | category | planned_tool_match | executed_tool_match | case_score | latency_ms |
| --- | --- | --- | --- | --- | --- |
| Glm 4 9B 0414 Q3_XXS | db_search | 1.0 | 1.0 | 1.0 | 82635.39 |
| Qwen2 7B Instruct Q4_0 | db_search | 1.0 | 1.0 | 0.9625 | 41716.76 |
| xLAM 7B Q2_K | db_search | 1.0 | 1.0 | 0.9625 | 477292.085 |
| xLAM 7B Q4_K_M | db_search | 1.0 | 1.0 | 0.9625 | 69513.465 |
| Glm 4 9B 0414 Q3_XXS | predictive_model | 0.8333 | 0.8333 | 0.9167 | 69718.115 |
| Qwen2 7B Instruct Q4_0 | predictive_model | 0.8333 | 0.8333 | 0.9167 | 39046.295 |
| xLAM 7B Q2_K | predictive_model | 0.6667 | 0.6667 | 0.8333 | 60743.785 |
| xLAM 7B Q4_K_M | predictive_model | 0.8333 | 0.8333 | 0.9167 | 77256.865 |
| Glm 4 9B 0414 Q3_XXS | simulation | 1.0 | 1.0 | 1.0 | 81596.18 |
| Qwen2 7B Instruct Q4_0 | simulation | 1.0 | 1.0 | 1.0 | 41655.58 |
| xLAM 7B Q2_K | simulation | 1.0 | 1.0 | 1.0 | 281068.02 |
| xLAM 7B Q4_K_M | simulation | 1.0 | 1.0 | 1.0 | 59835.405 |

## Tool-Menu Responsiveness

| orchestrator_model_name | menu_id | menu_label | runs | planned_tool_match | executed_tool_match | menu_adherence | case_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Glm 4 9B 0414 Q3_XXS | models_plus_db | All predictive models + DB | 2 | 1.0 | 1.0 | 1.0 | 1.0 |
| Qwen2 7B Instruct Q4_0 | models_plus_db | All predictive models + DB | 2 | 1.0 | 1.0 | 1.0 | 1.0 |
| xLAM 7B Q2_K | models_plus_db | All predictive models + DB | 2 | 0.5 | 0.5 | 1.0 | 0.75 |
| xLAM 7B Q4_K_M | models_plus_db | All predictive models + DB | 2 | 1.0 | 1.0 | 1.0 | 1.0 |
| Glm 4 9B 0414 Q3_XXS | models_plus_simulation | All predictive models + simulation | 1 | 1.0 | 1.0 | 1.0 | 1.0 |
| Qwen2 7B Instruct Q4_0 | models_plus_simulation | All predictive models + simulation | 1 | 1.0 | 1.0 | 1.0 | 1.0 |
| xLAM 7B Q2_K | models_plus_simulation | All predictive models + simulation | 1 | 1.0 | 1.0 | 1.0 | 1.0 |
| xLAM 7B Q4_K_M | models_plus_simulation | All predictive models + simulation | 1 | 1.0 | 1.0 | 1.0 | 1.0 |
| Glm 4 9B 0414 Q3_XXS | all_tools_all_models | All tools + all predictive models | 6 | 0.8333 | 0.8333 | 1.0 | 0.9167 |
| Qwen2 7B Instruct Q4_0 | all_tools_all_models | All tools + all predictive models | 6 | 0.8333 | 0.8333 | 1.0 | 0.8917 |
| xLAM 7B Q2_K | all_tools_all_models | All tools + all predictive models | 6 | 0.8333 | 0.8333 | 1.0 | 0.8917 |
| xLAM 7B Q4_K_M | all_tools_all_models | All tools + all predictive models | 6 | 0.8333 | 0.8333 | 1.0 | 0.8917 |
| Glm 4 9B 0414 Q3_XXS | db_only | DB search only | 1 | 1.0 | 1.0 | 1.0 | 1.0 |
| Qwen2 7B Instruct Q4_0 | db_only | DB search only | 1 | 1.0 | 1.0 | 1.0 | 1.0 |
| xLAM 7B Q2_K | db_only | DB search only | 1 | 1.0 | 1.0 | 1.0 | 1.0 |
| xLAM 7B Q4_K_M | db_only | DB search only | 1 | 1.0 | 1.0 | 1.0 | 1.0 |
| Glm 4 9B 0414 Q3_XXS | donor_only | Donor model only | 1 | 1.0 | 1.0 | 1.0 | 1.0 |
| Qwen2 7B Instruct Q4_0 | donor_only | Donor model only | 1 | 1.0 | 1.0 | 1.0 | 1.0 |
| xLAM 7B Q2_K | donor_only | Donor model only | 1 | 1.0 | 1.0 | 1.0 | 1.0 |
| xLAM 7B Q4_K_M | donor_only | Donor model only | 1 | 1.0 | 1.0 | 1.0 | 1.0 |
| Glm 4 9B 0414 Q3_XXS | hospital_only | Hospital shortage model only | 1 | 1.0 | 1.0 | 1.0 | 1.0 |
| Qwen2 7B Instruct Q4_0 | hospital_only | Hospital shortage model only | 1 | 1.0 | 1.0 | 1.0 | 1.0 |
| xLAM 7B Q2_K | hospital_only | Hospital shortage model only | 1 | 1.0 | 1.0 | 1.0 | 1.0 |
| xLAM 7B Q4_K_M | hospital_only | Hospital shortage model only | 1 | 1.0 | 1.0 | 1.0 | 1.0 |
| Glm 4 9B 0414 Q3_XXS | simulation_only | Simulation only | 1 | 1.0 | 1.0 | 1.0 | 1.0 |
| Qwen2 7B Instruct Q4_0 | simulation_only | Simulation only | 1 | 1.0 | 1.0 | 1.0 | 1.0 |
| xLAM 7B Q2_K | simulation_only | Simulation only | 1 | 1.0 | 1.0 | 1.0 | 1.0 |
| xLAM 7B Q4_K_M | simulation_only | Simulation only | 1 | 1.0 | 1.0 | 1.0 | 1.0 |
| Glm 4 9B 0414 Q3_XXS | stockout_only | Stockout-days model only | 1 | 1.0 | 1.0 | 1.0 | 1.0 |
| Qwen2 7B Instruct Q4_0 | stockout_only | Stockout-days model only | 1 | 1.0 | 1.0 | 1.0 | 1.0 |
| xLAM 7B Q2_K | stockout_only | Stockout-days model only | 1 | 1.0 | 1.0 | 1.0 | 1.0 |
| xLAM 7B Q4_K_M | stockout_only | Stockout-days model only | 1 | 1.0 | 1.0 | 1.0 | 1.0 |

## Dynamic Trace: Glm 4 9B 0414 Q3_XXS

| family | menu_label | available_tools | expected_tool | planned_tool | executed_tool | case_score |
| --- | --- | --- | --- | --- | --- | --- |
| donor_age_average | All predictive models + DB | db_tool, donor_propensity_model, hospital_shortage_predictor, stockout_days_predictor | db_tool | db_tool | db_tool | 1.0 |
| donor_propensity | All tools + all predictive models | db_tool, simulation, donor_propensity_model, hospital_shortage_predictor, stockout_days_predictor | donor_propensity_model | donor_propensity_model | donor_propensity_model | 1.0 |
| donor_propensity | Donor model only | donor_propensity_model | donor_propensity_model | donor_propensity_model | donor_propensity_model | 1.0 |
| historical_stock_average | All tools + all predictive models | db_tool, simulation, donor_propensity_model, hospital_shortage_predictor, stockout_days_predictor | db_tool | db_tool | db_tool | 1.0 |
| hospital_shortage | All tools + all predictive models | db_tool, simulation, donor_propensity_model, hospital_shortage_predictor, stockout_days_predictor | hospital_shortage_predictor | simulation | simulation | 0.5 |
| hospital_shortage | Hospital shortage model only | hospital_shortage_predictor | hospital_shortage_predictor | hospital_shortage_predictor | hospital_shortage_predictor | 1.0 |
| inventory_lookup | All tools + all predictive models | db_tool, simulation, donor_propensity_model, hospital_shortage_predictor, stockout_days_predictor | db_tool | db_tool | db_tool | 1.0 |
| inventory_lookup | DB search only | db_tool | db_tool | db_tool | db_tool | 1.0 |
| policy_compare | All tools + all predictive models | db_tool, simulation, donor_propensity_model, hospital_shortage_predictor, stockout_days_predictor | simulation | simulation | simulation | 1.0 |
| shock_forecast | All tools + all predictive models | db_tool, simulation, donor_propensity_model, hospital_shortage_predictor, stockout_days_predictor | simulation | simulation | simulation | 1.0 |
| shock_forecast | Simulation only | simulation | simulation | simulation | simulation | 1.0 |
| shock_recommend | All predictive models + simulation | simulation, donor_propensity_model, hospital_shortage_predictor, stockout_days_predictor | simulation | simulation | simulation | 1.0 |
| stockout_days | All predictive models + DB | db_tool, donor_propensity_model, hospital_shortage_predictor, stockout_days_predictor | stockout_days_predictor | stockout_days_predictor | stockout_days_predictor | 1.0 |
| stockout_days | Stockout-days model only | stockout_days_predictor | stockout_days_predictor | stockout_days_predictor | stockout_days_predictor | 1.0 |

## Dynamic Trace: Qwen2 7B Instruct Q4_0

| family | menu_label | available_tools | expected_tool | planned_tool | executed_tool | case_score |
| --- | --- | --- | --- | --- | --- | --- |
| donor_age_average | All predictive models + DB | db_tool, donor_propensity_model, hospital_shortage_predictor, stockout_days_predictor | db_tool | db_tool | db_tool | 1.0 |
| donor_propensity | All tools + all predictive models | db_tool, simulation, donor_propensity_model, hospital_shortage_predictor, stockout_days_predictor | donor_propensity_model | donor_propensity_model | donor_propensity_model | 1.0 |
| donor_propensity | Donor model only | donor_propensity_model | donor_propensity_model | donor_propensity_model | donor_propensity_model | 1.0 |
| historical_stock_average | All tools + all predictive models | db_tool, simulation, donor_propensity_model, hospital_shortage_predictor, stockout_days_predictor | db_tool | db_tool | db_tool | 1.0 |
| hospital_shortage | All tools + all predictive models | db_tool, simulation, donor_propensity_model, hospital_shortage_predictor, stockout_days_predictor | hospital_shortage_predictor | simulation | simulation | 0.5 |
| hospital_shortage | Hospital shortage model only | hospital_shortage_predictor | hospital_shortage_predictor | hospital_shortage_predictor | hospital_shortage_predictor | 1.0 |
| inventory_lookup | All tools + all predictive models | db_tool, simulation, donor_propensity_model, hospital_shortage_predictor, stockout_days_predictor | db_tool | db_tool | db_tool | 0.85 |
| inventory_lookup | DB search only | db_tool | db_tool | db_tool | db_tool | 1.0 |
| policy_compare | All tools + all predictive models | db_tool, simulation, donor_propensity_model, hospital_shortage_predictor, stockout_days_predictor | simulation | simulation | simulation | 1.0 |
| shock_forecast | All tools + all predictive models | db_tool, simulation, donor_propensity_model, hospital_shortage_predictor, stockout_days_predictor | simulation | simulation | simulation | 1.0 |
| shock_forecast | Simulation only | simulation | simulation | simulation | simulation | 1.0 |
| shock_recommend | All predictive models + simulation | simulation, donor_propensity_model, hospital_shortage_predictor, stockout_days_predictor | simulation | simulation | simulation | 1.0 |
| stockout_days | All predictive models + DB | db_tool, donor_propensity_model, hospital_shortage_predictor, stockout_days_predictor | stockout_days_predictor | stockout_days_predictor | stockout_days_predictor | 1.0 |
| stockout_days | Stockout-days model only | stockout_days_predictor | stockout_days_predictor | stockout_days_predictor | stockout_days_predictor | 1.0 |

## Dynamic Trace: xLAM 7B Q2_K

| family | menu_label | available_tools | expected_tool | planned_tool | executed_tool | case_score |
| --- | --- | --- | --- | --- | --- | --- |
| donor_age_average | All predictive models + DB | db_tool, donor_propensity_model, hospital_shortage_predictor, stockout_days_predictor | db_tool | db_tool | db_tool | 1.0 |
| donor_propensity | All tools + all predictive models | db_tool, simulation, donor_propensity_model, hospital_shortage_predictor, stockout_days_predictor | donor_propensity_model | donor_propensity_model | donor_propensity_model | 1.0 |
| donor_propensity | Donor model only | donor_propensity_model | donor_propensity_model | donor_propensity_model | donor_propensity_model | 1.0 |
| historical_stock_average | All tools + all predictive models | db_tool, simulation, donor_propensity_model, hospital_shortage_predictor, stockout_days_predictor | db_tool | db_tool | db_tool | 1.0 |
| hospital_shortage | All tools + all predictive models | db_tool, simulation, donor_propensity_model, hospital_shortage_predictor, stockout_days_predictor | hospital_shortage_predictor | simulation | simulation | 0.5 |
| hospital_shortage | Hospital shortage model only | hospital_shortage_predictor | hospital_shortage_predictor | hospital_shortage_predictor | hospital_shortage_predictor | 1.0 |
| inventory_lookup | All tools + all predictive models | db_tool, simulation, donor_propensity_model, hospital_shortage_predictor, stockout_days_predictor | db_tool | db_tool | db_tool | 0.85 |
| inventory_lookup | DB search only | db_tool | db_tool | db_tool | db_tool | 1.0 |
| policy_compare | All tools + all predictive models | db_tool, simulation, donor_propensity_model, hospital_shortage_predictor, stockout_days_predictor | simulation | simulation | simulation | 1.0 |
| shock_forecast | All tools + all predictive models | db_tool, simulation, donor_propensity_model, hospital_shortage_predictor, stockout_days_predictor | simulation | simulation | simulation | 1.0 |
| shock_forecast | Simulation only | simulation | simulation | simulation | simulation | 1.0 |
| shock_recommend | All predictive models + simulation | simulation, donor_propensity_model, hospital_shortage_predictor, stockout_days_predictor | simulation | simulation | simulation | 1.0 |
| stockout_days | All predictive models + DB | db_tool, donor_propensity_model, hospital_shortage_predictor, stockout_days_predictor | stockout_days_predictor | db_tool | db_tool | 0.5 |
| stockout_days | Stockout-days model only | stockout_days_predictor | stockout_days_predictor | stockout_days_predictor | stockout_days_predictor | 1.0 |

## Dynamic Trace: xLAM 7B Q4_K_M

| family | menu_label | available_tools | expected_tool | planned_tool | executed_tool | case_score |
| --- | --- | --- | --- | --- | --- | --- |
| donor_age_average | All predictive models + DB | db_tool, donor_propensity_model, hospital_shortage_predictor, stockout_days_predictor | db_tool | db_tool | db_tool | 1.0 |
| donor_propensity | All tools + all predictive models | db_tool, simulation, donor_propensity_model, hospital_shortage_predictor, stockout_days_predictor | donor_propensity_model | donor_propensity_model | donor_propensity_model | 1.0 |
| donor_propensity | Donor model only | donor_propensity_model | donor_propensity_model | donor_propensity_model | donor_propensity_model | 1.0 |
| historical_stock_average | All tools + all predictive models | db_tool, simulation, donor_propensity_model, hospital_shortage_predictor, stockout_days_predictor | db_tool | db_tool | db_tool | 1.0 |
| hospital_shortage | All tools + all predictive models | db_tool, simulation, donor_propensity_model, hospital_shortage_predictor, stockout_days_predictor | hospital_shortage_predictor | simulation | simulation | 0.5 |
| hospital_shortage | Hospital shortage model only | hospital_shortage_predictor | hospital_shortage_predictor | hospital_shortage_predictor | hospital_shortage_predictor | 1.0 |
| inventory_lookup | All tools + all predictive models | db_tool, simulation, donor_propensity_model, hospital_shortage_predictor, stockout_days_predictor | db_tool | db_tool | db_tool | 0.85 |
| inventory_lookup | DB search only | db_tool | db_tool | db_tool | db_tool | 1.0 |
| policy_compare | All tools + all predictive models | db_tool, simulation, donor_propensity_model, hospital_shortage_predictor, stockout_days_predictor | simulation | simulation | simulation | 1.0 |
| shock_forecast | All tools + all predictive models | db_tool, simulation, donor_propensity_model, hospital_shortage_predictor, stockout_days_predictor | simulation | simulation | simulation | 1.0 |
| shock_forecast | Simulation only | simulation | simulation | simulation | simulation | 1.0 |
| shock_recommend | All predictive models + simulation | simulation, donor_propensity_model, hospital_shortage_predictor, stockout_days_predictor | simulation | simulation | simulation | 1.0 |
| stockout_days | All predictive models + DB | db_tool, donor_propensity_model, hospital_shortage_predictor, stockout_days_predictor | stockout_days_predictor | stockout_days_predictor | stockout_days_predictor | 1.0 |
| stockout_days | Stockout-days model only | stockout_days_predictor | stockout_days_predictor | stockout_days_predictor | stockout_days_predictor | 1.0 |

## Plot Files

- `overall_quality_metrics.png`
- `planned_accuracy_by_tool_family.png`
- `dynamic_score_by_available_tool_count.png`
- `menu_responsiveness_heatmap.png`
- `latency_distribution.png`
- `tool_family_confusion_matrices.png`

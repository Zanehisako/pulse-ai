# Blood Supply Simulator

This directory contains the discrete-event blood supply simulator, the RL training entry points, the offline RL pipeline, the evaluation scripts, and the validation utilities used by the `simulation_studio` web app.

The current simulator supports:

- fixed intervention strategies
- online RL policies: PPO and SAC
- offline RL policies: IQL and CQL
- optional DreamerV3 evaluation when a usable checkpoint exists
- validation and budget-sensitivity analysis
- thesis-ready evaluation plots in PNG, PDF, and SVG

## What Is In This Folder

Core runtime:

- `core.py`: entities, graph loading, centre configuration, helpers
- `scenarios.py`: built-in scenarios, strategies, action definitions
- `engine.py`: simulator runtime, controllers, policy loading, reporting
- `main.py`: simple CLI runner for single scenario runs and interactive usage

Evaluation and reporting:

- `evaluate_strategies.py`: batch evaluation across scenarios and strategies, plus plots
- `compare_rl_benchmarks.py`: compare evaluated policies against the archived DreamerV3 `m1_continuous_run8_kpi_aligned` training log
- `compare_agents.py`: compare MAPPO and PPO checkpoints
- `eval_budget_strategies.py`: evaluate strategies across budget grids
- `run_validation.py`: simulation validation and sensitivity analysis
- `plot_dreamerv3_metrics.py`: plot DreamerV3 training metrics from `metrics.jsonl`

Training and dataset generation:

- `train_ppo.py`: PPO training in discrete or continuous action space
- `train_sac.py`: SAC training in continuous action space
- `collect_offline_dataset.py`: collect `.npz` transition datasets from simulator rollouts
- `train_offline_rl.py`: train IQL or CQL from a fixed dataset
- `train_mappo.py`: MAPPO training
- `train_ppo_budget.py`: budget-aware PPO on the fast environment
- `train_dreamer.py`: official DreamerV3 training wrapper
- `train_dreamer_budget.py`: budget-aware DreamerV3 training wrapper
- `train_dreamerv4.py`: legacy DreamerV4 workflow; not used by `simulation_studio`

Generated artifacts:

- `results/`: evaluation CSVs, plots, and manifests
- `validation_results/`: validation reports and sensitivity outputs
- `dreamerv3_runs/`: DreamerV3 logs and checkpoints
- `dreamerv4_runs/`: legacy DreamerV4 logs
- `ppo_blood_model.zip`, `sac_blood_model.zip`, `iql_blood_model.pt`, `cql_blood_model.pt`: local model artifacts
- `offline_blood_dataset.npz`: example offline dataset artifact

## Environment Setup

Recommended from the repo root:

```bash
cd /Users/mac/Documents/pios-1
python -m venv .venv
source .venv/bin/activate
pip install -e ./ml-backend
```

Useful environment variables:

- `PIOS_SIM_OFFLINE=1`: avoid live OpenStreetMap graph fetches and use the offline graph path instead
- `MPLCONFIGDIR=/tmp/matplotlib`: avoids Matplotlib cache warnings on machines where `~/.matplotlib` is not writable
- `PIOS_DREAMERV3_CHECKPOINT`: optional explicit DreamerV3 checkpoint path
- `PIOS_DREAMERV4_CHECKPOINT`: optional explicit DreamerV4 checkpoint path for legacy runs

Recommended shell setup for local work:

```bash
export PIOS_SIM_OFFLINE=1
export MPLCONFIGDIR=/tmp/matplotlib
```

## Built-In Scenarios

Training scenarios:

- `baseline`
- `donor_decrease`
- `demand_surge`
- `transport_disruption`
- `combined_crisis`

Holdout and generalization scenarios:

- `holiday_flu_wave`
- `regional_testing_backlog`
- `provincial_supply_crunch`
- `post_storm_backlog`

List the current scenario catalog directly from the CLI:

```bash
cd /Users/mac/Documents/pios-1/ml-backend/simulator
python main.py --list
```

## Built-In Strategies

Fixed strategies:

- `baseline`
- `mass_campaign`
- `lab_investment`
- `emergency_network`
- `full_response`

Learned strategies:

- `ppo_shortage_minimizer`
- `ppo_continuous`
- `sac_continuous`
- `iql_offline`
- `cql_offline`
- `dreamerv3_official`

Legacy:

- `dreamerv4`

`dreamerv3_official` and `dreamerv4` are evaluated only when a usable checkpoint can be resolved. If not, `evaluate_strategies.py` now records them as skipped instead of failing the run.

## Quick Start Commands

Run a single scenario and generate a text report:

```bash
cd /Users/mac/Documents/pios-1/ml-backend/simulator
python main.py --scenario donor_decrease --strategy mass_campaign --hours 168
```

Interactive mode:

```bash
python main.py
```

Run all scenarios with their default strategies:

```bash
python main.py --scenario all --quiet
```

Fast smoke run without detailed history:

```bash
python main.py --scenario baseline --strategy baseline --hours 72 --fast --quiet
```

## Thesis-Ready Strategy Evaluation

This is the main benchmarking command for cross-strategy evaluation and publication-style plots:

```bash
cd /Users/mac/Documents/pios-1/ml-backend/simulator
python evaluate_strategies.py \
  --runs 2 \
  --output-prefix strategy_eval_thesis \
  --thesis-ready
```

What this does:

- evaluates the default scenario set and the RL comparison default strategy set
- defaults to the RL comparison set: `ppo_continuous`, `sac_continuous`, `iql_offline`, `cql_offline`, and `dreamerv3_official`
- targets `dreamerv3_runs/m1_continuous_run8_kpi_aligned` by default for DreamerV3 evaluation
- exports summary tables
- exports heatmaps, Pareto plots, scenario plots, and thesis overview scorecards
- writes PNG, PDF, and SVG when `--thesis-ready` is enabled
- skips checkpoint-backed Dreamer strategies when checkpoints are not available

Useful variants:

```bash
python evaluate_strategies.py --scenario baseline --runs 3
python evaluate_strategies.py --strategy ppo_continuous --strategy sac_continuous --strategy iql_offline --strategy cql_offline
python evaluate_strategies.py --hours 72 --runs 1 --output-prefix smoke_eval
python evaluate_strategies.py --thesis-ready --export-format pdf --export-format svg
python evaluate_strategies.py --pareto-x-metric shortage_rate_mean --pareto-y-metric reward_total_mean --pareto-y-goal max
```

Main output files for `--output-prefix strategy_eval_thesis`:

- `results/strategy_eval_thesis_detail.csv`
- `results/strategy_eval_thesis_summary.csv`
- `results/strategy_eval_thesis_overall_summary.csv`
- `results/strategy_eval_thesis_strategy_status.csv`
- `results/strategy_eval_thesis_artifacts.json`
- `results/strategy_eval_thesis_heatmaps.{png,pdf,svg}`
- `results/strategy_eval_thesis_pareto_overall.{png,pdf,svg}`
- `results/strategy_eval_thesis_thesis_overview.{png,pdf,svg}`
- `results/strategy_eval_thesis_thesis_component_shortages.{png,pdf,svg}`
- `results/strategy_eval_thesis_thesis_scenario_wins.{png,pdf,svg}`
- `results/strategy_eval_thesis_scenario_plots/`
- `results/strategy_eval_thesis_pareto_plots/`

Local benchmark note from the current workspace:

- `ppo_shortage_minimizer` had the best aggregate shortage rate in the latest full strategy thesis run
- `sac_continuous` and `cql_offline` were the next strongest learned policies on aggregate shortage
- `dreamerv3_official` and `dreamerv4` were skipped because no runnable local checkpoint was available

## PPO and SAC Training

Discrete PPO:

```bash
cd /Users/mac/Documents/pios-1/ml-backend/simulator
python train_ppo.py --timesteps 200000 --save ppo_blood_model
```

Continuous-action PPO:

```bash
python train_ppo.py \
  --action-mode continuous \
  --timesteps 200000 \
  --save ppo_blood_model
```

Evaluate an existing PPO checkpoint:

```bash
python train_ppo.py --eval --model ppo_blood_model.zip --scenario baseline
```

Train SAC:

```bash
python train_sac.py --timesteps 200000 --save sac_blood_model
```

Evaluate an existing SAC checkpoint:

```bash
python train_sac.py --eval --model sac_blood_model.zip --scenario baseline
```

Common knobs:

- `--scenario`: single key, comma-separated keys, or `all` where supported
- `--eval-scenario`: evaluation scenario after multi-scenario training
- `--ep-hours`: hours per episode
- `--step-hours`: control interval in simulation hours
- `--seed`: reproducible training seed
- `--online-city-graph`: use the live city graph instead of the offline stub

## Offline RL: Dataset, IQL, and CQL

Collect a dataset:

```bash
cd /Users/mac/Documents/pios-1/ml-backend/simulator
python collect_offline_dataset.py \
  --episodes 200 \
  --scenario baseline,donor_decrease \
  --source-policy mixed \
  --expert-policy ppo_continuous \
  --output offline_blood_dataset.npz
```

Alternative dataset collection modes:

```bash
python collect_offline_dataset.py --source-policy random --episodes 100
python collect_offline_dataset.py --source-policy sac_continuous --episodes 100
python collect_offline_dataset.py --source-policy mixed --expert-policy sac_continuous --noise-std 0.08
```

Train IQL:

```bash
python train_offline_rl.py \
  --algorithm iql \
  --dataset offline_blood_dataset.npz \
  --save iql_blood_model.pt
```

Train CQL:

```bash
python train_offline_rl.py \
  --algorithm cql \
  --dataset offline_blood_dataset.npz \
  --save cql_blood_model.pt
```

Useful offline RL knobs:

- `--updates`: number of gradient updates
- `--batch-size`: minibatch size
- `--hidden-dim`: network width
- `--device`: `cpu`, `cuda`, or `mps` when available
- `--expectile`, `--beta`, `--tau`: IQL hyperparameters
- `--alpha`, `--cql-alpha`, `--action-samples`: CQL hyperparameters

Evaluate the learned policies together:

```bash
python evaluate_strategies.py \
  --strategy ppo_continuous \
  --strategy sac_continuous \
  --strategy iql_offline \
  --strategy cql_offline \
  --runs 3 \
  --output-prefix rl_head_to_head
```

## RL Comparison Utilities

Compare current strategy summaries against the archived DreamerV3 run log:

```bash
cd /Users/mac/Documents/pios-1/ml-backend/simulator
python compare_rl_benchmarks.py \
  --strategy-summary results/strategy_eval_thesis_summary.csv \
  --dreamer-run-dir dreamerv3_runs/m1_continuous_run8_kpi_aligned \
  --output results/strategy_eval_thesis_vs_dreamerv3.csv
```

Compare MAPPO against PPO:

```bash
python compare_agents.py \
  --mappo-model results/mappo.pt \
  --ppo-model ppo_blood_model.zip \
  --runs 3 \
  --output-prefix agent_compare
```

## Validation

Quick validation:

```bash
cd /Users/mac/Documents/pios-1/ml-backend/simulator
python run_validation.py --mode quick --scenario baseline
```

Standard validation with LaTeX tables:

```bash
python run_validation.py --mode standard --latex --output-dir validation_results
```

Full validation:

```bash
python run_validation.py --mode full --scenario baseline --hours 168
```

Sensitivity-only rerun:

```bash
python run_validation.py --mode sensitivity-only --scenario baseline
```

Rebuild the report from saved JSON:

```bash
python run_validation.py --mode report-only --output-dir validation_results
```

Validation outputs are written under `validation_results/` by default.

## Budget-Aware Training and Evaluation

Train budget-aware PPO:

```bash
cd /Users/mac/Documents/pios-1/ml-backend/simulator
python train_ppo_budget.py --timesteps 200000 --budget 6500 --save ppo_budget_model
```

Curriculum budget training:

```bash
python train_ppo_budget.py --budget-curriculum 3000,4500,6000,9000
```

Evaluate strategies over a budget grid:

```bash
python eval_budget_strategies.py \
  --budgets 3000,4500,6000,7500,9000,12000 \
  --runs 2 \
  --output-prefix budget_eval
```

Include holdout scenarios:

```bash
python eval_budget_strategies.py --include-holdouts --runs 1 --hours 72
```

## MAPPO and Dreamer Utilities

Train MAPPO:

```bash
cd /Users/mac/Documents/pios-1/ml-backend/simulator
python train_mappo.py --timesteps 200000 --save results/mappo.pt --plot results/mappo_training.png
```

Train DreamerV3:

```bash
python train_dreamer.py --steps 50000 --logdir dreamerv3_runs/local_run
```

Budget-aware DreamerV3:

```bash
python train_dreamer_budget.py --steps 50000 --scenario baseline
```

Evaluate a DreamerV3 checkpoint:

```bash
python eval_dreamer.py --model dreamerv3_runs/m1_continuous_run8_kpi_aligned/ckpt
```

Plot DreamerV3 metrics:

```bash
python plot_dreamerv3_metrics.py \
  dreamerv3_runs/m1_continuous_run8_kpi_aligned/metrics.jsonl \
  --output dreamerv3_runs/m1_continuous_run8_kpi_aligned/metrics_plot.png
```

Legacy DreamerV4 workflow:

```bash
python train_dreamerv4.py --help
```

DreamerV4 remains in this folder as a legacy research path. It is no longer exposed in `simulation_studio`.

## Command Reference

Every top-level CLI entry point in this folder can be inspected with `--help`.

Mainline commands:

| Command | Purpose | Example |
|---|---|---|
| `python main.py` | interactive and one-off simulator runs | `python main.py --scenario baseline --strategy baseline --hours 72` |
| `python evaluate_strategies.py` | batch evaluation and plot export | `python evaluate_strategies.py --runs 2 --thesis-ready` |
| `python compare_rl_benchmarks.py` | compare strategy summary vs archived DreamerV3 logs | `python compare_rl_benchmarks.py --strategy-summary results/strategy_eval_thesis_summary.csv` |
| `python compare_agents.py` | compare MAPPO and PPO checkpoints | `python compare_agents.py --mappo-model results/mappo.pt --ppo-model ppo_blood_model.zip` |
| `python run_validation.py` | validation and sensitivity analysis | `python run_validation.py --mode standard --latex` |

Training and data commands:

| Command | Purpose | Example |
|---|---|---|
| `python train_ppo.py` | PPO training and eval | `python train_ppo.py --action-mode continuous --timesteps 200000` |
| `python train_sac.py` | SAC training and eval | `python train_sac.py --timesteps 200000` |
| `python collect_offline_dataset.py` | collect offline dataset | `python collect_offline_dataset.py --episodes 200 --source-policy mixed` |
| `python train_offline_rl.py` | train IQL or CQL | `python train_offline_rl.py --algorithm iql --dataset offline_blood_dataset.npz` |
| `python train_mappo.py` | MAPPO training | `python train_mappo.py --timesteps 200000 --save results/mappo.pt` |
| `python train_ppo_budget.py` | budget-aware PPO | `python train_ppo_budget.py --budget 6500` |
| `python train_dreamer.py` | DreamerV3 training wrapper | `python train_dreamer.py --steps 50000 --logdir dreamerv3_runs/local_run` |
| `python train_dreamer_budget.py` | budget-aware DreamerV3 | `python train_dreamer_budget.py --steps 50000` |
| `python train_dreamerv4.py` | legacy DreamerV4 workflow | `python train_dreamerv4.py --help` |

Analysis and plotting commands:

| Command | Purpose | Example |
|---|---|---|
| `python eval_budget_strategies.py` | evaluate budget sensitivity | `python eval_budget_strategies.py --budgets 3000,4500,6000` |
| `python eval_dreamer.py` | evaluate a Dreamer checkpoint | `python eval_dreamer.py --help` |
| `python plot_dreamerv3_metrics.py` | plot DreamerV3 training metrics | `python plot_dreamerv3_metrics.py --help` |
| `python generate_mcd_diagram.py` | diagram generation helper | `python generate_mcd_diagram.py` |
| `python generate_mld_diagram.py` | diagram generation helper | `python generate_mld_diagram.py` |

## Results and Artifact Layout

Typical outputs:

- batch evaluation CSVs and figures: `results/`
- validation reports: `validation_results/`
- saved text reports from `main.py`: `reports/` or the directory passed with `--output`
- Dreamer metrics and score logs: `dreamerv3_runs/<run_name>/`
- local RL checkpoints: this folder unless you pass a custom output path

If you are producing figures for a thesis or paper, keep the `results/<prefix>_artifacts.json` manifest. It records the generated files for that evaluation run.

## Testing

Recommended verification commands from the repo root:

```bash
cd /Users/mac/Documents/pios-1
pytest -q ml-backend/tests/test_evaluate_strategies.py
pytest -q ml-backend/tests/test_offline_rl.py
pytest -q ml-backend/tests/test_plot_dreamerv3_metrics.py
```

## Troubleshooting

OpenStreetMap or graph-loading delay:

- set `PIOS_SIM_OFFLINE=1`

Matplotlib cache warnings:

- set `MPLCONFIGDIR=/tmp/matplotlib`

Dreamer strategy skipped during evaluation:

- verify the checkpoint directory contains a runnable checkpoint, not only replay data
- otherwise pass an explicit path with `--dreamerv3-checkpoint` or `--dreamerv4-checkpoint`

No PPO or SAC checkpoint found:

- point the evaluator to the file with `--ppo-continuous-model` or `--sac-model`

No offline RL checkpoint found:

- pass `--iql-model` or `--cql-model`

## Current Recommended Workflow

For the current codebase, the shortest end-to-end path is:

1. train or reuse `ppo_continuous`, `sac_continuous`, `iql_offline`, and `cql_offline`
2. run `evaluate_strategies.py --thesis-ready`
3. inspect `results/<prefix>_overall_summary.csv` and `results/<prefix>_strategy_status.csv`
4. load the same policies inside `simulation_studio` Simulation Lab for multi-policy comparison

That is the path the web app and the latest strategy benchmarking work are built around.

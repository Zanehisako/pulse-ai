# DreamerV4 — Research Exploration Status

## What this directory contains

DreamerV4 (`dreamerv4_runs/`) was a research exploration to evaluate whether the DreamerV4 architecture could improve upon the DreamerV3 world-model agent for blood supply chain control.

## Status: Research exploration — did not converge

- **`native/`**: Contains only `metrics.jsonl` (training metrics log). Zero `.pt` checkpoint files — no model weights were ever saved.
- **`best/`**: Contains only `metrics.jsonl` — same situation, no checkpoint files.

The native PyTorch DreamerV4 training loop ran for partial training but never produced a checkpoint meeting the `done` marker threshold. As a result, `evaluate_strategies.py` correctly skips the `dreamerv4` strategy.

## Why it's kept

The directory and code paths are retained for:
1. **Transparency** — acknowledging that not every architecture exploration succeeds
2. **Future work** — the training loop adapter exists and could be re-run if the DreamerV4 PyTorch library stabilizes
3. **Thesis narrative** — DreamerV3 (`m1_continuous_run8_kpi_aligned`) is the primary world-model contribution

## Primary DreamerV3 checkpoint

The working world-model agent is at:
```
dreamerv3_runs/m1_continuous_run8_kpi_aligned/
```

This checkpoint contains the complete agent (`agent.pkl` + `done` marker) and is used by the `dreamerv3_official` strategy in the thesis evaluation pipeline.

## Code references

- `engine.py`: `default_dreamer_checkpoint()` — auto-discovers DreamerV3 checkpoint
- `engine.py`: `resolve_dreamer_checkpoint_path()` — resolution with DreamerV4 support
- `engine.py`: `load_dreamerv4_agent()` — PyTorch DreamerV4 loading (requires `.pt` checkpoints)
- `evaluate_strategies.py`: Strategy `dreamerv4` — skipped when no `.pt` files found under `dreamerv4_runs/`

#!/usr/bin/env bash
# Multi-seed imagination-horizon ablation.
#   seeds {0,1,2} x H {1,5,15,30}, 500K steps each (converged; matched budget).
# Disk hygiene: the replay buffer is capacity-bounded (~1.8G) and this box has hit
# 100% full, so we delete each run's replay/ the instant training finishes and drop
# a DONE marker so the sweep is resumable after any crash/restart.
set -uo pipefail
cd "$(dirname "$0")"
PY=./../.venv/bin/python
export PIOS_SIM_OFFLINE=1 MPLCONFIGDIR=/tmp/matplotlib
SEEDS="${SEEDS:-0 1 2}"
HORIZONS="${HORIZONS:-1 5 15 30}"

for seed in $SEEDS; do
  for H in $HORIZONS; do
    logdir="dreamerv3_runs/ablation_H${H}_s${seed}"
    if [ -f "$logdir/DONE" ]; then
      echo "[skip] $logdir already DONE"; continue
    fi
    echo "=== TRAIN H=$H seed=$seed -> $logdir ($(date '+%H:%M')) ==="
    if $PY official_dreamerv3.py --script train --steps 500000 --size size1m \
        --seed "$seed" --imag-length "$H" --logdir "$logdir" --jax-profiler off; then
      rm -rf "$logdir/replay" "$logdir/plugins" && echo "[cleanup] freed replay+plugins for $logdir"
      touch "$logdir/DONE"
    else
      echo "[FAIL] H=$H seed=$seed — leaving un-DONE for resume"
      rm -rf "$logdir/replay" "$logdir/plugins" 2>/dev/null || true
    fi
    df -h /Users/mac | tail -1
  done
done
echo "ALL_MULTISEED_ABLATION_DONE"

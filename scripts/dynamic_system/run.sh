#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/dynamic_system/run.sh [model] [system]
#
# Examples:
#   scripts/dynamic_system/run.sh patch-tst lorenz
#   scripts/dynamic_system/run.sh time-xer rossler
#   NUM_EPOCHS=20 BATCH_SIZE=64 scripts/dynamic_system/run.sh dlinear lorenz96

MODEL="${1:-patch-tst}"
SYSTEM="${2:-lorenz}"

NUM_EPOCHS="${NUM_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BACKEND="${BACKEND:-wgpu}"
TASK_NAME="${TASK_NAME:-long-term-forecast}"

SEQ_LEN="${SEQ_LEN:-96}"
LABEL_LEN="${LABEL_LEN:-48}"
PRED_LEN="${PRED_LEN:-96}"

# Dynamic-system specific optional overrides.
N_TIMESTEPS="${N_TIMESTEPS:-10000}"
DT="${DT:-0.01}"
H="${H:-0.01}"

case "$SYSTEM" in
  logistic-map)
    DATA_CMD=(logistic-map --n-timesteps "$N_TIMESTEPS")
    ;;
  henon-map)
    DATA_CMD=(henon-map --n-timesteps "$N_TIMESTEPS")
    ;;
  lorenz)
    DATA_CMD=(lorenz --n-timesteps "$N_TIMESTEPS" --h "$H")
    ;;
  lorenz96)
    DATA_CMD=(lorenz96 --total-steps "$N_TIMESTEPS" --dt "$DT" --h "$H")
    ;;
  rossler)
    DATA_CMD=(rossler --n-timesteps "$N_TIMESTEPS" --h "$H")
    ;;
  double-scroll)
    DATA_CMD=(double-scroll --n-timesteps "$N_TIMESTEPS" --h "$H")
    ;;
  multi-scroll)
    DATA_CMD=(multi-scroll --n-timesteps "$N_TIMESTEPS" --h "$H")
    ;;
  rabinovich-fabrikant)
    DATA_CMD=(rabinovich-fabrikant --n-timesteps "$N_TIMESTEPS")
    ;;
  mackey-glass)
    DATA_CMD=(mackey-glass --n-timesteps "$N_TIMESTEPS")
    ;;
  narma)
    DATA_CMD=(narma --n-timesteps "$N_TIMESTEPS")
    ;;
  kuramoto-sivashinsky)
    DATA_CMD=(kuramoto-sivashinsky --n-timesteps "$N_TIMESTEPS")
    ;;
  *)
    echo "Unknown dynamic system: $SYSTEM" >&2
    echo "Supported: logistic-map, henon-map, lorenz, lorenz96, rossler, double-scroll, multi-scroll, rabinovich-fabrikant, mackey-glass, narma, kuramoto-sivashinsky" >&2
    exit 1
    ;;
esac

case "$MODEL" in
  patch-tst)
    MODEL_CMD=(gradient-model patch-tst --activation gelu)
    ;;
  time-xer)
    MODEL_CMD=(gradient-model time-xer --activation gelu)
    ;;
  dlinear)
    MODEL_CMD=(gradient-model d-linear)
    ;;
  *)
    echo "Unknown model: $MODEL" >&2
    echo "Supported: patch-tst, time-xer, dlinear" >&2
    exit 1
    ;;
esac

cd main
cargo run -p main -- \
  --num-epochs "$NUM_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --task-name "$TASK_NAME" \
  --backend "$BACKEND" \
  --seq-len "$SEQ_LEN" \
  --label-len "$LABEL_LEN" \
  --pred-len "$PRED_LEN" \
  "${MODEL_CMD[@]}" \
  "${DATA_CMD[@]}"

#!/usr/bin/env bash
set -euo pipefail

NUM_EPOCHS="${NUM_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BACKEND="${BACKEND:-wgpu}"
TASK_NAME="${TASK_NAME:-long-term-forecast}"

SEQ_LEN="${SEQ_LEN:-96}"
LABEL_LEN="${LABEL_LEN:-48}"
PRED_LEN="${PRED_LEN:-96}"

# NGRC specific arguments
DELAY="${DELAY:-2}"
STRIDE="${STRIDE:-1}"
POLY_ORDER="${POLY_ORDER:-2}"
RIDGE_PARAM="${RIDGE_PARAM:-0.001}"
TRANSIENTS="${TRANSIENTS:-5}"

MODEL_CMD=(rc-model ngrc --delay "$DELAY" --stride "$STRIDE" --poly-order "$POLY_ORDER" --ridge-param "$RIDGE_PARAM" --transients "$TRANSIENTS")

# Dynamic-system specific optional overrides.
N_TIMESTEPS="${N_TIMESTEPS:-6000}"
DT="${DT:-0.01}"
H="${H:-0.01}"
DATA_CMDS=(
    "bool-transform --n-timesteps $N_TIMESTEPS"
    "logistic-map --n-timesteps $N_TIMESTEPS"
    "lorenz --n-timesteps $N_TIMESTEPS --h $H"
    "lorenz96 --n-timesteps $N_TIMESTEPS --dt $DT --h $H"
    "rossler --n-timesteps $N_TIMESTEPS --h $H"
    "double-scroll --n-timesteps $N_TIMESTEPS --h $H"
    "multi-scroll --n-timesteps $N_TIMESTEPS --h $H"
    "mackey-glass --n-timesteps $N_TIMESTEPS"
    "narma --n-timesteps $N_TIMESTEPS"
)

# Go to repository root, then to main
cd "$(dirname "$0")/../main"

for system in "${DATA_CMDS[@]}"
 do
    echo "Running NGRC on $system"
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
        $system
done

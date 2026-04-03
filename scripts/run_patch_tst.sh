MODEL="${1:-patch-tst}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BACKEND="${BACKEND:-wgpu}"
TASK_NAME="${TASK_NAME:-long-term-forecast}"

SEQ_LEN="${SEQ_LEN:-96}"
LABEL_LEN="${LABEL_LEN:-48}"
PRED_LEN="${PRED_LEN:-96}"

MODEL_CMD=(gradient-model patch-tst --activation gelu)

# Dynamic-system specific optional overrides.
N_TIMESTEPS="${N_TIMESTEPS:-60000}"
DT="${DT:-0.01}"
H="${H:-0.01}"
DATA_CMDS=(
    "logistic-map --n-timesteps $N_TIMESTEPS"
    "henon-map --n-timesteps $N_TIMESTEPS"
    "lorenz --n-timesteps $N_TIMESTEPS --h $H"
    "lorenz96 --n-timesteps $N_TIMESTEPS --dt $DT --h $H"
    "rossler --n-timesteps $N_TIMESTEPS --h $H"
    "double-scroll --n-timesteps $N_TIMESTEPS --h $H"
    "multi-scroll --n-timesteps $N_TIMESTEPS --h $H"
    "rabinovich-fabrikant --n-timesteps $N_TIMESTEPS"
    "mackey-glass --n-timesteps $N_TIMESTEPS"
    "narma --n-timesteps $N_TIMESTEPS"
    "kuramoto-sivashinsky --n-timesteps $N_TIMESTEPS"
)
cd main
for system in "${DATA_CMDS[@]}"
 do
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

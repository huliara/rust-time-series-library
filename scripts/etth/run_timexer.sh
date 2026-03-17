cd main && cargo run -p main  -- \
    --num-epochs 1 \
    --task-name long-term-forecast \
    --backend wgpu \
    time-xer \
    --activation gelu \
    et-th1 \
    --train-features ot hufl \
    --targets ot hufl \
    --embed time-f \
    --path ETT/ETTh1.csv
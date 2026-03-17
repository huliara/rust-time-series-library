cd main && cargo run -p main -- \
    --num-epochs 1 \
    --task-name long-term-forecast \
    --backend wgpu \
    patch-tst \
    --activation gelu \
    et-th1 \
    --train-features ot \
    --targets ot \
    --embed time-f \
    --path ETT/ETTh1.csv
cd main && cargo run -p main -- \
    --num-epochs 20 \
    --loss-alpha 1.0 \
    --loss-scale 1.0 \
    --task-name long-term-forecast \
    --backend wgpu \
    patch-tst \
    --activation gelu \
    et-th1 \
    --train-features ot  \
    --targets ot  \
    --embed time-f \
    --path ETT/ETTh1.csv
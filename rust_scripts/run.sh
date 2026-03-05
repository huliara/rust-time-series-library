cargo run -- \
    --num-epochs 10 \
    --task-name long-term-forecast \
    --backend wgpu \
    --data et-th1 \
    --target ot \
    --embed time-f patch-tst \
    --activation gelu
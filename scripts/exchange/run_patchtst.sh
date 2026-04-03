cd main && cargo run -p main -- \
    --num-epochs 2 \
    --task-name long-term-forecast \
    --backend wgpu \
    --batch-size 64 \
    gradient-model \
    patch-tst \
    --activation gelu \
    exchange \
    --train-features open\
    --targets open \
    --embed time-f \
    --path v2/USDJPY/h1/20020101-20250810.csv
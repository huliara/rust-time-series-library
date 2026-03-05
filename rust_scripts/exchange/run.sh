cargo run -- \
    --num-epochs 10 \
    --task-name long-term-forecast \
    --backend wgpu \
    --data exchange \
    --targets high  \
    --embed time-f \
    --path v2/USDJPY/m1/20020101-20250810.csv\
    patch-tst \
    --activation gelu
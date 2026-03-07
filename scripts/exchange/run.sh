cargo run -p main  -- \
    --num-epochs 1 \
    --task-name long-term-forecast \
    --backend wgpu \
    --data exchange \
    --train-features open low\
    --targets high  \
    --embed time-f \
    --path v2/USDJPY/d1/20020101-20250810.csv\
    patch-tst \
    --activation gelu
cd main && cargo run -p main  -- \
    --num-epochs 1 \
    --task-name long-term-forecast \
    --backend wgpu \
    time-xer \
    --activation gelu \
    exchange \
    --train-features open low\
    --targets  open low \
    --embed time-f \
    --path v2/USDJPY/d1/20020101-20250810.csv
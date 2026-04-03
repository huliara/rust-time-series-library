cd main && cargo run -p main  -- \
    --num-epochs 50 \
    --task-name long-term-forecast \
    --backend wgpu \
    --batch-size 128 \
    gradient-model \
    patch-tst \
    --activation gelu \
    et-th1 \
    --train-features ot hufl hull mufl mull lufl lull \
    --targets ot \
    --embed time-f \
    --path ETT/ETTh1.csv
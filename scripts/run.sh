cargo run -- \
    --num-epochs 10 \
    --task-name long-term-forecast \
    --backend wgpu \
    --data et-th1cd  \
    --train-features ot\
    --targets ot  \
    --embed time-f \
    --path ETT/ETTh1.csv\
    patch-tst \
    --activation gelu
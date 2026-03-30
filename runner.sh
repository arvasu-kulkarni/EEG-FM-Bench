#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes=1 --nproc-per-node=1 baseline_main.py conf_file=assets/conf/baseline/manas/manas_bench_long_baseline_short.yaml
# CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nnodes=1 --nproc-per-node=1 baseline_main.py conf_file=assets/conf/baseline/manas/manas_bench_long_control.yaml
# CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes=1 --nproc-per-node=1 baseline_main.py conf_file=assets/conf/baseline/manas/manas_bench_long_aux_0p1.yaml
# CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes=1 --nproc-per-node=1 baseline_main.py conf_file=assets/conf/baseline/manas/manas_bench_long_memtok_singlescale_512.yaml
# CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nnodes=1 --nproc-per-node=1 baseline_main.py conf_file=assets/conf/baseline/manas/manas_bench_v0p1_large_1024bs.yaml
# CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nnodes=1 --nproc-per-node=1 baseline_main.py conf_file=assets/conf/baseline/manas/manas_bench_v0p1_large_1024bs_aux_0p1.yaml
# CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nnodes=1 --nproc-per-node=1 baseline_main.py conf_file=assets/conf/baseline/manas/manas_bench_v0p1_encreverse.yaml
# CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nnodes=1 --nproc-per-node=2 baseline_main.py conf_file=assets/conf/baseline/manas/manas_bench_v0p2_scaleattn_localaux.yaml
# CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nnodes=1 --nproc-per-node=1 baseline_main.py conf_file=assets/conf/baseline/manas/manas_bench_v0p2_scaleattn_localaux_nulltoken.yaml


QUEUE_FILE="jobs.txt"
LOG_FILE="queue.log"

touch "$QUEUE_FILE"

echo "Starting queue runner..."

while true; do
    if [ -s "$QUEUE_FILE" ]; then
        cmd=$(head -n 1 "$QUEUE_FILE")

        echo "====================================" | tee -a "$LOG_FILE"
        echo "Running: $cmd" | tee -a "$LOG_FILE"
        echo "Time: $(date)" | tee -a "$LOG_FILE"

        if eval "$cmd" >> "$LOG_FILE" 2>&1; then
            echo "Finished successfully" | tee -a "$LOG_FILE"
            sed -i '1d' "$QUEUE_FILE"
        else
            echo "Job failed. Keeping it in queue." | tee -a "$LOG_FILE"
            sleep 5
        fi
    else
        sleep 5
    fi
done
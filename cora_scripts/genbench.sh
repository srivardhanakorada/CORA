#!/bin/bash

# Define output log file
LOGFILE="logs/genbench_cora.log"

# Run the CoRA script with nohup
CUDA_VISIBLE_DEVICES=0 nohup python cora_src/main_cora_genbench.py \
    --csv_path gen_bench_40/gen_bench_40.csv \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
#!/bin/bash

# Define output log file
LOGFILE="logs/genbench_adavd.log"

# Run the CoRA script with nohup
CUDA_VISIBLE_DEVICES=1 nohup python adavd_src/main_adavd_genbench.py \
    --csv_path gen_bench_40/gen_bench_40.csv \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
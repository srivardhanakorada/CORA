#!/bin/bash

# Define output log file
LOGFILE="logs/adavd_single_trump.log"

# Run the AdaVD script with nohup
CUDA_VISIBLE_DEVICES=0 nohup python adavd_src/main_adavd.py \
  --mode "original,retain" \
  --erase_type simple \
  --target_concept "Donald Trump" \
  --contents "Donald Trump, President of the United States of America, Lemon, Dog" \
  --num_samples 1 --batch_size 1\
  --save_path "results/Donald Trump" \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
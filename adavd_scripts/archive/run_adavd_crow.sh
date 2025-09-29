#!/bin/bash

# Define output log file
LOGFILE="adavd_crow.log"

# Run the AdaVD script with nohup
CUDA_VISIBLE_DEVICES=2 nohup python adavd_src/main_adavd.py \
  --save_root outputs_adavd \
  --mode "original,retain" \
  --erase_type simple \
  --target_concept "Crow" \
  --contents "Crow" \
  --num_samples 10 --batch_size 10\
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
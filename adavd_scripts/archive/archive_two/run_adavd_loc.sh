#!/bin/bash

# Define output log file
LOGFILE="logs/adavd_single_trump.log"

# Run the AdaVD script with nohup
CUDA_VISIBLE_DEVICES=0 nohup python adavd_src/main_adavd.py \
  --mode "original,retain" \
  --erase_type combined_celebs \
  --target_concept "Tom Cruise" \
  --contents "Tom Cruise" \
  --num_samples 10 --batch_size 10\
  --save_path "outputs_adavd_single/loc/Tom Cruise" \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
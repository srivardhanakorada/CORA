#!/bin/bash

# Define output log file
LOGFILE="adavd_comb.log"

# Run the AdaVD script with nohup
CUDA_VISIBLE_DEVICES=1 nohup python adavd_src/main_adavd.py \
  --save_root outputs_adavd \
  --mode "original,retain" \
  --erase_type combined_celebs \
  --target_concept "Chris Hemsworth" \
  --contents "Chris Hemsworth, Taylor Swift" \
  --num_samples 10 --batch_size 10\
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
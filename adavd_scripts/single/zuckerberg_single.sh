#!/bin/bash

# Define output log file
LOGFILE="logs/zuckerberg_single_adavd.log"

# Run the AdaVD script with nohup
CUDA_VISIBLE_DEVICES=1 nohup python adavd_src/main_adavd.py \
  --mode "original,retain" \
  --erase_type celebs \
  --target_concept "Mark Zuckerberg" \
  --contents "Anne Hathaway, Tom Cruise" \
  --num_samples 100 --batch_size 10 \
  --save_path "results/adavd/zuckerberg" \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
#!/bin/bash

# Define output log file
LOGFILE="logs/trump_single_adavd.log"

# Run the AdaVD script with nohup
CUDA_VISIBLE_DEVICES=0 nohup python adavd_src/main_adavd.py \
  --mode "original,retain" \
  --erase_type celebs \
  --target_concept "Donald Trump" \
  --contents "President of the United States of America" \
  --num_samples 15 --batch_size 15 \
  --seed 10 \
  --save_path "results/adavd/temp" \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
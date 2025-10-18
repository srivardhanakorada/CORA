#!/bin/bash

# Define output log file
LOGFILE="logs/temp.log"

# Run the AdaVD script with nohup
CUDA_VISIBLE_DEVICES=1 nohup python adavd_src/main_adavd.py \
  --mode "original,retain" \
  --erase_type celebs \
  --target_concept "Mark Zuckerberg, Founder of Facebook" \
  --contents "Mark Zuckerberg, Founder of Facebook, CEO of Facebook ,Lemon, Dog" \
  --num_samples 10 --batch_size 10 \
  --save_path "results/adavd/par" \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
#!/bin/bash

# Define output log file
LOGFILE="logs/glasses_single_adavd.log"

# Run the AdaVD script with nohup
CUDA_VISIBLE_DEVICES=1 nohup python adavd_src/main_adavd.py \
  --mode "original,retain" \
  --erase_type celebs \
  --target_concept "Spectacles" \
  --contents "Spectacles, Reading-Glasses, Eye-Glasses, Train, Cat" \
  --num_samples 100 --batch_size 10 \
  --save_path "results/adavd/glasses" \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
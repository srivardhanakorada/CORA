#!/bin/bash

# Define output log file
LOGFILE="logs/brinjal_single_adavd.log"

# Run the AdaVD script with nohup
CUDA_VISIBLE_DEVICES=2 nohup python adavd_src/main_adavd.py \
  --mode "original,retain" \
  --erase_type celebs \
  --target_concept "Brinjal" \
  --contents "Brinjal, Aubergine, Eggplant, Train, Cat" \
  --num_samples 100 --batch_size 10 \
  --save_path "results/adavd/brinjal" \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
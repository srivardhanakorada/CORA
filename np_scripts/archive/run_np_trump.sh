#!/bin/bash

# Define output log file
LOGFILE="np_trump.log"

# Run the AdaVD script with nohup
CUDA_VISIBLE_DEVICES=2 nohup python np_src/main.py \
  --out_root "results/Donald_Trump_np" \
  --target_root "Donald Trump" \
  --names "Donald Trump" "President of United States of America" "Lemon" "Dog" \
  --negative_prompt "Donald Trump" \
  --num_samples 1 --steps 30 --guidance 7.5 --seed 0 \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
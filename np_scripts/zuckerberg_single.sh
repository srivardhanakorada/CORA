#!/bin/bash

# Define output log file
LOGFILE="logs/zuckerberg_single_np.log"

# Run the AdaVD script with nohup
CUDA_VISIBLE_DEVICES=0 nohup python np_src/main.py \
  --out_root "results/np/zuckerberg" \
  --names "Mark Zuckerberg" "Founder of Facebook" "CEO of Facebook" "Lemon" "Dog" \
  --negative_prompt "Mark Zuckerberg" \
  --num_samples 100 --steps 30 --guidance 7.5 --seed 0 \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
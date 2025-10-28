#!/bin/bash

# Define output log file
LOGFILE="logs/trump_single_np.log"

# Run the AdaVD script with nohup
CUDA_VISIBLE_DEVICES=0 nohup python np_src/main.py \
  --out_root "results/np/trump" \
  --names "Anne Hathaway" "Tom Cruise" \
  --negative_prompt "Donald Trump" \
  --num_samples 100 --steps 30 --guidance 7.5 --seed 0 \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
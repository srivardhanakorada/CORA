#!/bin/bash

# Define output log file
LOGFILE="logs/trump_single_np.log"

# Run the AdaVD script with nohup
CUDA_VISIBLE_DEVICES=2 nohup python np_src/main.py \
  --out_root "results/np/temp" \
  --names "President of the United States of America" \
  --negative_prompt "Donald Trump" \
  --num_samples 15 --steps 30 --guidance 7.5 --seed 10 --batch_size 15 \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
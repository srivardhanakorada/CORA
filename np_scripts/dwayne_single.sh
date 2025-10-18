#!/bin/bash

# Define output log file
LOGFILE="logs/dwayne_single_np.log"

# Run the AdaVD script with nohup
CUDA_VISIBLE_DEVICES=0 nohup python np_src/main.py \
  --out_root "results/np/dwayne" \
  --names "Dwayne Johnson" "Actor who is also known as the The Rock" "WWE fighter known as The Rock" "Lemon" "Dog" \
  --negative_prompt "Dwayne Johnson" \
  --num_samples 100 --steps 30 --guidance 7.5 --seed 0 \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
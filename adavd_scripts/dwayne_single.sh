#!/bin/bash

# Define output log file
LOGFILE="logs/dwayne_single_adavd.log"

# Run the AdaVD script with nohup
CUDA_VISIBLE_DEVICES=2 nohup python adavd_src/main_adavd.py \
  --mode "original,retain" \
  --erase_type celebs \
  --target_concept "Dwayne Johnson" \
  --contents "Dwayne Johnson, Actor who is also known as the The Rock, WWE fighter known as The Rock, Lemon, Dog" \
  --num_samples 100 --batch_size 10 \
  --save_path "results/adavd/dwayne" \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
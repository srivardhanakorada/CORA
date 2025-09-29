#!/bin/bash

# Define output log file
LOGFILE="adavd_spider.log"

# Run the AdaVD script with nohup
CUDA_VISIBLE_DEVICES=2 nohup python src/main.py \
  --save_root outputs_adavd \
  --mode "original,retain" \
  --erase_type simple \
  --target_concept "Spider Man" \
  --contents "Spider Man, Iron Man, Venom, Bat Man, Super Man, Marvel superhero who got bitten by a radio active spider wearing red and blue suit" \
  --num_samples 1000 --batch_size 10 \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
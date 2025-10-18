#!/bin/bash

# Define output log file
LOGFILE="logs/dog_single_adavd.log"

# Run the AdaVD script with nohup
CUDA_VISIBLE_DEVICES=0 nohup python adavd_src/main_adavd.py \
  --mode "original,retain" \
  --erase_type celebs \
  --target_concept "Dog" \
  --contents "Dog, Animal known as man's best friend, Pet known as man's best friend, Train, Cat" \
  --num_samples 100 --batch_size 10 \
  --save_path "results/adavd/dog" \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
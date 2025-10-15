#!/bin/bash

# Define output log file
LOGFILE="logs/adavd_single_dog.log"

# Run the AdaVD script with nohup
CUDA_VISIBLE_DEVICES=0 nohup python adavd_src/main_adavd.py \
  --mode "original,retain" \
  --erase_type simple \
  --target_concept "Dog" \
  --contents "Dog, Cat, Lion, Wolf, Rat, Cow, Goat" \
  --num_samples 1000 --batch_size 10\
  --save_path "outputs_adavd/Dog" \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
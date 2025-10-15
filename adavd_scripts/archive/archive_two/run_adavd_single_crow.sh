#!/bin/bash

# Define output log file
LOGFILE="logs/adavd_single_crow.log"

# Run the AdaVD script with nohup
CUDA_VISIBLE_DEVICES=1 nohup python adavd_src/main_adavd.py \
  --mode "original,retain" \
  --erase_type simple \
  --target_concept "Crow" \
  --contents "Crow, Eagle, Raven, Vulture, Parrot, Sparrow" \
  --num_samples 1000 --batch_size 10\
  --save_path "outputs_adavd/Crow" \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
#!/bin/bash

# Define output log file
LOGFILE="logs/adavd_single_apple.log"

# Run the AdaVD script with nohup
CUDA_VISIBLE_DEVICES=2 nohup python adavd_src/main_adavd.py \
  --mode "original,retain" \
  --erase_type simple \
  --target_concept "an Apple" \
  --contents "an Apple, an Orange, a Pineapple, a Custard apple, a Banana, a Jackfruit, a Lemon" \
  --num_samples 1000 --batch_size 10\
  --save_path "outputs_adavd/Apple" \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
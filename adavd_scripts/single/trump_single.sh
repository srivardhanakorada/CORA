#!/bin/bash

# Define output log file
LOGFILE="logs/trump_single_adavd.log"

# Run the AdaVD script with nohup
CUDA_VISIBLE_DEVICES=0 nohup python adavd_src/main_adavd.py \
  --mode "original,retain" \
  --erase_type celebs \
  --target_concept "Donald Trump" \
  --contents "Donald Trump, President of the United States of America, Husband of Melania Trump ,Lemon, Dog" \
  --num_samples 100 --batch_size 10 \
  --save_path "results/adavd/trump" \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
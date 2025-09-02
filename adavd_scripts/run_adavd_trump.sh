#!/bin/bash

# Define output log file
LOGFILE="adavd_trump.log"

# Run the AdaVD script with nohup
CUDA_VISIBLE_DEVICES=1 nohup python src/main.py \
  --save_root outputs_adavd \
  --mode "original,retain" \
  --erase_type simple \
  --target_concept "Donald Trump" \
  --contents "Donald Trump, Elon Musk, Bill Gates, Sachin Tendulkar, Barack Obama, President of United States of America" \
  --num_samples 1000 --batch_size 10\
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
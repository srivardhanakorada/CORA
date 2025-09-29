#!/bin/bash

# Define output log file
LOGFILE="logs/adavd_single_chris.log"

# Run the AdaVD script with nohup
CUDA_VISIBLE_DEVICES=1 nohup python adavd_src/main_adavd.py \
  --mode "original,retain" \
  --erase_type simple \
  --target_concept "Chris Evans" \
  --contents "Chris Evans, Chris Hemsworth, Sebastian Stan, Tom Hiddleston, Sachin Tendulkar, Anne Hathaway, Bill Gates, President of the United States of America" \
  --num_samples 1000 --batch_size 10\
  --save_path "outputs_adavd/Chris Evans" \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
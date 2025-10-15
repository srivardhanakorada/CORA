#!/bin/bash

# Define output log file
LOGFILE="logs/adavd_single_taylor.log"

# Run the AdaVD script with nohup
CUDA_VISIBLE_DEVICES=2 nohup python adavd_src/main_adavd.py \
  --mode "original,retain" \
  --erase_type simple \
  --target_concept "Taylor Swift" \
  --contents "Taylor Swift, Selena Gomez, Ed Sheeran, Ariana Grande, Sachin Tendulkar, Anne Hathaway, Bill Gates, President of United States of America" \
  --num_samples 1000 --batch_size 10\
  --save_path "outputs_adavd/Taylor Swift" \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
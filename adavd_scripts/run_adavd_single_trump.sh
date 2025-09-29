#!/bin/bash

# Define output log file
LOGFILE="logs/adavd_single_trump.log"

# Run the AdaVD script with nohup
CUDA_VISIBLE_DEVICES=0 nohup python adavd_src/main_adavd.py \
  --mode "original,retain" \
  --erase_type simple \
  --target_concept "Donald Trump" \
  --contents "Donald Trump, Tom Cruise, Elon Musk, Barack Obama, Sachin Tendulkar, Anne Hathaway, Bill Gates, President of the United States of America" \
  --num_samples 1000 --batch_size 10\
  --save_path "outputs_adavd/Donald Trump" \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
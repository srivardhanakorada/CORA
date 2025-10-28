#!/bin/bash

# Define output log file
LOGFILE="logs/melania_single_adavd.log"

# Run the AdaVD script with nohup
CUDA_VISIBLE_DEVICES=0 nohup python adavd_src/main_adavd.py \
  --mode "original,retain" \
  --erase_type celebs \
  --target_concept "Melania Trump" \
  --contents "Bruce Lee, Marilyn Monroe, Anne Hathaway, Tom Cruise, Donald Trump's Wife who is the first of United States of America" \
  --num_samples 100 --batch_size 10 \
  --save_path "results/adavd/melania" \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
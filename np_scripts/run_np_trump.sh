#!/bin/bash

# Define output log file
LOGFILE="np_trump.log"

# Run the AdaVD script with nohup
CUDA_VISIBLE_DEVICES=2 nohup python negative_prompting/main.py \
  --out_root "outputs_np" \
  --target_root "Donald Trump" \
  --names "Donald Trump" "Bill Gates" "Elon Musk" "President of United States of America" "Sachin Tendulkar" "Barack Obama" \
  --negative_prompt "Donald Trump" \
  --num_samples 1000 --steps 30 --guidance 7.5 --seed 0 \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
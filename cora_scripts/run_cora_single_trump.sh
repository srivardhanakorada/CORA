#!/bin/bash

# Define output log file
LOGFILE="logs/cora_single_trump.log"

# Run the CoRA script with nohup
CUDA_VISIBLE_DEVICES=0 nohup python cora_src/main_cora_anc.py \
  --save_root outputs_cora_anc \
  --mode "original,erase" \
  --erase_type simple \
  --target_concept "Donald Trump" \
  --anchor_pool "a man, a person, a politician, a celebrity" \
  --preserve_concepts "Elon Musk, Barack Obama" \
  --contents "Donald Trump, Tom Cruise, Elon Musk, Barack Obama, Sachin Tendulkar, Anne Hathaway, Bill Gates, President of the United States of America" \
  --beta 0.5 --tau 0.1 \
  --num_samples 1000 --batch_size 10 --total_timesteps 30 \
  --guidance_scale 7.5 \
  --xformers --tf32 \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
#!/bin/bash

# Define output log file
LOGFILE="cora_trump.log"

# Run the CoRA script with nohup
CUDA_VISIBLE_DEVICES=0 nohup python src/main_cora_three.py \
  --save_root outputs_cora \
  --mode "original,erase" \
  --erase_type simple \
  --target_concept "Donald Trump" \
  --anchor_concept "a man" \
  --preserve_concepts "Elon Musk, Barack Obama" \
  --contents "Donald Trump, Elon Musk, Bill Gates, Sachin Tendulkar, Barack Obama, President of United States of America" \
  --beta 0.5 --tau 0.1 \
  --num_samples 1000 --batch_size 10 --total_timesteps 30 \
  --guidance_scale 7.5 \
  --xformers --tf32 \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
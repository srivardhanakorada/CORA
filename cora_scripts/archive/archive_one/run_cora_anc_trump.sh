#!/bin/bash

# Define output log file
LOGFILE="cora_trump.log"

# Run the CoRA script with nohup
CUDA_VISIBLE_DEVICES=2 nohup python src/main_cora_four.py \
  --save_root outputs_cora_anc \
  --mode "original,erase" \
  --erase_type simple \
  --target_concept "Donald Trump" \
  --anchor_pool "a person, a politician, a celebrity" \
  --preserve_concepts "Elon Musk, Barack Obama" \
  --contents "Donald Trump, Barack Obama, Elon Musk, President of United States of America, Sachin Tendulkar, Bill Gates" \
  --num_samples 1000 --batch_size 10 \
  --total_timesteps 30 --guidance_scale 7.5 \
  --beta 0.5 --tau 0.1 \
  --xformers --tf32 --debug \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
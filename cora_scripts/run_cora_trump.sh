#!/bin/bash

# Define output log file
LOGFILE="cora_trump.log"

# Run the CoRA script with nohup
CUDA_VISIBLE_DEVICES=1 nohup python src/main_cora_three.py \
  --save_root temp \
  --mode "original,erase" \
  --erase_type simple \
  --target_concept "Donald Trump" \
  --anchor_concept "a man" \
  --preserve_concepts "Barack Obama, Joe Biden, Kamala Harris" \
  --contents "Donald Trump, Barack Obama, Joe Biden, Sachin Tendulkar, President of the United States of America" \
  --beta 0.5 --tau 0.1 \
  --num_samples 10 --batch_size 5 --total_timesteps 30 \
  --guidance_scale 7.5 \
  --xformers --tf32 \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"

# CUDA_VISIBLE_DEVICES=0 nohup python src/main_cora_two.py \
#   --save_root outputs_cora_sp \
#   --mode "original,erase" \
#   --erase_type simple \
#   --target_concept "Donald Trump" \
#   --anchor_concept "Sachin Tendulkar" \
#   --preserve_concepts "Barack Obama, Joe Biden, Kamala Harris" \
#   --contents "Donald Trump, Joe Biden, Sachin Tendulkar, President of the United States of America" \
#   --beta 0.5 --tau 0.1 \
#   --num_samples 10 --batch_size 5 --total_timesteps 30 --guidance_scale 7.5 > "$LOGFILE" 2>&1 &
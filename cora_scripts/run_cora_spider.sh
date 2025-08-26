#!/bin/bash

# Define output log file
LOGFILE="cora_spider.log"

# Run the CoRA script with nohup
CUDA_VISIBLE_DEVICES=1 nohup python src/main_cora_two.py \
  --save_root outputs_cora \
  --mode "original,erase" \
  --erase_type simple \
  --target_concept "Spider Man" \
  --anchor_concept "a man" \
  --preserve_concepts "Iron Man, Captain America, Thor" \
  --contents "Spider Man, Marvel superhero who was bitten by a radioactive spider wearing a red-and-blue suit, Iron Man, Lemon" \
  --beta 0.5 --tau 0.1 \
  --num_samples 10 --batch_size 5 --total_timesteps 30 --guidance_scale 7.5 \
  > "$LOGFILE" 2>&1 &
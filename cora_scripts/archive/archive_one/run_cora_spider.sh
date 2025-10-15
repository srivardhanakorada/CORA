#!/bin/bash

# Define output log file
LOGFILE="cora_spider.log"

# Run the CoRA script with nohup
CUDA_VISIBLE_DEVICES=0 nohup python src/main_cora_three.py \
  --save_root outputs_cora \
  --mode "original,erase" \
  --erase_type simple \
  --target_concept "Spider Man" \
  --anchor_concept "a man" \
  --preserve_concepts "Iron Man, Venom, Super Man, Bat Man" \
  --contents "Spider Man, Iron Man, Venom, Bat Man, Super Man, Marvel superhero who got bitten by a radio active spider wearing red and blue suit" \
  --beta 0.5 --tau 0.1 \
  --num_samples 1000 --batch_size 20 --total_timesteps 30 --guidance_scale 7.5 \
  --xformers --tf32 --debug \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
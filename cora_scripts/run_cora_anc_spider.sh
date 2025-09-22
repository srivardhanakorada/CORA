#!/bin/bash

# Define output log file
LOGFILE="cora_spider_anc.log"

# Run the CoRA script with nohup
CUDA_VISIBLE_DEVICES=1 nohup python src/main_cora_four.py \
  --save_root outputs_cora_anc \
  --mode "original,erase" \
  --erase_type simple \
  --target_concept "Spider Man" \
  --anchor_pool "a person, a superhero, a comic-book character" \
  --preserve_concepts "Iron Man, Venom, Super Man, Bat Man" \
  --contents "Spider Man, Iron Man, Venom, Bat Man, Super Man, Marvel superhero who got bitten by a radio active spider wearing red and blue suit" \
  --num_samples 1000 --batch_size 20 \
  --total_timesteps 30 --guidance_scale 7.5 \
  --beta 0.5 --tau 0.1 \
  --xformers --tf32 --debug \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
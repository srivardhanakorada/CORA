#!/bin/bash

# Define output log file
LOGFILE="logs/brinjal_single_cora.log"

# Run the CoRA script with nohup
CUDA_VISIBLE_DEVICES=2 nohup python cora_src/main_cora_anc.py \
  --mode "original,erase" \
  --erase_type celebs \
  --target_concept "Brinjal" \
  --anchor_pool "Tomato" \
  --preserve_concepts "Vegetables, Fruits" \
  --contents "Brinjal, Aubergine, Eggplant, Train, Cat" \
  --beta 0.5 --tau 0.1 \
  --num_samples 100 --batch_size 10 --total_timesteps 30 \
  --guidance_scale 7.5 \
  --xformers --tf32 \
  --debug \
  --save_path "results/cora/brinjal" \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
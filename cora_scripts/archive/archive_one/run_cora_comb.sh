#!/bin/bash

# Define output log file
LOGFILE="cora_comb.log"

# Run the CoRA script with nohup
CUDA_VISIBLE_DEVICES=0 nohup python cora_src/main_cora.py \
  --save_root outputs_cora \
  --mode "original,erase" \
  --erase_type combined_celebs \
  --target_concept "Chris Hemsworth" \
  --anchor_concept "Sachin Tendulkar" \
  --preserve_concepts "Chris Evans" \
  --contents "Chris Hemsworth, Taylor Swift" \
  --beta 0.5 --tau 0.1 \
  --num_samples 10 --batch_size 10 --total_timesteps 30 \
  --guidance_scale 7.5 \
  --xformers --tf32 \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
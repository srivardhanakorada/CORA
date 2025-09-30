#!/bin/bash

# Define output log file
LOGFILE="logs/cora_single_spider_gen.log"

# Run the CoRA script with nohup
CUDA_VISIBLE_DEVICES=1 nohup python cora_src/main_cora_anc.py \
  --mode "original,erase" \
  --erase_type simple \
  --target_concept "Spider Man" \
  --anchor_pool "Bat Man" \
  --preserve_concepts "Iron Man, Venom" \
  --contents "Venom" \
  --beta 0.5 --tau 0.1 \
  --num_samples 100 --batch_size 10 --total_timesteps 30 \
  --guidance_scale 7.5 \
  --xformers --tf32 \
  --debug \
  --save_path "outputs_cora_anc/Spider/gen" \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
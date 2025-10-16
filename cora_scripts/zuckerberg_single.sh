#!/bin/bash

# Define output log file
LOGFILE="logs/zuckerberg_single_cora.log"

# Run the CoRA script with nohup
CUDA_VISIBLE_DEVICES=1 nohup python cora_src/main_cora_anc.py \
  --mode "original,erase" \
  --erase_type celebs \
  --target_concept "Mark Zuckerberg" \
  --anchor_pool "Tom Cruise" \
  --preserve_concepts "Elon Musk, Bill Gates" \
  --contents "Mark Zuckerberg, Founder of Facebook, CEO of Facebook ,Lemon, Dog" \
  --beta 0.5 --tau 0.1 \
  --num_samples 100 --batch_size 10 --total_timesteps 30 \
  --guidance_scale 7.5 \
  --xformers --tf32 \
  --debug \
  --save_path "results/cora/zuckerberg" \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
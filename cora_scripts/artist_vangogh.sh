#!/bin/bash

# Define output log file
LOGFILE="logs/artist_vangogh_cora.log"

# Run the CoRA script with nohup
CUDA_VISIBLE_DEVICES=1 nohup python cora_src/main_cora_anc.py \
  --mode "original,erase" \
  --erase_type art \
  --target_concept "Van Gogh" \
  --anchor_pool "Picasso" \
  --preserve_concepts "Monet" \
  --contents "Van Gogh, Picasso, Monet, Andy Warhol, Caravaggio" \
  --beta 0.5 --tau 0.1 \
  --num_samples 10 --batch_size 10 --total_timesteps 30 \
  --guidance_scale 7.5 \
  --seed 0 \
  --xformers --tf32 \
  --debug \
  --save_path "results/cora/artist_vangogh" \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
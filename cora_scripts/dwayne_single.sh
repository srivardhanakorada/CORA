#!/bin/bash

# Define output log file
LOGFILE="logs/dwayne_single_cora.log"

# Run the CoRA script with nohup
CUDA_VISIBLE_DEVICES=2 nohup python cora_src/main_cora_anc.py \
  --mode "original,erase" \
  --erase_type celebs \
  --target_concept "Dwayne Johnson" \
  --anchor_pool "a man" \
  --preserve_concepts "Kevin Hart, Ata Johnson" \
  --contents "Dwayne Johnson, Actor who is also known as the The Rock, WWE fighter known as The Rock, Lemon, Dog" \
  --beta 0.5 --tau 0.1 \
  --num_samples 100 --batch_size 10 --total_timesteps 30 \
  --guidance_scale 7.5 \
  --xformers --tf32 \
  --debug \
  --save_path "results/cora/dwayne" \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
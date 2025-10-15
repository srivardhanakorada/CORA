#!/bin/bash

# Define output log file
LOGFILE="logs/cora_single_trump_int.log"

# Run the CoRA script with nohup
CUDA_VISIBLE_DEVICES=1 nohup python cora_src/main_cora_anc.py \
  --mode "original,erase" \
  --erase_type simple \
  --target_concept "Donald Trump" \
  --anchor_pool "Tom Cruise" \
  --preserve_concepts "Elon Musk, Barack Obama" \
  --contents "Donald Trump, President of the United States of America, Lemon, Dog" \
  --beta 0.5 --tau 0.1 \
  --num_samples 1 --batch_size 1 --total_timesteps 30 \
  --guidance_scale 7.5 \
  --xformers --tf32 \
  --debug \
  --save_path "results/Donald_Trump_cora" \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
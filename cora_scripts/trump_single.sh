#!/bin/bash

# Define output log file
LOGFILE="logs/trump_single_cora.log"

# Run the CoRA script with nohup
CUDA_VISIBLE_DEVICES=0 nohup python cora_src/main_cora_anc.py \
  --mode "original,erase" \
  --erase_type celebs \
  --target_concept "Donald Trump" \
  --anchor_pool "Tom Cruise" \
  --preserve_concepts "Elon Musk, Barack Obama" \
  --contents "Donald Trump, President of the United States of America, Husband of Melania Trump ,Lemon, Dog" \
  --beta 0.5 --tau 0.1 \
  --num_samples 100 --batch_size 10 --total_timesteps 30 \
  --guidance_scale 7.5 \
  --xformers --tf32 \
  --debug \
  --save_path "results/cora/trump" \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"
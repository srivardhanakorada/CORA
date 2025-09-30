#!/bin/bash

# Define output log file
LOGFILE="logs/cora_single_trump_int.log"

# Run the CoRA script with nohup
CUDA_VISIBLE_DEVICES=1 nohup python cora_src/main_cora_anc.py \
  --mode "original,erase" \
  --erase_type combined_celebs \
  --target_concept "Tom Cruise" \
  --anchor_pool "Chris Hemsworth" \
  --preserve_concepts "Elon Musk, Barack Obama" \
  --contents "Tom Cruise" \
  --beta 0.5 --tau 0.1 \
  --num_samples 10 --batch_size 10 --total_timesteps 30 \
  --guidance_scale 7.5 \
  --xformers --tf32 \
  --debug \
  --save_path "outputs_cora_anc/loc" \
  > "$LOGFILE" 2>&1 < /dev/null &
echo "PID: $!"